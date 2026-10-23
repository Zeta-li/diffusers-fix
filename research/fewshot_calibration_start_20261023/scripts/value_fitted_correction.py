import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("VALUE_CHANNEL_STAGE4_SCRIPTS", str(HERE))

from value_channel_correction import ValueChannelExecutor


def coefficient_payload_hash(alpha, delta, observed_pairs):
    digest = hashlib.sha256()
    for value in [alpha, delta, observed_pairs]:
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class ValueFittedExecutor(ValueChannelExecutor):
    def table_bytes(self):
        extras = [getattr(self, "observed_pairs", None), getattr(self, "_expected_model_timesteps", None)]
        return super().table_bytes() + sum(value.numel() * value.element_size() for value in extras if value is not None)

    def coefficient_contract(self, timesteps, *, model_identity):
        if not isinstance(model_identity, str) or not model_identity.strip():
            raise ValueError("A caller-verified model identity is required")
        model = self.model
        if model.training:
            raise ValueError("Fitted correction requires an eval model")
        if model.zero_cond_t or getattr(model.time_text_embed, "use_additional_t_cond", False):
            raise ValueError("Fitted correction requires ordinary timestep conditioning")
        if getattr(model.config, "guidance_embeds", False):
            raise ValueError("Guidance embedding models are unsupported")
        if any("lora_" in name for name, _ in model.named_parameters()):
            raise ValueError("Fuse and unload LoRA before preparing fitted correction")
        weight = model.img_in.weight
        raw = torch.as_tensor(timesteps, device=weight.device).reshape(-1)
        if raw.numel() != 8 or not bool(torch.isfinite(raw).all()):
            raise ValueError("Exactly eight finite scheduler timesteps are required")
        if not bool((raw[:-1] > raw[1:]).all()):
            raise ValueError("Scheduler timesteps must be strictly descending")
        layers = len(model.transformer_blocks)
        heads = model.transformer_blocks[0].attn.heads
        inner = model.transformer_blocks[0].attn.inner_dim
        if inner % heads:
            raise ValueError("Invalid attention head geometry")
        projections = []
        for block in model.transformer_blocks:
            attention = block.attn
            if attention.heads != heads or attention.inner_dim != inner:
                raise ValueError("All layers must share attention geometry")
            if tuple(attention.to_v.weight.shape) != (inner, model.inner_dim):
                raise ValueError("Unexpected raw Value projection geometry")
            projections.append(dict(weight_shape=list(attention.to_v.weight.shape),
                bias_shape=list(attention.to_v.bias.shape) if attention.to_v.bias is not None else None))
        return dict(schema_version=1, model_identity=model_identity, layers=layers,
            heads=heads, head_dim=inner // heads, input_width=model.inner_dim,
            model_dtype=str(weight.dtype), raw_scheduler_timesteps=raw.float().cpu().tolist(),
            model_timesteps=(raw.to(dtype=weight.dtype) / 1000).float().cpu().tolist(),
            raw_value_projections=projections,
            tensor_layout="layer,current_step,anchor_step,head,channel",
            indexing="zero_based", value_semantics="raw Value before attention; no RoPE",
            timestep_conversion="Cast scheduler timestep to model dtype, then divide by 1000")

    def _parameter_signature(self):
        return tuple((name, id(value), value.data_ptr(), value._version, tuple(value.shape),
                      str(value.dtype), str(value.device)) for name, value in self.model.named_parameters())

    def prepare_tables(self, timesteps, coefficients=None, *, model_identity=None):
        if coefficients is None or model_identity is None:
            raise ValueError("Route B requires fitted coefficients and model_identity; norm statistics are not accepted")
        return self.prepare_fitted_tables(timesteps, coefficients, model_identity=model_identity)

    def prepare_fitted_tables(self, timesteps, artifact, *, model_identity):
        if self.layer_cache:
            raise RuntimeError("Clear request caches before replacing fitted tables")
        required = {"alpha", "delta", "observed_pairs", "contract", "source", "source_hash", "training_image_ids"}
        if not isinstance(artifact, dict) or not required.issubset(artifact):
            raise ValueError("Incomplete fitted coefficient artifact")
        contract = self.coefficient_contract(timesteps, model_identity=model_identity)
        if artifact["contract"] != contract:
            raise ValueError("Model, geometry, dtype or exact timestep contract mismatch")
        if not isinstance(artifact["source"], str) or not artifact["source"].strip():
            raise ValueError("Coefficient provenance source is required")
        source_hash = artifact["source_hash"]
        if not isinstance(source_hash, str) or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
            raise ValueError("source_hash must be a lowercase SHA256 digest")
        image_ids = artifact["training_image_ids"]
        if not isinstance(image_ids, list) or not image_ids or any(not isinstance(v, str) or not v for v in image_ids):
            raise ValueError("Nonempty training_image_ids are required")
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("Duplicate training image identities")
        weight = self.model.img_in.weight
        device, dtype = weight.device, weight.dtype
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        shape = (contract["layers"], 8, 8, contract["heads"], contract["head_dim"])
        alpha = torch.as_tensor(artifact["alpha"], dtype=torch.float32, device=device).detach().clone()
        delta = torch.as_tensor(artifact["delta"], dtype=torch.float32, device=device).detach().clone()
        if tuple(alpha.shape) != shape or tuple(delta.shape) != shape:
            raise ValueError(f"alpha and delta must have shape {shape}")
        if not bool(torch.isfinite(alpha).all()) or not bool(torch.isfinite(delta).all()):
            raise ValueError("All supplied coefficients must be finite, including unseen pairs")
        observed = torch.as_tensor(artifact["observed_pairs"], device=device)
        if observed.dtype != torch.bool:
            raise ValueError("observed_pairs must be boolean")
        if tuple(observed.shape) == (8, 8):
            observed = observed[None].expand(contract["layers"], -1, -1)
        elif tuple(observed.shape) != (contract["layers"], 8, 8):
            raise ValueError("observed_pairs must have shape [8,8] or [layer,8,8]")
        observed = observed.clone()
        future = torch.triu(torch.ones((8, 8), device=device, dtype=torch.bool), diagonal=1)
        if bool(observed[:, future].any()):
            raise ValueError("Observed future-anchor pairs are not causal")
        raw_digest = coefficient_payload_hash(alpha, delta, observed)
        expected_digest = artifact.get("coefficient_sha256")
        if expected_digest is not None and expected_digest != raw_digest:
            raise ValueError("Coefficient content digest mismatch")
        alpha.masked_fill_(~observed[:, :, :, None, None], 1)
        delta.masked_fill_(~observed[:, :, :, None, None], 0)
        diagonal = torch.arange(8, device=device)
        alpha[:, diagonal, diagonal] = 1
        delta[:, diagonal, diagonal] = 0
        self.alpha, self.delta = alpha, delta
        self.projected_shifts, self.rho = None, None
        self.observed_pairs = observed
        self.fitted_contract = contract
        self._table_device, self._table_dtype = device, dtype
        self._loaded_parameter_signature = self._parameter_signature()
        self._expected_model_timesteps = torch.as_tensor(contract["model_timesteps"], device=device, dtype=dtype)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.table_metadata = dict(status="prepared", route="B_fitted_raw_value_affine",
            prepare_seconds=time.perf_counter() - started, table_bytes=self.table_bytes(),
            observed_mask_bytes=observed.numel() * observed.element_size(),
            table_dtype="torch.float32", alpha_shape=list(alpha.shape), delta_shape=list(delta.shape),
            model_device=str(device), model_dtype=str(dtype), contract=contract,
            source=artifact["source"], source_hash=source_hash, training_image_ids=list(image_ids),
            declared_training_image_count=len(image_ids), sample_count_verified=False,
            calibration_performed_here=False, raw_coefficient_sha256=raw_digest,
            effective_coefficient_sha256=coefficient_payload_hash(alpha, delta, observed),
            observed_pairs_per_layer=observed.flatten(1).sum(1).cpu().tolist(),
            unseen_pair_fallback="identity", diagonal_alpha=1.0, diagonal_delta=0.0,
            alpha_clipped=False, alpha_min=float(alpha.min()), alpha_max=float(alpha.max()),
            delta_min=float(delta.min()), delta_max=float(delta.max()),
            cache_write_semantics="Temporary read only; raw K/V and real anchor timestamps remain unchanged",
            model_identity_verification="Caller-verified frozen weight identity; full weights are not rehashed here",
            provenance_responsibility="Caller verifies source-level split, manifest hash, model identity and fitted label provenance",
            runtime_checks="Parameter object/storage/version signature at begin and exact model timestep on each corrected forward",
            interpretation="Direct fitted per-channel raw Value correction only; not norm-statistics reconstruction and no quality guarantee")
        return self.table_metadata

    def assert_schedule(self, timesteps, *, model_identity):
        if self.table_metadata is None or self.coefficient_contract(timesteps, model_identity=model_identity) != self.fitted_contract:
            raise ValueError("Fitted schedule or model identity mismatch")
        return True

    def begin(self, *args, **kwargs):
        if self.correction_mode != "none":
            if self.table_metadata is None:
                raise RuntimeError("Load fitted tables before enabling correction")
            if self._parameter_signature() != self._loaded_parameter_signature:
                raise RuntimeError("Model parameters changed after loading fitted tables")
        return super().begin(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.correction_mode != "none":
            if self.table_metadata is None or not 0 <= self.step < 8:
                raise RuntimeError("Fitted correction requires a prepared fresh eight-step request")
            timestep = kwargs.get("timestep", args[3] if len(args) > 3 else None)
            if timestep is None:
                raise ValueError("A model timestep is required")
            actual = torch.as_tensor(timestep, device=self._table_device, dtype=self._table_dtype).reshape(-1)
            expected = self._expected_model_timesteps[self.step].reshape(-1)
            if not torch.equal(actual, expected):
                raise ValueError("Runtime model timestep differs from fitted contract")
        return super().forward(*args, **kwargs)
