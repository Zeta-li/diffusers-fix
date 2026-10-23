import hashlib
import os
import sys
import time
from pathlib import Path

import torch


DEFAULT_STAGE4 = Path(__file__).resolve().parents[2] / "背景轮转与上下文优化_20261022" / "scripts"
STAGE4_SCRIPTS = Path(os.environ.get("VALUE_CHANNEL_STAGE4_SCRIPTS", str(DEFAULT_STAGE4))).expanduser().resolve()
if not (STAGE4_SCRIPTS / "value_time_correction.py").is_file():
    raise ImportError(f"Set VALUE_CHANNEL_STAGE4_SCRIPTS to the stage-4 scripts directory: {STAGE4_SCRIPTS}")
sys.path.insert(0, str(STAGE4_SCRIPTS))

from value_time_correction import ValueTimeExecutor


RIDGE_RELATIVE = 1e-6
RIDGE_FLOOR = 1e-12


def channel_coefficients(weight, units, projected, mean, variance):
    weight, units, projected = weight.float(), units.float(), projected.float()
    mean, variance = mean.float(), variance.float()
    steps, width = units.shape
    if weight.ndim != 2 or weight.shape[1] != width or projected.shape != (steps, weight.shape[0]):
        raise ValueError("Channel coefficient geometry is inconsistent")
    if mean.shape != units.shape or variance.shape != units.shape:
        raise ValueError("Channel statistics must have shape timesteps by input channels")
    squared = weight.square()
    denominator = (variance * units.square()) @ squared.T
    pairs = units[:, None] * units[None] * variance[None]
    numerator = (pairs.reshape(steps * steps, width) @ squared.T).reshape(steps, steps, -1)
    regularizer = denominator.mean(dim=-1, keepdim=True) * RIDGE_RELATIVE + RIDGE_FLOOR
    alpha = (numerator + regularizer[None]) / (denominator[None] + regularizer[None])
    center_features = units[:, None] * mean[None]
    target_center = projected[:, None] + (center_features.reshape(steps * steps, width) @ weight.T).reshape(steps, steps, -1)
    anchor_center = projected + (units * mean) @ weight.T
    delta = target_center - alpha * anchor_center[None]
    target_variance_sum = (units[:, None].square() * variance[None] * squared.sum(dim=0)).sum(dim=-1)
    residual_sum = target_variance_sum + (alpha.square() * denominator[None] - 2 * alpha * numerator).sum(dim=-1)
    rho = (residual_sum.clamp_min(0) / target_variance_sum.clamp_min(RIDGE_FLOOR)).sqrt()
    diagonal = torch.arange(steps, device=weight.device)
    alpha[diagonal, diagonal] = 1
    delta[diagonal, diagonal] = 0
    rho[diagonal, diagonal] = 0
    return alpha, delta, rho, regularizer


class ValueChannelExecutor(ValueTimeExecutor):
    @property
    def mode(self):
        return self.correction_mode

    @mode.setter
    def mode(self, value):
        self.correction_mode = value

    def begin(self, *args, **kwargs):
        if self.correction_mode not in {"none", "affine"}:
            raise ValueError("Channel correction mode must be none or affine")
        return super().begin(*args, **kwargs)

    def _statistics(self, statistics, layers, width, device):
        if statistics is None:
            mean = torch.zeros((layers, 8, width), device=device, dtype=torch.float32)
            variance = torch.ones_like(mean)
            metadata = dict(statistics_mode="no_sample_mean_zero_variance_one", input_shapes=None,
                            caller_statistics_source=None, caller_statistics_source_hash=None)
        else:
            if not isinstance(statistics, dict) or "mean" not in statistics or "variance" not in statistics:
                raise ValueError("Statistics must supply mean and variance")
            arrays, shapes = {}, {}
            for name in ["mean", "variance"]:
                array = torch.as_tensor(statistics[name], device=device, dtype=torch.float32).detach()
                shapes[name] = list(array.shape)
                if array.shape == (layers, width):
                    array = array[:, None].expand(-1, 8, -1)
                elif array.shape != (layers, 8, width):
                    raise ValueError(f"{name} must have shape {(layers, width)} or {(layers, 8, width)}")
                if not bool(torch.isfinite(array).all()):
                    raise ValueError(f"Nonfinite {name} statistics")
                arrays[name] = array
            mean, variance = arrays["mean"], arrays["variance"]
            if bool((variance < 0).any()):
                raise ValueError("Variance statistics must be nonnegative population variances")
            source, source_hash = statistics.get("source"), statistics.get("source_hash")
            if any(value is not None and not isinstance(value, str) for value in [source, source_hash]):
                raise ValueError("Statistics source and source_hash must be strings when supplied")
            metadata = dict(statistics_mode="caller_provided_unverified_statistics", input_shapes=shapes,
                            caller_statistics_source=source, caller_statistics_source_hash=source_hash)
        digest = hashlib.sha256()
        for array in [mean, variance]:
            digest.update(array.contiguous().cpu().numpy().tobytes())
        metadata.update(statistics_float32_content_sha256=digest.hexdigest(),
                        calibration_performed_here=False, sample_count_verified=False,
                        provenance_responsibility="Caller supplies and verifies source hashes, calibration-only data and image-level split",
                        statistic_semantics="Target img_norm1 output before modulation; population variance, not post-modulation features",
                        timestep_statistics_semantics="For pair (t,s), the same frozen h=mean_s+xi_s is used at both timesteps with diagonal Var(xi_s)=variance_s; anchor mean and variance only")
        return mean, variance, metadata

    def prepare_tables(self, timesteps, statistics=None):
        model = self.model
        if model.zero_cond_t or getattr(model.time_text_embed, "use_additional_t_cond", False):
            raise ValueError("Channel pilot requires ordinary timestep conditioning")
        if getattr(model.config, "guidance_embeds", False):
            raise ValueError("Guidance embedding models are outside the channel pilot")
        if any("lora_" in name for name, _ in model.named_parameters()):
            raise ValueError("Prepare channel tables after fusing and unloading LoRA")
        weight = model.img_in.weight
        device, dtype = weight.device, weight.dtype
        raw_timesteps = torch.as_tensor(timesteps, device=device).reshape(-1)
        if raw_timesteps.numel() != 8 or not bool(torch.isfinite(raw_timesteps).all()):
            raise ValueError("Channel pilot requires eight finite fixed scheduler timesteps")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        heads = model.transformer_blocks[0].attn.heads
        head_dim = model.transformer_blocks[0].attn.inner_dim // heads
        layers, width = len(model.transformer_blocks), model.inner_dim
        with torch.inference_mode():
            mean, variance, statistics_metadata = self._statistics(statistics, layers, width, device)
            model_timesteps = raw_timesteps.to(dtype=dtype) / 1000
            dummy = torch.empty((1, 1, width), device=device, dtype=dtype)
            embeddings = [model.time_text_embed(value.reshape(1), dummy, None) for value in model_timesteps]
            alphas, deltas, projected, residuals, regularizers = [], [], [], [], []
            for layer, block in enumerate(model.transformer_blocks):
                if block.attn.heads != heads or block.attn.inner_dim != heads * head_dim:
                    raise ValueError("Channel pilot requires identical attention dimensions across layers")
                units, values = [], []
                for embedding in embeddings:
                    first, _ = block.img_mod(embedding).chunk(2, dim=-1)
                    shift, scale, _ = first.chunk(3, dim=-1)
                    units.append((scale + 1).squeeze(0).float())
                    values.append(block.attn.to_v(shift).reshape(heads * head_dim).float())
                units, values = torch.stack(units), torch.stack(values)
                alpha, delta, rho, ridge = channel_coefficients(block.attn.to_v.weight, units, values,
                                                               mean[layer], variance[layer])
                alphas.append(alpha.reshape(8, 8, heads, head_dim))
                deltas.append(delta.reshape(8, 8, heads, head_dim))
                projected.append(values.reshape(8, heads, head_dim))
                residuals.append(rho)
                regularizers.append(ridge)
            self.alpha, self.delta = torch.stack(alphas), torch.stack(deltas)
            self.projected_shifts, self.rho = torch.stack(projected), torch.stack(residuals)
            ridge = torch.stack(regularizers)
            finite = all(bool(torch.isfinite(value).all()) for value in
                         [self.alpha, self.delta, self.projected_shifts, self.rho, ridge])
            if not finite:
                raise RuntimeError("Nonfinite channel correction tables")
            causal = torch.tril(torch.ones((8, 8), dtype=torch.bool, device=device))
            stats = dict(alpha_min=float(self.alpha.min()), alpha_max=float(self.alpha.max()),
                         alpha_causal_min=float(self.alpha[:, causal].min()), alpha_causal_max=float(self.alpha[:, causal].max()),
                         rho_causal_min=float(self.rho[:, causal].min()), rho_causal_max=float(self.rho[:, causal].max()),
                         delta_min=float(self.delta.min()), delta_max=float(self.delta.max()),
                         regularizer_min=float(ridge.min()), regularizer_max=float(ridge.max()), finite=finite)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._table_device, self._table_dtype = device, dtype
        self.table_metadata = dict(
            status="prepared", prepare_seconds=time.perf_counter() - started, table_bytes=self.table_bytes(),
            table_dtype="torch.float32", alpha_shape=list(self.alpha.shape), delta_shape=list(self.delta.shape),
            projected_shift_shape=list(self.projected_shifts.shape), rho_shape=list(self.rho.shape),
            model_device=str(device), model_dtype=str(dtype), raw_scheduler_timesteps=raw_timesteps.float().cpu().tolist(),
            model_timesteps=model_timesteps.float().cpu().tolist(),
            timestep_conversion="Cast scheduler timestep to model dtype, then divide by 1000, matching pipeline input",
            preparation_batch_size=1, preparation_timestep_calls=8, shift_projection_includes_linear_bias=True,
            feature_mean_projection_has_bias=False, alpha_clipped=False, diagonal_alpha=1.0, diagonal_delta=0.0,
            ridge_relative=RIDGE_RELATIVE, ridge_floor=RIDGE_FLOOR, hyperparameter_search_performed=False,
            ridge_definition="lambda_s=1e-6*mean_output_channel(D_s)+1e-12; a=(N+lambda_s)/(D+lambda_s)",
            objective="Diagonal-covariance frozen-centered-feature MSE plus lambda_s*(a-1)^2 for each output channel",
            rho_definition="Square root of summed predicted centered Value residual variance divided by summed target variance; surrogate diagnostic only",
            schedule_contract="Reuse only with identical frozen fused weights, dtype, device, statistics and exact eight model timesteps; caller verifies scheduler",
            inference_cost="No extra token projection; eager per-token channel coefficient gathers add bandwidth and temporary memory",
            interpretation="Per-output-channel attention Value approximation; does not recover hidden drift, stale K or general cross-channel covariance; no quality claim",
            stage4_scripts=str(STAGE4_SCRIPTS), **statistics_metadata, **stats)
        return self.table_metadata

    def read_value_cache(self, layer_index, cv, selected):
        if self._request_correction_mode is not None and self.correction_mode != self._request_correction_mode:
            raise RuntimeError("Changing correction mode during a request requires begin")
        if self.correction_mode == "none" or selected.numel() == self.target_count:
            return cv
        if self.correction_mode != "affine":
            raise ValueError("Invalid channel correction mode")
        if self.table_metadata is None:
            raise RuntimeError("Channel tables are not prepared")
        if cv.device != self._table_device or cv.dtype != self._table_dtype:
            raise RuntimeError("Channel table device or dtype differs from cache")
        if self._value_read_step != self.step:
            ages = self.age.clone()
            selected_cpu = self.all_targets_cpu if self.step in self.full_steps else self.indices_cpu[self.step]
            ages.index_fill_(0, selected_cpu, self.step)
            if bool((ages < 0).any()) or bool((ages > self.step).any()):
                raise RuntimeError("Channel correction requires initialized real cache timestamps")
            self._value_ages_gpu = ages.to(device=cv.device)
            self._value_read_step = self.step
        ages = self._value_ages_gpu
        alpha = self.alpha[layer_index, self.step].index_select(0, ages)
        delta = self.delta[layer_index, self.step].index_select(0, ages)
        return (cv.float() * alpha[None] + delta[None]).to(cv.dtype)
