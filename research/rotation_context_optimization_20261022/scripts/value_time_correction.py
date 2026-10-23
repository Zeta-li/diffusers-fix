import time

import torch

from local_executor import LocalTargetExecutor


class ValueTimeExecutor(LocalTargetExecutor):
    def __init__(self, model):
        self.correction_mode = "none"
        self.alpha = None
        self.delta = None
        self.projected_shifts = None
        self.rho = None
        self.table_metadata = None
        self._table_device = None
        self._table_dtype = None
        super().__init__(model)

    def clear(self):
        super().clear()
        self._value_read_step = None
        self._value_ages_gpu = None
        self._request_correction_mode = None

    def begin(self, *args, **kwargs):
        if self.correction_mode not in {"none", "shift", "affine"}:
            raise ValueError("correction_mode must be none, shift or affine")
        if self.correction_mode != "none" and self.table_metadata is None:
            raise RuntimeError("prepare_tables must complete before enabling Value correction")
        super().begin(*args, **kwargs)
        self._request_correction_mode = self.correction_mode

    def table_bytes(self):
        return sum(value.numel() * value.element_size() for value in
                   [self.alpha, self.delta, self.projected_shifts, self.rho] if value is not None)

    def prepare_tables(self, timesteps):
        model = self.model
        if model.zero_cond_t or getattr(model.time_text_embed, "use_additional_t_cond", False):
            raise ValueError("Value pilot requires ordinary timestep conditioning")
        if getattr(model.config, "guidance_embeds", False):
            raise ValueError("Guidance embedding models are outside the Value pilot")
        if any("lora_" in name for name, _ in model.named_parameters()):
            raise ValueError("Prepare Value tables after fusing and unloading LoRA")
        weight = model.img_in.weight
        device, dtype = weight.device, weight.dtype
        raw_timesteps = torch.as_tensor(timesteps, device=device).reshape(-1)
        if raw_timesteps.numel() != 8:
            raise ValueError("Value pilot requires an eight-step fixed schedule")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        heads = model.transformer_blocks[0].attn.heads
        head_dim = model.transformer_blocks[0].attn.inner_dim // heads
        alphas, deltas, projected, residuals = [], [], [], []
        with torch.inference_mode():
            model_timesteps = raw_timesteps.to(dtype=dtype) / 1000
            dummy = torch.empty((1, 1, model.inner_dim), device=device, dtype=dtype)
            embeddings = [model.time_text_embed(value.reshape(1), dummy, None) for value in model_timesteps]
            for block in model.transformer_blocks:
                if block.attn.heads != heads or block.attn.inner_dim != heads * head_dim:
                    raise ValueError("Value pilot requires identical attention dimensions across layers")
                units, values = [], []
                for embedding in embeddings:
                    first, _ = block.img_mod(embedding).chunk(2, dim=-1)
                    shift, scale, _ = first.chunk(3, dim=-1)
                    units.append((scale + 1).squeeze(0).float())
                    values.append(block.attn.to_v(shift).reshape(heads, head_dim).float())
                units = torch.stack(units)
                values = torch.stack(values)
                alpha = (units @ units.transpose(0, 1)) / (units.square().sum(-1)[None, :] + 1e-12)
                diagonal = torch.arange(8, device=device)
                alpha[diagonal, diagonal] = 1.0
                delta = values[:, None] - alpha[:, :, None, None] * values[None, :]
                delta[diagonal, diagonal] = 0.0
                rho = (units[:, None] - alpha[:, :, None] * units[None, :]).norm(dim=-1) / units.norm(dim=-1)[:, None].clamp_min(1e-12)
                rho[diagonal, diagonal] = 0.0
                alphas.append(alpha)
                deltas.append(delta)
                projected.append(values)
                residuals.append(rho)
            self.alpha = torch.stack(alphas)
            self.delta = torch.stack(deltas)
            self.projected_shifts = torch.stack(projected)
            self.rho = torch.stack(residuals)
            finite = all(bool(torch.isfinite(value).all()) for value in [self.alpha, self.delta, self.projected_shifts, self.rho])
            if not finite:
                raise RuntimeError("Nonfinite Value correction tables")
            causal = torch.tril(torch.ones((8, 8), dtype=torch.bool, device=device))
            alpha_causal = self.alpha[:, causal]
            rho_causal = self.rho[:, causal]
            table_stats = dict(
                alpha_min=float(self.alpha.min()), alpha_max=float(self.alpha.max()),
                alpha_causal_min=float(alpha_causal.min()), alpha_causal_max=float(alpha_causal.max()),
                rho_min=float(self.rho.min()), rho_max=float(self.rho.max()),
                rho_causal_min=float(rho_causal.min()), rho_causal_max=float(rho_causal.max()),
                delta_min=float(self.delta.min()), delta_max=float(self.delta.max()), finite=finite,
                rho_causal_max_per_layer=self.rho[:, causal].max(dim=1).values.cpu().tolist(),
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._table_device = device
        self._table_dtype = dtype
        self.table_metadata = dict(
            status="prepared", prepare_seconds=time.perf_counter() - started,
            table_bytes=self.table_bytes(), table_dtype="torch.float32",
            alpha_shape=list(self.alpha.shape), delta_shape=list(self.delta.shape),
            projected_shift_shape=list(self.projected_shifts.shape), rho_shape=list(self.rho.shape),
            model_device=str(device), model_dtype=str(dtype),
            raw_scheduler_timesteps=raw_timesteps.float().cpu().tolist(),
            model_timesteps=model_timesteps.float().cpu().tolist(),
            timestep_conversion="scheduler timestep cast to model dtype, then divided by 1000, matching pipeline input",
            preparation_batch_size=1, preparation_timestep_calls=8,
            shift_projection_includes_linear_bias=True, alpha_clipped=False,
            diagonal_alpha=1.0, diagonal_delta=0.0,
            schedule_contract="Reuse only with the same frozen model weights, dtype, device and exact eight model timesteps; caller must verify the scheduler before each request",
            interpretation="Frozen hidden-state, scalar-scale approximation for attention Value only; no claim of exact current context or improved image quality",
            **table_stats,
        )
        return self.table_metadata

    def read_value_cache(self, layer_index, cv, selected):
        if self._request_correction_mode is not None and self.correction_mode != self._request_correction_mode:
            raise RuntimeError("Changing correction mode during a request requires begin")
        if self.correction_mode == "none" or selected.numel() == self.target_count:
            return cv
        if self.correction_mode not in {"shift", "affine"}:
            raise ValueError("Invalid Value correction mode")
        if self.table_metadata is None:
            raise RuntimeError("Value tables are not prepared")
        if cv.device != self._table_device or cv.dtype != self._table_dtype:
            raise RuntimeError("Value table device or dtype differs from cache")
        if self._value_read_step != self.step:
            ages = self.age.clone()
            selected_cpu = self.all_targets_cpu if self.step in self.full_steps else self.indices_cpu[self.step]
            ages.index_fill_(0, selected_cpu, self.step)
            if bool((ages < 0).any()) or bool((ages > self.step).any()):
                raise RuntimeError("Value correction requires initialized real cache timestamps")
            self._value_ages_gpu = ages.to(device=cv.device)
            self._value_read_step = self.step
        ages = self._value_ages_gpu
        if self.correction_mode == "affine":
            alpha = self.alpha[layer_index, self.step].index_select(0, ages)
            delta = self.delta[layer_index, self.step].index_select(0, ages)
            corrected = cv.float() * alpha[None, :, None, None] + delta[None]
        else:
            shifts = self.projected_shifts[layer_index]
            delta_by_age = shifts[self.step][None] - shifts
            delta = delta_by_age.index_select(0, ages)
            corrected = cv.float() + delta[None]
        return corrected.to(cv.dtype)
