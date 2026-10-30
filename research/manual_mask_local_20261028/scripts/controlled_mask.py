"""Controlled MASK_FULL baseline around the native Qwen image-edit pipeline.

This is a project latent-blend wrapper, not the unmodified official inpaint
pipeline. It deliberately supports only one RGB image, one output, CFG=1,
eight deterministic FlowMatch Euler steps, fixed packed noise, and a binary
mask. A white mask retains the native update; a black mask restores the
source/noise trajectory *after* each scheduler update. There is no RGB copy.

Usage::

    pipe = ControlledMaskQwenImageEditPipeline.from_pretrained(path, ...)
    result = pipe(image=image, prompt=prompt, mask_image=mask,
                  initial_noise=fixed_packed_noise, num_inference_steps=8,
                  true_cfg_scale=1, trace_latents=True)
    manifest = pipe.last_mask_run.manifest()

The same scheduler, fused adapter, attention backend and explicit noise must
be used by NATIVE_FULL. ``trace_latents`` is for diagnosis; its CPU transfers
and the contract checks are not a performance-optimized execution path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit as native_module


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash dtype, shape and exact bytes, including BF16 without NumPy casting."""
    value = tensor.detach().contiguous().cpu()
    header = f"{value.dtype}:{tuple(value.shape)}:".encode()
    return hashlib.sha256(header + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains non-finite values")


def _binary_pixel_mask(mask_image: Any, original_size: tuple[int, int]) -> torch.Tensor:
    """Return CPU [1,1,H,W] bool; reject soft masks instead of thresholding."""
    if isinstance(mask_image, Image.Image):
        if mask_image.mode not in ("1", "L"):
            raise ValueError("mask_image must be a single-channel binary PIL mask")
        array = np.asarray(mask_image).copy()
        mask = torch.from_numpy(array)
    elif isinstance(mask_image, torch.Tensor):
        mask = mask_image.detach().cpu()
    elif isinstance(mask_image, np.ndarray):
        mask = torch.from_numpy(np.asarray(mask_image).copy())
    else:
        raise TypeError("mask_image must be PIL L/1, a 2D array, or a 2D tensor")
    if mask.ndim != 2:
        raise ValueError("mask_image must be 2D; batch/channel broadcasting is forbidden")
    if tuple(mask.shape) != (original_size[1], original_size[0]):
        raise ValueError("mask_image must use the original source image coordinates")
    _finite(mask.float(), "pixel mask")
    if mask.dtype != torch.bool:
        zero_one = ((mask == 0) | (mask == 1)).all().item()
        zero_255 = ((mask == 0) | (mask == 255)).all().item()
        if not (zero_one or zero_255):
            raise ValueError("soft masks are outside this binary-mask protocol")
    return mask.bool()[None, None]


def pack_binary_mask(
    mask_image: Any,
    original_size: tuple[int, int],
    target_height: int,
    target_width: int,
    vae_scale_factor: int,
    latent_channels: int,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resize nearest, then use the exact native C,sub_y,sub_x packing order.

    The returned packed bool mask has shape [1,N,4*C], not [1,N,1]. Thus
    partially selected 2x2 tokens preserve the correct latent subpositions.
    All-zero masks are permitted as a diagnostic; the pilot manifest must
    reject empty task selections separately, before any generation.
    """
    if min(target_height, target_width, latent_channels, vae_scale_factor) <= 0:
        raise ValueError("mask geometry must be positive")
    if target_height % (2 * vae_scale_factor) or target_width % (2 * vae_scale_factor):
        raise ValueError("target geometry must be divisible by VAE scale times packing scale")
    pixels = _binary_pixel_mask(mask_image, original_size)
    transformed = F.interpolate(pixels.float(), size=(target_height, target_width), mode="nearest").bool()
    latent = F.interpolate(
        transformed.float(),
        size=(target_height // vae_scale_factor, target_width // vae_scale_factor),
        mode="nearest",
    ).bool()
    height, width = latent.shape[-2:]
    channels = latent.expand(1, latent_channels, height, width).contiguous()
    packed = QwenImageEditPipeline._pack_latents(channels, 1, latent_channels, height, width)
    packed = packed.to(device=device)
    active = torch.nonzero(packed.any(dim=-1)[0], as_tuple=False).flatten()
    return transformed, latent, packed, active


def validate_compute_target_ids(packed_mask: torch.Tensor, indices: torch.Tensor) -> None:
    """Guard future MASK_LOCAL selectors: every selected subposition needs Q.

    Expansion may add compute tokens. It must never remove a token with even
    one white latent subposition. Full-anchor steps may pass all target ids.
    """
    if packed_mask.ndim != 3 or packed_mask.shape[0] != 1 or packed_mask.dtype != torch.bool:
        raise ValueError("packed_mask must be batch-one bool [1,N,4*C]")
    values = torch.as_tensor(indices, device=packed_mask.device)
    if values.ndim != 1 or values.dtype not in (torch.int32, torch.int64):
        raise ValueError("compute target ids must be a 1D integer tensor")
    if values.numel() and (values.min().item() < 0 or values.max().item() >= packed_mask.shape[1]):
        raise ValueError("compute target id is outside the target sequence")
    if values.unique().numel() != values.numel():
        raise ValueError("compute target ids contain duplicates")
    selected = torch.zeros(packed_mask.shape[1], device=packed_mask.device, dtype=torch.bool)
    selected[values.long()] = True
    if bool((packed_mask.any(-1)[0] & ~selected).any().item()):
        raise ValueError("compute selection omits a mask-bearing token")


def blend_after_euler(
    updated: torch.Tensor,
    source: torch.Tensor,
    noise: torch.Tensor,
    packed_mask: torch.Tensor,
    sigma_next: torch.Tensor | float,
) -> torch.Tensor:
    """Binary fusion at the *next actual shifted sigma*, without RNG use."""
    if source.shape != updated.shape or noise.shape != updated.shape or packed_mask.shape != updated.shape:
        raise ValueError("updated/source/noise/mask must have exactly identical packed shapes")
    if packed_mask.dtype != torch.bool or packed_mask.device != updated.device:
        raise ValueError("packed mask must be bool on the latent device")
    for name, value in (("source", source), ("noise", noise), ("updated", updated)):
        if value.dtype != updated.dtype or value.device != updated.device:
            raise ValueError(f"{name} dtype/device differs from updated")
        _finite(value, name)
    # Keep the scheduler's FP32 sigma. Casting sigma to BF16 would round the
    # schedule before multiplication; casting only the final background also
    # differs from native BF16 pointwise semantics. The expression below is
    # the declared project background path, with FP32 arithmetic then one cast.
    sigma = torch.as_tensor(sigma_next, device=updated.device, dtype=torch.float32)
    if sigma.numel() != 1 or not torch.isfinite(sigma).all().item() or not 0 <= sigma.item() <= 1:
        raise ValueError("sigma_next must be one finite value in [0,1]")
    if sigma.item() == 0:
        background = source
    else:
        background = (sigma * noise.float() + (1 - sigma) * source.float()).to(updated.dtype)
    return torch.where(packed_mask, updated, background)


@dataclass
class MaskRunRecord:
    """Request-specific states. Tensor fields are diagnostic snapshots."""

    target_height: int
    target_width: int
    pixel_mask: torch.Tensor
    latent_mask: torch.Tensor
    packed_mask: torch.Tensor
    active_target_ids: torch.Tensor
    source: torch.Tensor
    ref_cond: torch.Tensor
    noise: torch.Tensor
    sigmas: list[float] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    step_latents: list[torch.Tensor] = field(default_factory=list)
    dit_calls: int = 0
    completed: bool = False

    def manifest(self) -> dict[str, Any]:
        return {
            "protocol": "project_MASK_FULL_native_post_euler_binary_blend_v1",
            "completed": self.completed,
            "target_height": self.target_height,
            "target_width": self.target_width,
            "latent_shape": list(self.source.shape),
            "dtype": str(self.source.dtype),
            "active_target_ids": self.active_target_ids.detach().cpu().tolist(),
            "active_target_count": self.active_target_ids.numel(),
            "target_count": self.source.shape[1],
            "all_white": bool(self.packed_mask.all().item()),
            "all_black": not bool(self.packed_mask.any().item()),
            "sigmas": self.sigmas,
            "dit_calls": self.dit_calls,
            "steps": self.steps,
            "tensor_sha256": {
                name: tensor_digest(getattr(self, name))
                for name in ("pixel_mask", "latent_mask", "packed_mask", "source", "ref_cond", "noise")
            },
            "background_arithmetic": "FP32 shifted sigma/noise/source mixture, cast once to latent dtype; zero sigma uses source",
            "rgb_copy": False,
        }


class ControlledMaskQwenImageEditPipeline(QwenImageEditPipeline):
    """A narrowly constrained native subclass; the original denoiser is reused."""

    def prepare_latents(
        self, image, batch_size, num_channels_latents, height, width,
        dtype, device, generator, latents=None,
    ):
        context = getattr(self, "_controlled_mask_context", None)
        if context is None:
            raise RuntimeError("use the controlled __call__ entry point")
        if batch_size != 1 or latents is None:
            raise ValueError("batch=1 with explicit packed initial_noise is required")
        if latents.dtype != dtype or latents.device != torch.device(device):
            raise ValueError("initial_noise must already have the exact prompt dtype and execution device")
        result = super().prepare_latents(
            image, batch_size, num_channels_latents, height, width,
            dtype, device, generator, latents=latents,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("native prepare_latents contract changed; expected (noise, ref_cond)")
        initial, ref_cond = result
        if ref_cond is None or initial.ndim != 3 or initial.shape != ref_cond.shape:
            raise ValueError("source/reference and target must use the identical packed grid")
        expected = (1, (height // (2 * self.vae_scale_factor)) * (width // (2 * self.vae_scale_factor)), 4 * num_channels_latents)
        if tuple(initial.shape) != expected:
            raise ValueError(f"initial_noise shape {tuple(initial.shape)} differs from target shape {expected}")
        _finite(initial, "initial_noise")
        _finite(ref_cond, "native ref_cond")
        if ref_cond.dtype != initial.dtype or ref_cond.device != initial.device:
            raise ValueError("native ref_cond dtype/device differs from target")
        pixels, latent, packed, active = pack_binary_mask(
            context["mask_image"], context["original_size"], height, width,
            self.vae_scale_factor, num_channels_latents, device=initial.device,
        )
        record = MaskRunRecord(
            target_height=height, target_width=width,
            pixel_mask=pixels.clone(), latent_mask=latent.clone(),
            packed_mask=packed, active_target_ids=active,
            source=ref_cond.detach().clone(), ref_cond=ref_cond.detach().clone(),
            noise=initial.detach().clone(),
        )
        context["record"] = record
        context["immutable"] = (record.source.clone(), record.ref_cond.clone(), record.noise.clone(), packed.clone())
        self.last_mask_run = record
        return initial, ref_cond

    def _check_schedule(self, record: MaskRunRecord) -> None:
        schedule = self.scheduler.sigmas
        if len(self.scheduler.timesteps) != 8 or schedule.numel() != 9:
            raise ValueError("this protocol requires exactly 8 model timesteps and 9 sigmas")
        _finite(schedule, "scheduler sigmas")
        if schedule[0].item() != 1 or schedule[-1].item() != 0 or not bool((schedule[:-1] > schedule[1:]).all().item()):
            raise ValueError("expected strictly decreasing sigmas from exactly 1 to exactly 0")
        record.sigmas = schedule.detach().float().cpu().tolist()

    @torch.no_grad()
    def __call__(
        self, image=None, prompt=None, *, mask_image, initial_noise: torch.Tensor,
        num_inference_steps: int = 8, true_cfg_scale: float = 1.0,
        trace_latents: bool = False, **kwargs,
    ):
        if getattr(self, "_controlled_mask_context", None) is not None:
            raise RuntimeError("overlapping/reentrant requests are forbidden")
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("this protocol requires one RGB PIL source image")
        if num_inference_steps != 8 or true_cfg_scale != 1:
            raise ValueError("this protocol fixes num_inference_steps=8 and true_cfg_scale=1")
        if kwargs.get("num_images_per_prompt", 1) != 1:
            raise ValueError("this protocol requires one output image")
        if isinstance(prompt, list) or (kwargs.get("prompt_embeds") is not None and kwargs["prompt_embeds"].shape[0] != 1):
            raise ValueError("this protocol requires a batch of exactly one prompt")
        if not isinstance(initial_noise, torch.Tensor) or initial_noise.ndim != 3 or initial_noise.shape[0] != 1:
            raise ValueError("initial_noise must be explicit packed [1,N,4*C] tensor")
        if not initial_noise.is_floating_point():
            raise ValueError("initial_noise must use a floating dtype")
        forbidden = ("latents", "callback_on_step_end", "callback_on_step_end_tensor_inputs", "strength")
        if any(name in kwargs for name in forbidden):
            raise ValueError(f"controlled wrapper owns these unsupported arguments: {forbidden}")
        if kwargs.get("negative_prompt") is not None or kwargs.get("negative_prompt_embeds") is not None:
            raise ValueError("negative prompt branches are outside the CFG1 protocol")
        if kwargs.get("guidance_scale") is not None or self.transformer.config.guidance_embeds:
            raise ValueError("guidance-distilled embedding is outside this protocol")
        if not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            raise ValueError("only deterministic FlowMatchEulerDiscreteScheduler is supported")
        if self.scheduler.config.get("stochastic_sampling", False) or self.scheduler.config.get("invert_sigmas", False):
            raise ValueError("stochastic/inverted scheduler is outside this protocol")
        calculated_width, calculated_height, _ = native_module.calculate_dimensions(1024 * 1024, image.width / image.height)
        multiple = 2 * self.vae_scale_factor
        calculated_width = calculated_width // multiple * multiple
        calculated_height = calculated_height // multiple * multiple
        for name, actual in (("height", calculated_height), ("width", calculated_width)):
            if kwargs.get(name) is not None and kwargs[name] != actual:
                raise ValueError("source/target grids differ; use the native calculated geometry")
            kwargs[name] = actual
        _binary_pixel_mask(mask_image, image.size)  # Reject invalid masks before any model call.
        self.last_mask_run = None
        context = {"mask_image": mask_image, "original_size": image.size, "trace": trace_latents}
        self._controlled_mask_context = context

        def before_transformer(module, args, forward_kwargs):
            record = context.get("record")
            if record is None:
                raise RuntimeError("transformer called before source/noise capture")
            if record.dit_calls == 0:
                self._check_schedule(record)
                if self.scheduler.step_index is not None:
                    raise RuntimeError("scheduler retained a previous request step index")
            if record.dit_calls >= 8:
                raise RuntimeError("unexpected extra DiT call")
            hidden = forward_kwargs.get("hidden_states")
            if hidden is None or hidden.shape != (1, 2 * record.source.shape[1], record.source.shape[2]):
                raise ValueError("transformer input is not target plus same-grid reference")
            _finite(hidden, "DiT input")
            if not torch.equal(hidden[:, record.source.shape[1]:], record.ref_cond):
                raise RuntimeError("reference condition changed within a request")
            if record.dit_calls == 0 and not torch.equal(hidden[:, :record.noise.shape[1]], record.noise):
                raise RuntimeError("native first target state differs from the explicit fixed noise")
            record.dit_calls += 1

        def after_step(pipe, index, timestep, callback_kwargs):
            record = context["record"]
            if index != len(record.steps) or record.dit_calls != index + 1:
                raise RuntimeError("scheduler callback/DiT order changed")
            if self.scheduler.step_index != index + 1:
                raise RuntimeError("callback must run after exactly one Euler scheduler step")
            updated = callback_kwargs["latents"]
            sigma_next = self.scheduler.sigmas[index + 1]
            merged = blend_after_euler(updated, record.source, record.noise, record.packed_mask, sigma_next)
            record.steps.append({
                "step": index, "timestep": float(timestep.item()),
                "scheduler_step_index": self.scheduler.step_index,
                "sigma_current": float(self.scheduler.sigmas[index].item()),
                "sigma_next": float(sigma_next.item()),
                "active_target_count": record.active_target_ids.numel(),
                "state_sha256": tensor_digest(merged) if trace_latents else None,
            })
            if trace_latents:
                record.step_latents.append(merged.detach().cpu().clone())
            return {"latents": merged}

        handle = self.transformer.register_forward_pre_hook(before_transformer, with_kwargs=True)
        try:
            result = super().__call__(
                image=image, prompt=prompt, latents=initial_noise.detach().clone(),
                true_cfg_scale=true_cfg_scale, num_inference_steps=num_inference_steps,
                callback_on_step_end=after_step,
                callback_on_step_end_tensor_inputs=["latents"], **kwargs,
            )
            record = context["record"]
            if record.dit_calls != 8 or len(record.steps) != 8:
                raise RuntimeError("request did not finish exactly eight DiT/Euler/fusion steps")
            for value, frozen in zip((record.source, record.ref_cond, record.noise, record.packed_mask), context["immutable"]):
                if not torch.equal(value, frozen):
                    raise RuntimeError("immutable source/reference/noise/mask changed within a request")
            record.completed = True
            return result
        finally:
            handle.remove()
            self._controlled_mask_context = None
