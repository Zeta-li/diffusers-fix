"""Native Qwen editing with an optional ROI *compute hint*.

There is no image mask, image crop, latent background replacement, final
pixel copy, or extra reference image in this module. A source bounding box
and an optional anticipated target envelope only choose which target rows
are freshly computed. The native scheduler updates ALL target latents at
every step, including rows whose velocity/KV are reused by the executor.

All boxes use [x0,y0,x1,y1) in the ORIGINAL source image's pixel canvas.
The target envelope can be larger than the original object, e.g. cat ->
giraffe, and must be frozen before viewing generated outputs. It is a
compute-location prior, never a permitted-change boundary.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import math
from numbers import Real
from typing import Any

import torch
from PIL import Image

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit as native_module

from legacy_refresh_policy import BudgetController, make_partition
from local_mask_bridge import install_local_executor


def _digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(
        f"{value.dtype}:{tuple(value.shape)}:".encode()
        + value.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains non-finite values")


def _dimensions(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(type(v) is not int or v <= 0 for v in value):
        raise ValueError(f"{name} must contain two positive integers")
    return tuple(value)


def _box(value, source_wh, name):
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{name} must contain four pixel coordinates")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(float(v)) for v in value):
        raise ValueError(f"{name} must contain finite numeric coordinates")
    x0, y0, x1, y1 = map(float, value)
    w, h = source_wh
    if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
        raise ValueError(f"{name} must be a nonempty box inside the source canvas")
    return [x0, y0, x1, y1]


def _box_token_ids(box, source_wh, grid_hw):
    """Conservative area-intersection mapping, independent of image content."""
    w, h = source_wh
    gh, gw = grid_hw
    x0, y0, x1, y1 = box
    left = max(0, min(gw - 1, math.floor(x0 * gw / w)))
    top = max(0, min(gh - 1, math.floor(y0 * gh / h)))
    right = min(gw, max(left + 1, math.ceil(x1 * gw / w)))
    bottom = min(gh, max(top + 1, math.ceil(y1 * gh / h)))
    return [y * gw + x for y in range(top, bottom) for x in range(left, right)]


def build_region_plan(
    source_bbox_xyxy, source_wh, target_grid_hw, case_id, halo=2,
    target_envelope_xyxy=None, kind="front", *, packed_channels=64,
):
    """Freeze a JSON-serializable free-generation local-compute schedule.

    ``packed_channels=64`` is original Qwen's packed latent channel count;
    tiny CPU fixtures explicitly pass 8. ``front`` spends the historical
    six group units as [15,3,0,0,0,0]; ``all_selected`` is the executor's
    full-computation equivalence control. Neither option restricts pixels.
    """
    source_wh = _dimensions(source_wh, "source_wh")
    grid_hw = _dimensions(target_grid_hw, "target_grid_hw")
    bbox = _box(source_bbox_xyxy, source_wh, "source_bbox_xyxy")
    envelope = None if target_envelope_xyxy is None else _box(target_envelope_xyxy, source_wh, "target_envelope_xyxy")
    if type(halo) is not int or halo < 0:
        raise ValueError("halo must be a nonnegative integer")
    if type(packed_channels) is not int or packed_channels <= 0 or packed_channels % 4:
        raise ValueError("packed_channels must be a positive multiple of four")
    if kind not in ("front", "all_selected"):
        raise ValueError("kind must be front or all_selected")
    gh, gw = grid_hw
    source_ids = _box_token_ids(bbox, source_wh, grid_hw)
    target_ids = [] if envelope is None else _box_token_ids(envelope, source_wh, grid_hw)
    support = sorted(set(source_ids) | set(target_ids))
    core = sorted({
        y * gw + x for index in support
        for y in range(max(0, index // gw - halo), min(gh, index // gw + halo + 1))
        for x in range(max(0, index % gw - halo), min(gw, index % gw + halo + 1))
    })
    n = gh * gw
    full = list(range(n))
    partition = None
    fallback = None
    if kind == "all_selected":
        selected = [full.copy() for _ in range(8)]
    else:
        partition = make_partition(str(case_id), list(grid_hw), core)
        if not partition["rl_eligible"]:
            fallback = "fewer_than_four_noncore_tokens"
            selected = [full.copy() for _ in range(8)]
        else:
            controller = BudgetController(partition)
            actions = [15, 3, 0, 0, 0, 0]
            selected = [controller.select(k, None if k in (0, 7) else actions[k - 1]) for k in range(8)]
    return {
        "schema_version": 1,
        "region_protocol": "native_free_generation_roi_compute_hint_v1",
        "case_id": str(case_id), "kind": kind, "halo": halo,
        "source_wh": list(source_wh), "source_bbox_xyxy": bbox,
        "target_envelope_xyxy": envelope,
        "box_coordinate_frame": "original_source_pixel_canvas_xyxy_exclusive",
        "grid_hw": list(grid_hw), "target_count": n, "packed_channels": packed_channels,
        "source_bbox_indices": source_ids, "target_envelope_indices": target_ids,
        "support_indices": support, "core_indices": core,
        "selected_indices_per_step": selected,
        "target_tokens_per_step": list(map(len, selected)),
        "total_target_token_updates": sum(map(len, selected)),
        "partition": partition, "fallback_reason": fallback,
        "velocity_correction": "none", "halo_metric": "square_Chebyshev_token_grid",
        "roi_role": "compute_hint_only_not_generation_permission",
        "source_bbox_is_output_boundary": False,
        "target_envelope_is_output_boundary": False,
        "all_target_latents_scheduler_updated_each_step": True,
        "background_fusion": False, "image_crop": False,
        "rgb_copy": False, "extra_reference_image": False,
    }


def _validate_region_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("plan must be a frozen region-plan dictionary")
    try:
        expected = build_region_plan(
            plan["source_bbox_xyxy"], plan["source_wh"], plan["grid_hw"], plan["case_id"],
            halo=plan["halo"], target_envelope_xyxy=plan["target_envelope_xyxy"],
            kind=plan["kind"], packed_channels=plan["packed_channels"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("incomplete region plan") from exc
    if expected != plan:
        raise ValueError("region plan changed after construction; rebuild and freeze explicitly")


@dataclass
class FreeRegionRunRecord:
    """Diagnostic snapshots; no source-state/background constraint exists."""

    plan: dict[str, Any] | None
    target_height: int
    target_width: int
    initial_noise_sha256: str
    dit_calls: int = 0
    sigmas: list[float] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    step_latents: list[torch.Tensor] = field(default_factory=list)
    reference_condition_sha256: str | None = None
    text_sha256: str | None = None
    text_mask_sha256: str | None = None
    executor_record: dict[str, Any] | None = None
    completed: bool = False

    def manifest(self):
        return {
            "protocol": "native_free_generation_roi_compute_hint_v1",
            "completed": self.completed, "dit_calls": self.dit_calls,
            "target_hw": [self.target_height, self.target_width],
            "initial_noise_sha256": self.initial_noise_sha256,
            "reference_condition_sha256": self.reference_condition_sha256,
            "text_sha256": self.text_sha256, "text_mask_sha256": self.text_mask_sha256,
            "sigmas": self.sigmas, "steps": copy.deepcopy(self.steps),
            "plan": copy.deepcopy(self.plan), "executor_record": copy.deepcopy(self.executor_record),
            "roi_role": "compute_hint_only_not_generation_permission",
            "source_bbox_is_output_boundary": False,
            "all_target_latents_scheduler_updated_each_step": True,
            "background_fusion": False, "image_crop": False, "rgb_copy": False,
            "extra_reference_image": False, "net_latency_claim": False,
        }


@torch.no_grad()
def run_free_region(
    pipe, image, prompt, initial_noise, plan=None, trace_latents=True, *,
    prompt_embeds=None, prompt_embeds_mask=None, generator=None, output_type="latent",
):
    """Run the genuine native pipe and return its normal native output.

    ``pipe.last_free_region_run`` stores diagnostic records and, by default,
    every full target latent after its unmodified scheduler step. Keyword
    arguments are explicit: masked-pipeline controls/callbacks cannot leak
    into this entry point. The bridge changes only transformer execution.
    """
    if type(pipe) is not QwenImageEditPipeline:
        raise TypeError("run_free_region requires the genuine native QwenImageEditPipeline class")
    if getattr(pipe, "_free_region_request_active", False) or getattr(pipe.transformer, "_controlled_local_bridge_active", False):
        raise RuntimeError("overlapping/nested region requests are forbidden")
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("one complete RGB PIL source image is required")
    if isinstance(prompt, list) or (prompt_embeds is not None and prompt_embeds.shape[0] != 1):
        raise ValueError("a single prompt/output is required")
    if prompt is not None and prompt_embeds is not None:
        raise ValueError("use prompt or explicit prompt_embeds, not both")
    if not isinstance(initial_noise, torch.Tensor) or initial_noise.ndim != 3 or initial_noise.shape[0] != 1:
        raise ValueError("explicit batch-one packed initial_noise is required")
    if not initial_noise.is_floating_point():
        raise ValueError("initial_noise must have a floating dtype")
    _finite(initial_noise, "initial_noise")
    if not isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler):
        raise ValueError("only FlowMatchEulerDiscreteScheduler is supported")
    if pipe.scheduler.config.get("stochastic_sampling", False) or pipe.scheduler.config.get("invert_sigmas", False):
        raise ValueError("only deterministic non-inverted Euler scheduling is supported")
    if pipe.transformer.config.guidance_embeds:
        raise ValueError("this CFG1 contract excludes guidance-distilled embeddings")
    width, height, _ = native_module.calculate_dimensions(1024 * 1024, image.width / image.height)
    multiple = 2 * pipe.vae_scale_factor
    width, height = width // multiple * multiple, height // multiple * multiple
    gh, gw = height // multiple, width // multiple
    channels = pipe.transformer.config.in_channels
    if tuple(initial_noise.shape) != (1, gh * gw, channels):
        raise ValueError("initial_noise shape disagrees with the native full-image target geometry")
    expected_dtype = pipe.transformer.dtype if prompt_embeds is None else prompt_embeds.dtype
    if initial_noise.dtype != expected_dtype or initial_noise.device != torch.device(pipe._execution_device):
        raise ValueError("initial_noise must already match the native prompt dtype/execution device")
    frozen = copy.deepcopy(plan)
    if frozen is not None:
        _validate_region_plan(frozen)
        if frozen["source_wh"] != list(image.size) or frozen["grid_hw"] != [gh, gw] or frozen["packed_channels"] != channels:
            raise ValueError("region plan differs from the actual native image/target geometry")
    record = FreeRegionRunRecord(frozen, height, width, _digest(initial_noise))
    pipe.last_free_region_run = record
    pipe._free_region_request_active = True
    original_forward = pipe.transformer.forward
    executor = None

    def before_transformer(module, positional, fields):
        if positional or record.dit_calls >= 8:
            raise RuntimeError("unexpected positional or extra transformer invocation")
        hidden = fields.get("hidden_states")
        if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] <= gh * gw or hidden.shape[2] != channels:
            raise ValueError("native input must contain full target plus image-condition tokens")
        _finite(hidden, "native transformer input")
        if record.dit_calls == 0:
            if not torch.equal(hidden[:, :gh * gw], initial_noise):
                raise RuntimeError("native first target state differs from explicit initial noise")
            sigmas = pipe.scheduler.sigmas
            if len(pipe.scheduler.timesteps) != 8 or sigmas.numel() != 9 or sigmas[0].item() != 1 or sigmas[-1].item() != 0:
                raise RuntimeError("expected eight native steps with sigmas from exactly one to zero")
            _finite(sigmas, "native scheduler sigmas")
            if not (sigmas[:-1] > sigmas[1:]).all().item() or pipe.scheduler.step_index is not None:
                raise RuntimeError("native schedule is not fresh and strictly decreasing")
            record.sigmas = sigmas.detach().float().cpu().tolist()
            record.reference_condition_sha256 = _digest(hidden[:, gh * gw:])
            record.text_sha256 = _digest(fields["encoder_hidden_states"])
            text_mask = fields.get("encoder_hidden_states_mask")
            record.text_mask_sha256 = None if text_mask is None else _digest(text_mask)
        record.dit_calls += 1

    def after_step(native_pipe, index, timestep, values):
        if index != len(record.steps) or record.dit_calls != index + 1 or native_pipe.scheduler.step_index != index + 1:
            raise RuntimeError("native transformer/scheduler/callback order changed")
        latent = values["latents"]
        if latent.shape != initial_noise.shape:
            raise RuntimeError("native scheduler did not return all target latents")
        _finite(latent, "native scheduler state")
        record.steps.append({
            "step": index, "timestep": float(timestep.item()),
            "sigma_current": float(native_pipe.scheduler.sigmas[index].item()),
            "sigma_next": float(native_pipe.scheduler.sigmas[index + 1].item()),
            "full_target_shape": list(latent.shape),
            "all_target_latents_scheduler_updated": True,
            "state_sha256": _digest(latent) if trace_latents else None,
        })
        if trace_latents:
            record.step_latents.append(latent.detach().cpu().clone())
        # Returning an empty dict leaves the genuine native scheduler result
        # untouched. No pixel/token region is copied, clamped, or fused.
        return {}

    handle = pipe.transformer.register_forward_pre_hook(before_transformer, with_kwargs=True)
    try:
        context = nullcontext(None) if frozen is None else install_local_executor(pipe, frozen)
        with context as executor:
            result = pipe(
                image=image, prompt=prompt, prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask, generator=generator,
                true_cfg_scale=1, num_inference_steps=8, num_images_per_prompt=1,
                height=height, width=width, latents=initial_noise.detach().clone(),
                output_type=output_type, callback_on_step_end=after_step,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        if record.dit_calls != 8 or len(record.steps) != 8:
            raise RuntimeError("native request did not finish exactly eight full scheduler updates")
        if _digest(initial_noise) != record.initial_noise_sha256:
            raise RuntimeError("caller initial_noise changed during the request")
        record.completed = True
        return result
    finally:
        if executor is not None and hasattr(executor, "bridge_record"):
            record.executor_record = copy.deepcopy(executor.bridge_record)
        handle.remove()
        pipe._free_region_request_active = False
        if pipe.transformer.forward != original_forward:
            # Context cleanup should restore this even after an exception.
            pipe.transformer.forward = original_forward
