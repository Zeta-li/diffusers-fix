import copy
from contextlib import nullcontext

from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit as native_module

from free_region import build_region_plan
from legacy_refresh_policy import BudgetController
from local_mask_bridge import install_local_executor, _validate_plan


ACTIONS = {
    "EARLY": [15, 3, 0, 0, 0, 0],
    "MIDDLE": [15, 0, 0, 3, 0, 0],
    "LATE": [15, 0, 0, 0, 0, 3],
}


def make_diagnostic_plan(case, schedule="EARLY", grid=(64, 64), channels=64):
    if schedule not in ACTIONS:
        raise ValueError("Unknown frozen temporal schedule")
    x0, y0, x1, y1 = case["bbox"]
    w, h = case["source_wh"]
    envelope = [max(0, x0 - 32), max(0, y0 - 32), min(w, x1 + 32), min(h, y1 + 32)]
    plan = build_region_plan(case["bbox"], case["source_wh"], list(grid), case["case_id"],
                             halo=0, target_envelope_xyxy=envelope, packed_channels=channels)
    if plan["fallback_reason"]:
        raise ValueError("Temporal diagnostic requires a nonempty four-way background partition")
    if schedule == "EARLY":
        return plan
    plan = copy.deepcopy(plan)
    controller = BudgetController(plan["partition"])
    selected = [controller.select(k, None if k in (0, 7) else ACTIONS[schedule][k - 1]) for k in range(8)]
    plan["kind"] = "diagnostic_temporal_" + schedule.lower()
    plan["selected_indices_per_step"] = selected
    plan["target_tokens_per_step"] = list(map(len, selected))
    plan["total_target_token_updates"] = sum(map(len, selected))
    _validate_plan(plan)
    return plan


def native_request(pipe, image, prompt, noise, plan=None, callback=None, *, output_type="pil",
                   prompt_embeds=None, prompt_embeds_mask=None):
    width, height, _ = native_module.calculate_dimensions(1024 * 1024, image.width / image.height)
    multiple = 2 * pipe.vae_scale_factor
    width, height = width // multiple * multiple, height // multiple * multiple
    if plan is not None:
        _validate_plan(plan)
        if plan["source_wh"] != list(image.size) or plan["grid_hw"] != [height // multiple, width // multiple]:
            raise ValueError("Plan and actual source/target geometry disagree")
    kwargs = {}
    if callback is not None:
        kwargs.update(callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
    context = nullcontext(None) if plan is None else install_local_executor(pipe, plan)
    with context as executor:
        result = pipe(image=image, prompt=prompt, prompt_embeds=prompt_embeds,
                      prompt_embeds_mask=prompt_embeds_mask, true_cfg_scale=1,
                      num_inference_steps=8, num_images_per_prompt=1, height=height, width=width,
                      latents=noise.detach().clone(), output_type=output_type, **kwargs)
    return result, None if executor is None else executor.bridge_record
