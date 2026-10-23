import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel, apply_rotary_emb_qwen

from local_executor import LocalTargetExecutor


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "tiny_correctness.json"
TARGETS = 6
CONDITIONS = 4
TEXT = 5
SHAPES = [[(1, 2, 3), (1, 2, 2)]]
TINY_CONFIG = dict(
    patch_size=2,
    in_channels=8,
    out_channels=2,
    num_layers=2,
    attention_head_dim=16,
    num_attention_heads=2,
    joint_attention_dim=24,
    axes_dims_rope=(4, 6, 6),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(value):
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def fixture(dtype):
    torch.manual_seed(1873)
    model = QwenImageTransformer2DModel(**TINY_CONFIG).to(device="cpu", dtype=dtype).eval()
    model.set_attention_backend("native")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(5817)
    raw = torch.randn((1, TARGETS + CONDITIONS, 8), generator=generator).to(dtype)
    text = torch.randn((1, TEXT, 24), generator=generator).to(dtype)
    mask = torch.tensor([[True, False, True, True, False]])
    executor = LocalTargetExecutor(model)
    executor.install()
    return model, executor, raw, text, mask


def inputs(raw, text, mask, timestep=0.8):
    return dict(
        hidden_states=raw,
        encoder_hidden_states=text,
        encoder_hidden_states_mask=mask,
        timestep=torch.tensor([timestep], dtype=raw.dtype),
        img_shapes=SHAPES,
        return_dict=False,
    )


def compare(actual, expected, dtype):
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    difference = actual_float - expected_float
    metrics = dict(
        max_abs=float(difference.abs().max()),
        mean_abs=float(difference.abs().mean()),
        relative_l2=float(difference.norm() / expected_float.norm().clamp_min(1e-12)),
        bitwise_equal=bool(torch.equal(actual, expected)),
        actual_dtype=str(actual.dtype),
        expected_dtype=str(expected.dtype),
    )
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    metrics.update(atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    return metrics


def expect_rejected(function, message=None):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        if message is not None:
            assert message in str(error), str(error)
        return dict(exception=type(error).__name__, message=str(error))
    raise AssertionError("Invalid operation was accepted")


def scheduler():
    result = FlowMatchEulerDiscreteScheduler(
        base_shift=math.log(3),
        max_shift=math.log(3),
        base_image_seq_len=256,
        max_image_seq_len=8192,
        use_dynamic_shifting=True,
        time_shift_type="exponential",
        shift_terminal=None,
    )
    result.set_timesteps(sigmas=[1 - index / 8 for index in range(8)], device="cpu", mu=math.log(3))
    return result


def full_forward(dtype, reverse=False):
    model, executor, raw, text, mask = fixture(dtype)
    kwargs = inputs(raw, text, mask)
    expected = executor.original_forward(**kwargs)[0].detach().clone()
    executor.begin(TARGETS, CONDITIONS, list(range(TARGETS)), set(range(8)), reverse_full=reverse)
    actual = model(**kwargs)[0].detach().clone()
    metrics = compare(actual, expected, dtype)
    assert executor.step == 1
    assert len(executor.layer_cache) == len(model.transformer_blocks)
    assert all(row["image_qkv_tokens"] == TARGETS + CONDITIONS for row in executor.trace)
    assert all(row["image_mlp_tokens"] == TARGETS + CONDITIONS for row in executor.trace)
    assert torch.equal(executor.age, torch.zeros(TARGETS, dtype=torch.long))
    executor.begin(TARGETS, CONDITIONS, list(range(TARGETS)), set(range(8)), reverse_full=reverse)
    dict_kwargs = dict(kwargs, return_dict=True)
    dict_output = model(**dict_kwargs).sample.detach().clone()
    metrics["return_dict"] = compare(dict_output, expected, dtype)
    metrics["reverse_full"] = reverse
    metrics["cache_bytes"] = executor.cache_bytes()
    return metrics


def full_eight_steps(dtype, reverse=False):
    model, executor, raw, text, mask = fixture(dtype)
    reference_scheduler = scheduler()
    local_scheduler = scheduler()
    executor.begin(TARGETS, CONDITIONS, list(range(TARGETS)), set(range(8)), reverse_full=reverse)
    reference_latent = raw[:, :TARGETS].clone()
    local_latent = reference_latent.clone()
    condition = raw[:, TARGETS:].clone()
    rows = []
    for step, timestep in enumerate(reference_scheduler.timesteps):
        reference_raw = torch.cat([reference_latent, condition], dim=1)
        local_raw = torch.cat([local_latent, condition], dim=1)
        scalar = float(timestep / 1000)
        reference_velocity = executor.original_forward(**inputs(reference_raw, text, mask, scalar))[0].detach().clone()
        local_velocity = model(**inputs(local_raw, text, mask, scalar))[0].detach().clone()
        velocity_error = compare(local_velocity, reference_velocity, dtype)
        reference_previous = reference_latent.clone()
        reference_latent = reference_scheduler.step(reference_velocity[:, :TARGETS], timestep, reference_latent, return_dict=False)[0]
        local_latent = local_scheduler.step(local_velocity[:, :TARGETS], local_scheduler.timesteps[step], local_latent, return_dict=False)[0]
        assert not torch.equal(reference_previous, reference_latent), f"Latent did not change at step {step}"
        rows.append(dict(step=step + 1, timestep=float(timestep), velocity_error=velocity_error,
                         latent_error=compare(local_latent, reference_latent, dtype)))
        assert len(executor.layer_cache) == len(model.transformer_blocks)
    assert executor.step == 8
    rejection = expect_rejected(lambda: model(**inputs(torch.cat([local_latent, condition], dim=1), text, mask)), "begin")
    executor.begin(TARGETS, CONDITIONS, list(range(TARGETS)), {0, 7})
    assert executor.step == 0 and not executor.layer_cache and executor.velocity is None
    assert torch.equal(executor.age, torch.full((TARGETS,), -1, dtype=torch.long))
    reset_raw = torch.cat([local_latent + 0.03125, condition], dim=1)
    reset_text = text + 0.125
    reset_expected = executor.original_forward(**inputs(reset_raw, reset_text, mask))[0].detach().clone()
    reset_actual = model(**inputs(reset_raw, reset_text, mask))[0].detach().clone()
    return dict(reverse_full=reverse, steps=rows, ninth_call_rejected=rejection,
                explicit_new_request_reset=compare(reset_actual, reset_expected, dtype))


class ModuleAudit:
    def __init__(self, model):
        self.active = False
        self.rows = []
        self.raw_inputs = []
        self.handles = []
        self.handles.append(model.img_in.register_forward_pre_hook(self.capture("img_in", raw=True)))
        for index, block in enumerate(model.transformer_blocks):
            modules = dict(
                image_norm1=block.img_norm1,
                image_norm2=block.img_norm2,
                image_q=block.attn.to_q,
                image_k=block.attn.to_k,
                image_v=block.attn.to_v,
                image_out=block.attn.to_out[0],
                image_mlp=block.img_mlp,
                text_q=block.attn.add_q_proj,
                text_k=block.attn.add_k_proj,
                text_v=block.attn.add_v_proj,
                text_mlp=block.txt_mlp,
            )
            for name, module in modules.items():
                self.handles.append(module.register_forward_pre_hook(self.capture(name, layer=index)))

    def capture(self, name, layer=None, raw=False):
        def hook(module, arguments):
            if self.active:
                value = arguments[0]
                self.rows.append(dict(name=name, layer=layer, shape=list(value.shape)))
                if raw:
                    self.raw_inputs.append(value.detach().clone())
        return hook

    def reset(self):
        self.rows = []
        self.raw_inputs = []

    def close(self):
        for handle in self.handles:
            handle.remove()


def partial_and_dynamic(dtype):
    model, executor, raw, text, mask = fixture(dtype)
    selections = {0: list(range(TARGETS)), 1: [1], 2: [1], 3: [1, 2], 4: [2, 4], 5: [4], 6: [0, 4, 5], 7: list(range(TARGETS))}
    executor.begin(TARGETS, CONDITIONS, [1], {0, 7}, per_step_indices=selections)
    audit = ModuleAudit(model)
    flow = scheduler()
    latent = raw[:, :TARGETS].clone()
    condition = raw[:, TARGETS:].clone()
    rows = []
    try:
        for step, timestep in enumerate(flow.timesteps):
            selected = torch.tensor(selections[step], dtype=torch.long)
            inactive = torch.tensor([index for index in range(TARGETS) if index not in selections[step]], dtype=torch.long)
            image_indices = torch.cat([selected, torch.arange(TARGETS, TARGETS + CONDITIONS)])
            current = torch.cat([latent, condition], dim=1)
            expected_raw = current.index_select(1, image_indices).clone()
            previous_velocity = None if executor.velocity is None else executor.velocity.detach().clone()
            previous_cache = {index: (key.clone(), value.clone()) for index, (key, value) in executor.layer_cache.items()}
            previous_age = executor.age.clone()
            audit.reset()
            audit.active = True
            kwargs = inputs(current, text, mask, float(timestep / 1000))
            output = model(**kwargs)[0].detach().clone()
            audit.active = False
            assert len(audit.raw_inputs) == 1
            assert torch.equal(audit.raw_inputs[0], expected_raw), "img_in did not receive current raw gathered input"
            for row in audit.rows:
                expected_tokens = TEXT if row["name"].startswith("text_") else len(selected) + CONDITIONS
                assert row["shape"][1] == expected_tokens, row
            for layer in range(len(model.transformer_blocks)):
                for name in ["image_norm1", "image_norm2", "image_q", "image_k", "image_v", "image_out", "image_mlp", "text_q", "text_k", "text_v", "text_mlp"]:
                    assert sum(row["name"] == name and row["layer"] == layer for row in audit.rows) == 1
            assert len(executor.layer_cache) == len(model.transformer_blocks)
            if inactive.numel() and previous_velocity is not None:
                assert torch.equal(output.index_select(1, inactive), previous_velocity.index_select(1, inactive))
                for layer, (cached_key, cached_value) in executor.layer_cache.items():
                    old_key, old_value = previous_cache[layer]
                    assert torch.equal(cached_key.index_select(1, inactive), old_key.index_select(1, inactive))
                    assert torch.equal(cached_value.index_select(1, inactive), old_value.index_select(1, inactive))
            block = model.transformer_blocks[0]
            projected = model.img_in(expected_raw)
            temb = model.time_text_embed(kwargs["timestep"].to(projected.dtype), projected, None)
            modulation, _ = block.img_mod(temb).chunk(2, dim=-1)
            normalized, _ = block._modulate(block.img_norm1(projected), modulation)
            expected_key = block.attn.to_k(normalized).unflatten(-1, (-1, block.attn.inner_dim // block.attn.heads))
            if block.attn.norm_k is not None:
                expected_key = block.attn.norm_k(expected_key)
            rope = model.pos_embed(SHAPES, max_txt_seq_len=TEXT, device=projected.device)[0].index_select(0, image_indices)
            expected_key = apply_rotary_emb_qwen(expected_key, rope, use_real=False)
            expected_value = block.attn.to_v(normalized).unflatten(-1, (-1, block.attn.inner_dim // block.attn.heads))
            key_error = compare(executor.layer_cache[0][0].index_select(1, selected), expected_key[:, :len(selected)], dtype)
            value_error = compare(executor.layer_cache[0][1].index_select(1, selected), expected_value[:, :len(selected)], dtype)
            assert torch.equal(executor.age.index_select(0, selected), torch.full((len(selected),), step, dtype=torch.long))
            if inactive.numel():
                assert torch.equal(executor.age.index_select(0, inactive), previous_age.index_select(0, inactive))
            expected_reactivated = int(((previous_age.index_select(0, selected) < step - 1) & (previous_age.index_select(0, selected) >= 0)).sum()) if step else 0
            traces = [row for row in executor.trace if row["step"] == step + 1]
            assert traces and all(row["newly_activated_from_current_latent"] == expected_reactivated for row in traces)
            full_error = None
            if len(selected) == TARGETS:
                expected_full = executor.original_forward(**kwargs)[0].detach().clone()
                full_error = compare(output, expected_full, dtype)
            old_latent = latent.clone()
            latent = flow.step(output[:, :TARGETS], timestep, latent, return_dict=False)[0]
            inactive_latent_updated = None
            if inactive.numel():
                inactive_latent_updated = not torch.equal(latent.index_select(1, inactive), old_latent.index_select(1, inactive))
                assert inactive_latent_updated, "Inactive latent was frozen rather than stepped by the scheduler"
            rows.append(dict(step=step + 1, selected_targets=selected.tolist(), image_indices=image_indices.tolist(),
                             source_gather_exact=True, module_inputs=audit.rows, cached_key_error=key_error,
                             cached_value_error=value_error, full_step_error=full_error,
                             inactive_latent_updated=inactive_latent_updated, reactivated_from_current_latent=expected_reactivated,
                             cache_bytes=executor.cache_bytes()))
    finally:
        audit.close()
    assert rows[3]["reactivated_from_current_latent"] == 1
    return dict(steps=rows, stale_inactive_velocity_is_explicit_approximation=True,
                inactive_keys_values_preserved=True, no_dense_image_projection_before_gather=True)


def pack_unpack_geometry():
    raw = torch.arange(48, dtype=torch.float32).reshape(1, 2, 1, 4, 6)
    packed = QwenImageEditPipeline._pack_latents(raw, 1, 2, 4, 6)
    assert list(packed.shape) == [1, 6, 8]
    unpacked = QwenImageEditPipeline._unpack_latents(packed, 32, 48, 8)
    assert torch.equal(unpacked, raw)
    indices = []
    for row in range(2):
        for column in range(3):
            index = row * 3 + column
            expected = raw[:, :, 0, row * 2:row * 2 + 2, column * 2:column * 2 + 2].reshape(1, 8)
            assert torch.equal(packed[:, index], expected)
            indices.append(dict(index=index, packed_yx=[row, column], output_xyxy=[column * 16, row * 16, (column + 1) * 16, (row + 1) * 16]))
    return dict(target_packed_grid=[2, 3], target_pixels_hw=[32, 48], packed_shape=list(packed.shape),
                condition_grid_is_independently_sized=[2, 2], roundtrip_exact=True, mappings=indices)


def invalid_indices():
    _, executor, _, _, _ = fixture(torch.float32)
    invalid = dict(empty=[], duplicate=[1, 1], negative=[-1], out_of_range=[TARGETS])
    return {name: expect_rejected(lambda value=value: executor.begin(TARGETS, CONDITIONS, value, {0, 7}), "Invalid target indices") for name, value in invalid.items()}


def missing_cache_and_state_change():
    model, executor, raw, text, mask = fixture(torch.float32)
    kwargs = inputs(raw, text, mask)
    executor.begin(TARGETS, CONDITIONS, [1], set())
    initial_partial = expect_rejected(lambda: model(**kwargs), "First step")
    executor.begin(TARGETS, CONDITIONS, [1], {0, 7})
    model(**kwargs)
    del executor.layer_cache[0]
    missing_layer = expect_rejected(lambda: model(**kwargs), "Missing initialized layer cache")
    changed = {}
    mutations = dict(
        hidden_shape=dict(kwargs, hidden_states=torch.cat([raw, raw[:, :1]], dim=1)),
        hidden_dtype=dict(kwargs, hidden_states=raw.double()),
        text_sequence_length=dict(kwargs, encoder_hidden_states=torch.cat([text, text[:, :1]], dim=1)),
    )
    for name, mutated in mutations.items():
        executor.begin(TARGETS, CONDITIONS, [1], {0, 7})
        model(**kwargs)
        changed[name] = expect_rejected(lambda mutated=mutated: model(**mutated), "changed")
    return dict(first_partial_rejected=initial_partial, missing_partial_cache_rejected=missing_layer,
                state_changes_rejected=changed,
                reset_contract="Same-shape new images and prompts require an explicit begin() by the caller")


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    packages = {}
    for name in ["torch", "peft", "transformers", "huggingface_hub", "torchao"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = dict(
        status="running",
        started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        device="cpu",
        gpu_requests=0,
        tiny_model_config=TINY_CONFIG,
        target_tokens=TARGETS,
        condition_tokens=CONDITIONS,
        text_tokens=TEXT,
        text_mask=[[1, 0, 1, 1, 0]],
        fp32_tolerances=dict(atol=1e-5, rtol=1e-5),
        bf16_tolerances=dict(atol=2e-2, rtol=2e-2),
        python=sys.version,
        packages=packages,
        diffusers_version=diffusers.__version__,
        diffusers_source=diffusers.__file__,
        model_source=inspect.getfile(QwenImageTransformer2DModel),
        model_source_sha256=digest(inspect.getfile(QwenImageTransformer2DModel)),
        executor_source_sha256=digest(Path(__file__).with_name("local_executor.py")),
        test_source_sha256=digest(__file__),
        tests=[],
        limitations=["Synthetic tiny-model correctness does not establish 20B model quality or speed", "BF16 tolerances allow floating point variation and do not imply lossless editing", "Callers must begin() for every new request even when tensor shapes match"],
    )
    save(report)
    tests = [("pack_unpack_non_square", pack_unpack_geometry), ("invalid_indices_rejected", invalid_indices),
             ("missing_cache_and_state_change_rejected", missing_cache_and_state_change)]
    for dtype, label in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        tests.extend([
            (f"{label}_full_custom", lambda dtype=dtype: full_forward(dtype)),
            (f"{label}_reverse_full_custom", lambda dtype=dtype: full_forward(dtype, reverse=True)),
            (f"{label}_eight_full_steps", lambda dtype=dtype: full_eight_steps(dtype)),
            (f"{label}_eight_reverse_full_steps", lambda dtype=dtype: full_eight_steps(dtype, reverse=True)),
            (f"{label}_partial_dynamic_module_audit", lambda dtype=dtype: partial_and_dynamic(dtype)),
        ])
    with torch.inference_mode():
        for name, function in tests:
            test_started = time.perf_counter()
            row = dict(name=name)
            try:
                row.update(status="pass", result=function())
            except Exception as error:
                row.update(status="fail", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            row["seconds"] = time.perf_counter() - test_started
            report["tests"].append(row)
            save(report)
            print(json.dumps(dict(name=name, status=row["status"], seconds=row["seconds"], error=row.get("error")), ensure_ascii=False), flush=True)
    failures = [row["name"] for row in report["tests"] if row["status"] != "pass"]
    report.update(status="fail" if failures else "pass", failed_tests=failures, test_count=len(tests),
                  passed_tests=len(tests) - len(failures), elapsed_seconds=time.perf_counter() - started,
                  completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    save(report)
    print(json.dumps(dict(status=report["status"], output=str(OUTPUT), failed_tests=failures), ensure_ascii=False), flush=True)
    assert not failures, f"Tiny executor correctness failed: {failures}"


if __name__ == "__main__":
    main()
