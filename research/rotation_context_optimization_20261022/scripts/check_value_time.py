import argparse
import hashlib
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

import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from local_executor import LocalTargetExecutor
from value_time_correction import ValueTimeExecutor


ROOT = Path(__file__).resolve().parent.parent
CONFIG = dict(patch_size=2, in_channels=8, out_channels=2, num_layers=2,
              attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
              axes_dims_rope=(4, 6, 6))
SHAPES = [[(1, 2, 3), (1, 2, 2)]]
SELECTIONS = {0: list(range(6)), 1: [4, 1], 2: [1], 3: [4, 1], 4: [0, 3], 5: [3], 6: [2, 5], 7: list(range(6))}
FORWARD_COUNTS = {}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def tracked_forward(function, label):
    def wrapped(*args, **kwargs):
        counts = FORWARD_COUNTS.setdefault(label, dict(invoked=0, completed=0, failed=0))
        counts["invoked"] += 1
        try:
            result = function(*args, **kwargs)
        except Exception:
            counts["failed"] += 1
            raise
        counts["completed"] += 1
        return result
    return wrapped


def install_counted(executor, label):
    executor.forward = tracked_forward(executor.forward, label)
    executor.original_forward = tracked_forward(executor.original_forward, "native_reference")
    executor.install()


def forward_accounting():
    labels = {label: dict(counts) for label, counts in FORWARD_COUNTS.items()}
    return dict(by_path=labels, **{field: sum(counts[field] for counts in labels.values())
                                 for field in ["invoked", "completed", "failed"]})


def flow(device):
    scheduler = FlowMatchEulerDiscreteScheduler(base_shift=math.log(3), max_shift=math.log(3),
        use_dynamic_shifting=True, time_shift_type="exponential", shift_terminal=None)
    scheduler.set_timesteps(sigmas=[1 - step / 8 for step in range(8)], device=device, mu=math.log(3))
    return scheduler


def fixture(device, dtype):
    torch.manual_seed(1873)
    model = QwenImageTransformer2DModel(**CONFIG).to(device=device, dtype=dtype).eval()
    model.set_attention_backend("native")
    generator = torch.Generator().manual_seed(5817)
    raw = torch.randn((1, 10, 8), generator=generator).to(device=device, dtype=dtype)
    text = torch.randn((1, 5, 24), generator=generator).to(device=device, dtype=dtype)
    mask = torch.tensor([[True, False, True, True, False]], device=device)
    executor = ValueTimeExecutor(model)
    install_counted(executor, "value_local")
    scheduler = flow(device)
    metadata = executor.prepare_tables(scheduler.timesteps)
    return model, executor, raw, text, mask, scheduler, metadata


def rejected(function):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        return str(error)
    raise AssertionError("Invalid Value correction state was accepted")


def theory(device, mode):
    model, executor, _, _, _, _, _ = fixture(device, torch.float32)
    linear = model.transformer_blocks[0].attn.to_v
    linear.bias.copy_(torch.linspace(-0.3, 0.4, linear.bias.numel(), device=device))
    generator = torch.Generator().manual_seed(4321)
    base_unit = torch.rand((32,), generator=generator).to(device) + 0.5
    scales = torch.linspace(0.75, 1.75, 8, device=device) if mode == "affine" else torch.ones(8, device=device)
    units = scales[:, None] * base_unit[None]
    shifts = torch.randn((8, 32), generator=generator).to(device) * 0.2
    projected = torch.stack([linear(shift.reshape(1, -1)).reshape(2, 16) for shift in shifts])
    alpha = (units @ units.T) / (units.square().sum(-1)[None] + 1e-12)
    alpha.fill_diagonal_(1)
    delta = projected[:, None] - alpha[:, :, None, None] * projected[None]
    indices = torch.arange(8, device=device)
    delta[indices, indices] = 0
    executor.alpha[0].copy_(alpha)
    executor.delta[0].copy_(delta)
    executor.projected_shifts[0].copy_(projected)
    executor.correction_mode = mode
    executor.begin(6, 4, [4, 1], [0, 7])
    executor.step = 3
    executor.age.copy_(torch.tensor([0, 1, 2, 0, 1, 2]))
    effective = executor.age.clone()
    effective[[4, 1]] = 3
    normalized = torch.randn((6, 32), generator=generator).to(device)
    normalized = torch.nn.functional.layer_norm(normalized, (32,))
    cached = torch.stack([linear((units[int(age)] * normalized[index] + shifts[int(age)]).reshape(1, -1)).reshape(2, 16)
                          for index, age in enumerate(effective)]).unsqueeze(0)
    expected = linear(normalized * units[3] + shifts[3]).reshape(1, 6, 2, 16)
    original = cached.clone()
    old_age = executor.age.clone()
    old_key = torch.randn_like(cached)
    executor.layer_cache = {0: (old_key.clone(), cached)}
    corrected = executor.read_value_cache(0, cached, torch.tensor([4, 1], device=device))
    torch.testing.assert_close(corrected, expected, atol=1e-5, rtol=1e-5)
    assert torch.equal(cached, original) and torch.equal(executor.age, old_age)
    assert torch.equal(executor.layer_cache[0][0], old_key)
    assert corrected.data_ptr() != cached.data_ptr()
    error = float((corrected - expected).abs().max())
    corrected[0, 0, 0, 0] += 1
    assert torch.equal(cached, original)
    executor.age.fill_(3)
    executor._value_read_step = None
    fresh = expected.clone()
    unchanged = executor.read_value_cache(0, fresh, torch.tensor([4, 1], device=device))
    assert torch.equal(unchanged, fresh)
    return dict(mode=mode, max_abs=error, nonzero_linear_bias=True, same_timestep_exact=True,
                raw_cache_and_age_unchanged=True, no_alias=True, scale_collinear=mode == "affine")


def tables(device, dtype):
    model, executor, _, _, _, scheduler, metadata = fixture(device, dtype)
    diagonal = torch.arange(8, device=device)
    assert torch.equal(executor.alpha[:, diagonal, diagonal], torch.ones_like(executor.alpha[:, diagonal, diagonal]))
    assert torch.count_nonzero(executor.delta[:, diagonal, diagonal]) == 0
    assert all(value.dtype == torch.float32 for value in [executor.alpha, executor.delta, executor.projected_shifts, executor.rho])
    assert metadata["table_bytes"] == executor.table_bytes()
    json.dumps(metadata)
    timestep = scheduler.timesteps.to(dtype=dtype) / 1000
    assert metadata["model_timesteps"] == timestep.float().cpu().tolist()
    dummy = torch.empty((1, 1, model.inner_dim), device=device, dtype=dtype)
    for step in range(8):
        embedding = model.time_text_embed(timestep[step].reshape(1), dummy, None)
        for layer, block in enumerate(model.transformer_blocks):
            first, _ = block.img_mod(embedding).chunk(2, dim=-1)
            shift, _, _ = first.chunk(3, dim=-1)
            expected = block.attn.to_v(shift).reshape(2, 16).float()
            assert torch.equal(executor.projected_shifts[layer, step], expected)
    alpha_ptr, delta_ptr = executor.alpha.data_ptr(), executor.delta.data_ptr()
    executor.clear()
    assert executor.alpha.data_ptr() == alpha_ptr and executor.delta.data_ptr() == delta_ptr
    return metadata


def trajectory(device, dtype, full):
    results = []
    for mode in ["none", "shift", "affine"]:
        model, executor, raw, text, mask, scheduler, metadata = fixture(device, dtype)
        executor.correction_mode = mode
        executor.begin(6, 4, [4, 1], list(range(8)) if full else [0, 7], per_step_indices=SELECTIONS)
        native_read = executor.read_value_cache
        audits = []

        def audit(layer, cv, selected):
            before_v = cv.clone()
            before_k = executor.layer_cache[layer][0].clone()
            before_age = executor.age.clone()
            current = native_read(layer, cv, selected)
            assert torch.equal(cv, before_v) and torch.equal(executor.age, before_age)
            assert torch.equal(executor.layer_cache[layer][0], before_k)
            ages = before_age.clone()
            selected_cpu = executor.all_targets_cpu if executor.step in executor.full_steps else executor.indices_cpu[executor.step]
            ages[selected_cpu] = executor.step
            expected = before_v.clone()
            if mode != "none" and selected.numel() < 6:
                for index, age in enumerate(ages.tolist()):
                    if mode == "shift":
                        expected[:, index] = (before_v[:, index].float() + (executor.projected_shifts[layer, executor.step] - executor.projected_shifts[layer, age])).to(dtype)
                    else:
                        expected[:, index] = (before_v[:, index].float() * executor.alpha[layer, executor.step, age] + executor.delta[layer, executor.step, age]).to(dtype)
                assert current.data_ptr() != cv.data_ptr()
            assert torch.equal(current, expected)
            audits.append(dict(step=executor.step + 1, layer=layer, mode=mode,
                               true_age_before=before_age.tolist(), effective_age=ages.tolist(),
                               true_cache_unchanged=True, exact_per_token_formula=True))
            return current

        executor.read_value_cache = audit
        latent, condition = raw[:, :6].clone(), raw[:, 6:].clone()
        base_model = None
        if mode == "none" and not full:
            torch.manual_seed(1873)
            base_model = QwenImageTransformer2DModel(**CONFIG).to(device=device, dtype=dtype).eval()
            base_model.set_attention_backend("native")
            base = LocalTargetExecutor(base_model)
            install_counted(base, "reference_local")
            base.begin(6, 4, [4, 1], [0, 7], per_step_indices=SELECTIONS)
        rows = []
        for step, timestep in enumerate(scheduler.timesteps):
            current = torch.cat([latent, condition], dim=1)
            kwargs = dict(hidden_states=current, encoder_hidden_states=text, encoder_hidden_states_mask=mask,
                          timestep=timestep.expand(1).to(dtype=dtype) / 1000, img_shapes=SHAPES, return_dict=False)
            output = model(**kwargs)[0].clone()
            if full or step in [0, 7]:
                dense = executor.original_forward(**kwargs)[0]
                assert torch.equal(output, dense)
            if base_model is not None:
                assert torch.equal(output, base_model(**kwargs)[0])
            assert len(executor.layer_cache) == 2
            assert len(audits) == (step + 1) * 2, "Base executor did not invoke read_value_cache once per layer"
            rows.append(dict(step=step + 1, full_step=full or step in [0, 7],
                             age_after=executor.age.tolist(), finite=bool(torch.isfinite(output).all())))
            assert rows[-1]["finite"]
            latent = scheduler.step(output[:, :6], timestep, latent, return_dict=False)[0]
        executor.clear()
        assert executor.table_bytes() == metadata["table_bytes"]
        results.append(dict(mode=mode, steps=rows, cache_audits=audits,
                            none_matches_original_partial=base_model is not None))
    return dict(full_selection=full, modes=results)


def guards(device):
    model, executor, _, _, _, scheduler, _ = fixture(device, torch.float32)
    errors = {}
    executor.correction_mode = "bad"
    errors["invalid_mode"] = rejected(lambda: executor.begin(6, 4, [1], [0, 7]))
    executor.correction_mode = "affine"
    executor.begin(6, 4, [1], [0, 7])
    executor.step = 1
    cached = torch.zeros((1, 6, 2, 16), device=device)
    errors["uninitialized_age"] = rejected(lambda: executor.read_value_cache(0, cached, torch.tensor([1], device=device)))
    executor.age.fill_(0)
    executor.correction_mode = "shift"
    errors["mode_changed_mid_request"] = rejected(lambda: executor.read_value_cache(0, cached, torch.tensor([1], device=device)))
    model.zero_cond_t = True
    errors["zero_cond_t"] = rejected(lambda: executor.prepare_tables(scheduler.timesteps))
    model.zero_cond_t = False
    errors["non_eight_schedule"] = rejected(lambda: executor.prepare_tables(scheduler.timesteps[:4]))
    bare_model = QwenImageTransformer2DModel(**CONFIG).to(device)
    bare = ValueTimeExecutor(bare_model)
    bare.correction_mode = "shift"
    errors["tables_missing"] = rejected(lambda: bare.begin(6, 4, [1], [0, 7]))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    FORWARD_COUNTS.clear()
    output = ROOT / "data" / "value_time_tiny_correctness.json"
    report = dict(status="running", device=str(device), torch_version=torch.__version__,
                  started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), tiny_model_config=CONFIG,
                  real_20b_model_forwards=0, hypothesis_only=True,
                  actual_tiny_forward_calls=forward_accounting(), expected_completed_tiny_forward_calls=172,
                  forward_accounting_scope="Whole tiny Transformer calls only; table and theory submodule calls excluded",
                  sources={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                           for name in ["value_time_correction.py", "check_value_time.py", "local_executor.py"]}, tests=[])
    save(output, report)
    tests = [("theory_shift_bias_frozen_hidden", lambda: theory(device, "shift")),
             ("theory_affine_collinear_scale_bias_frozen_hidden", lambda: theory(device, "affine")),
             ("unsupported_states", lambda: guards(device))]
    for dtype, label in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        tests.extend([(f"{label}_precompute_diagonal_and_schedule", lambda dtype=dtype: tables(device, dtype)),
                      (f"{label}_all_modes_eight_full_steps", lambda dtype=dtype: trajectory(device, dtype, True)),
                      (f"{label}_all_modes_dynamic_true_anchors", lambda dtype=dtype: trajectory(device, dtype, False))])
    started = time.perf_counter()
    with torch.inference_mode():
        for name, function in tests:
            row = dict(name=name)
            test_started = time.perf_counter()
            before_counts = forward_accounting()
            try:
                row.update(status="pass", result=function())
            except Exception as error:
                row.update(status="fail", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            row["seconds"] = time.perf_counter() - test_started
            after_counts = forward_accounting()
            row["actual_tiny_forward_calls"] = dict(
                by_path={label: {field: counts[field] - before_counts["by_path"].get(label, {}).get(field, 0)
                                 for field in ["invoked", "completed", "failed"]}
                         for label, counts in after_counts["by_path"].items()},
                **{field: after_counts[field] - before_counts[field]
                   for field in ["invoked", "completed", "failed"]})
            report["tests"].append(row)
            report["actual_tiny_forward_calls"] = after_counts
            save(output, report)
            print(json.dumps(dict(name=name, status=row["status"], error=row.get("error")), ensure_ascii=False), flush=True)
    failures = [row["name"] for row in report["tests"] if row["status"] != "pass"]
    if not failures and forward_accounting()["completed"] != report["expected_completed_tiny_forward_calls"]:
        failures.append("actual_tiny_forward_call_count")
    report.update(status="fail" if failures else "pass", failed_tests=failures, elapsed_seconds=time.perf_counter() - started,
                  completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    save(output, report)
    assert not failures, f"Value time correction tiny checks failed: {failures}"


if __name__ == "__main__":
    main()
