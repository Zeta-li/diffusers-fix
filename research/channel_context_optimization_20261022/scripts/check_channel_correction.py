import argparse
import hashlib
import json
import math
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument("--stage4-scripts")
bootstrap_args, _ = bootstrap.parse_known_args()
if bootstrap_args.stage4_scripts:
    os.environ["VALUE_CHANNEL_STAGE4_SCRIPTS"] = bootstrap_args.stage4_scripts
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from value_channel_correction import (
    RIDGE_FLOOR,
    RIDGE_RELATIVE,
    STAGE4_SCRIPTS,
    ValueChannelExecutor,
    channel_coefficients,
)
from local_executor import LocalTargetExecutor


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


def synthetic_statistics(device):
    mean = torch.linspace(-0.4, 0.5, 64, device=device).reshape(2, 32)
    variance = torch.linspace(0.2, 1.7, 64, device=device).reshape(2, 32)
    return dict(mean=mean, variance=variance, source="Synthetic tiny-test statistics; no Train8 calibration",
                source_hash=hashlib.sha256(b"synthetic-tiny-mean-variance-v1").hexdigest())


def fixture(device, dtype, statistics=None):
    torch.manual_seed(1873)
    model = QwenImageTransformer2DModel(**CONFIG).to(device=device, dtype=dtype).eval()
    model.set_attention_backend("native")
    generator = torch.Generator().manual_seed(5817)
    raw = torch.randn((1, 10, 8), generator=generator).to(device=device, dtype=dtype)
    text = torch.randn((1, 5, 24), generator=generator).to(device=device, dtype=dtype)
    mask = torch.tensor([[True, False, True, True, False]], device=device)
    executor = ValueChannelExecutor(model)
    install_counted(executor, "channel_local")
    scheduler = flow(device)
    metadata = executor.prepare_tables(scheduler.timesteps, statistics=statistics)
    return model, executor, raw, text, mask, scheduler, metadata


def rejected(function):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        return str(error)
    raise AssertionError("Invalid channel correction state was accepted")


def closed_form(device):
    weight = torch.tensor([[0.3, -0.8, 0.4], [-0.5, 0.6, 1.1]], device=device)
    step = torch.arange(8, device=device).float()[:, None]
    units = torch.cat([1 + 0.1 * step, 0.8 - 0.03 * step, 0.4 + 0.02 * step], dim=1)
    shift = torch.cat([0.2 * step, -0.1 * step, 0.05 + 0.04 * step], dim=1)
    bias = torch.tensor([0.37, -0.28], device=device)
    projected = shift @ weight.T + bias
    mean = torch.cat([0.1 + step, -0.2 + 0.4 * step, 0.3 - 0.2 * step], dim=1)
    variance = torch.cat([0.2 + 0.03 * step, 0.5 + 0.02 * step, 1.4 - 0.04 * step], dim=1)
    alpha, delta, rho, ridge = channel_coefficients(weight, units, projected, mean, variance)
    t, s = 6, 2
    denominator = torch.stack([sum(weight[j, i].square() * variance[s, i] * units[s, i].square()
                                    for i in range(3)) for j in range(2)])
    numerator = torch.stack([sum(weight[j, i].square() * variance[s, i] * units[t, i] * units[s, i]
                                  for i in range(3)) for j in range(2)])
    regularizer = denominator.mean() * RIDGE_RELATIVE + RIDGE_FLOOR
    expected_alpha = (numerator + regularizer) / (denominator + regularizer)
    target_center = projected[t] + weight @ (units[t] * mean[s])
    anchor_center = projected[s] + weight @ (units[s] * mean[s])
    expected_delta = target_center - expected_alpha * anchor_center
    torch.testing.assert_close(alpha[t, s], expected_alpha, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(delta[t, s], expected_delta, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ridge[s, 0], regularizer, atol=1e-10, rtol=1e-6)
    direct_residual = (weight.square() * variance[s][None]
                       * (units[t][None] - alpha[t, s, :, None] * units[s][None]).square()).sum()
    direct_target_variance = (weight.square() * variance[s][None] * units[t][None].square()).sum()
    direct_rho = (direct_residual / direct_target_variance).sqrt()
    torch.testing.assert_close(rho[t, s], direct_rho, atol=1e-6, rtol=1e-4)
    wrong_current_mean = projected[t] + weight @ (units[t] * mean[t]) - expected_alpha * anchor_center
    assert float((wrong_current_mean - delta[t, s]).abs().max()) > 0.1
    diagonal = torch.arange(8, device=device)
    assert torch.equal(alpha[diagonal, diagonal], torch.ones((8, 2), device=device))
    assert torch.count_nonzero(delta[diagonal, diagonal]) == 0
    assert not torch.equal(alpha[t, s, 0], alpha[t, s, 1])
    return dict(nonzero_mean_variance_bias=True, anchor_mean_verified=True, channel_gains_distinct=True,
                max_alpha_error=float((alpha[t, s] - expected_alpha).abs().max()),
                max_delta_error=float((delta[t, s] - expected_delta).abs().max()))


def monte_carlo(device):
    generator = torch.Generator().manual_seed(3781)
    weight = torch.randn((4, 5), generator=generator).to(device) * 0.4
    units = torch.randn((8, 5), generator=generator).to(device) * 0.3 + 0.9
    mean = torch.randn((8, 5), generator=generator).to(device) * 0.7
    variance = torch.rand((8, 5), generator=generator).to(device) + 0.2
    shift = torch.randn((8, 5), generator=generator).to(device) * 0.2
    bias = torch.linspace(-0.4, 0.5, 4, device=device)
    projected = shift @ weight.T + bias
    alpha, delta, _, ridge = channel_coefficients(weight, units, projected, mean, variance)
    t, s, samples = 6, 1, 120000
    hidden = torch.randn((samples, 5), generator=generator).to(device) * variance[s].sqrt() + mean[s]
    anchor = (hidden * units[s] + shift[s]) @ weight.T + bias
    target = (hidden * units[t] + shift[t]) @ weight.T + bias
    prediction = anchor * alpha[t, s] + delta[t, s]
    empirical = (target - prediction).square().mean(dim=0)
    squared = weight.square()
    denominator = (variance[s] * units[s].square()) @ squared.T
    numerator = (variance[s] * units[t] * units[s]) @ squared.T
    target_variance = (variance[s] * units[t].square()) @ squared.T
    slope = alpha[t, s]
    theoretical = target_variance - 2 * slope * numerator + slope.square() * denominator
    torch.testing.assert_close(empirical, theoretical, atol=2e-3, rtol=0.03)
    center = projected[t] + weight @ (units[t] * mean[s])
    torch.testing.assert_close(target.mean(dim=0), center, atol=0.015, rtol=0.015)
    objective = lambda a: target_variance - 2 * a * numerator + a.square() * denominator + ridge[s, 0] * (a - 1).square()
    optimum = objective(slope)
    assert bool((objective(slope + 0.05) >= optimum - 1e-6).all())
    assert bool((objective(slope - 0.05) >= optimum - 1e-6).all())
    return dict(samples=samples, empirical_mse=empirical.cpu().tolist(), theoretical_mse=theoretical.cpu().tolist(),
                max_absolute_mse_gap=float((empirical - theoretical).abs().max()),
                ridge_objective_local_optimum=True, sample_calibration_performed=False)


def collinear_and_degenerate(device):
    generator = torch.Generator().manual_seed(7285)
    weight = torch.randn((4, 5), generator=generator).to(device) * 0.3
    weight[0] = 0
    scales = torch.linspace(0.8, 1.6, 8, device=device)
    base_unit = torch.rand((5,), generator=generator).to(device) + 0.5
    units = scales[:, None] * base_unit
    mean = torch.randn((8, 5), generator=generator).to(device) * 0.4
    variance = torch.rand((8, 5), generator=generator).to(device) + 0.5
    shift = torch.randn((8, 5), generator=generator).to(device) * 0.2
    bias = torch.linspace(0.1, 0.4, 4, device=device)
    projected = shift @ weight.T + bias
    alpha, delta, _, ridge = channel_coefficients(weight, units, projected, mean, variance)
    t, s = 6, 1
    denominator = (variance[s] * units[s].square()) @ weight.square().T
    ratio = scales[t] / scales[s]
    expected_alpha = (ratio * denominator + ridge[s, 0]) / (denominator + ridge[s, 0])
    torch.testing.assert_close(alpha[t, s], expected_alpha, atol=1e-6, rtol=1e-6)
    hidden = torch.randn((256, 5), generator=generator).to(device) * variance[s].sqrt() + mean[s]
    anchor = (hidden * units[s] + shift[s]) @ weight.T + bias
    target = (hidden * units[t] + shift[t]) @ weight.T + bias
    anchor_center = projected[s] + weight @ (units[s] * mean[s])
    prediction = anchor * alpha[t, s] + delta[t, s]
    expected_error = (alpha[t, s] - ratio) * (anchor - anchor_center)
    torch.testing.assert_close(prediction - target, expected_error, atol=3e-6, rtol=1e-4)
    zero_alpha, zero_delta, zero_rho, _ = channel_coefficients(weight, units, projected, mean, torch.zeros_like(variance))
    assert all(bool(torch.isfinite(array).all()) for array in [zero_alpha, zero_delta, zero_rho])
    assert torch.equal(zero_alpha, torch.ones_like(zero_alpha))
    zero_anchor = projected[s] + weight @ (units[s] * mean[s])
    zero_target = projected[t] + weight @ (units[t] * mean[s])
    torch.testing.assert_close(zero_anchor * zero_alpha[t, s] + zero_delta[t, s], zero_target, atol=1e-6, rtol=1e-6)
    return dict(zero_projection_row_finite=True, zero_variance_finite=True,
                max_collinear_prediction_error=float((prediction - target).abs().max()),
                max_analytic_ridge_error_gap=float((prediction - target - expected_error).abs().max()),
                collinear_interpretation="Matches analytic identity-prior ridge bias; not mathematically zero for nonunit scale")


def table_checks(device, dtype):
    model, executor, _, _, _, scheduler, metadata = fixture(device, dtype)
    assert metadata["statistics_mode"] == "no_sample_mean_zero_variance_one"
    assert metadata["calibration_performed_here"] is False
    assert metadata["table_bytes"] == executor.table_bytes()
    assert metadata["model_timesteps"] == (scheduler.timesteps.to(dtype=dtype) / 1000).float().cpu().tolist()
    assert list(executor.alpha.shape) == [2, 8, 8, 2, 16]
    diagonal = torch.arange(8, device=device)
    assert torch.equal(executor.alpha[:, diagonal, diagonal], torch.ones_like(executor.alpha[:, diagonal, diagonal]))
    assert torch.count_nonzero(executor.delta[:, diagonal, diagonal]) == 0
    assert all(array.dtype == torch.float32 for array in [executor.alpha, executor.delta, executor.projected_shifts, executor.rho])
    json.dumps(metadata)
    dummy = torch.empty((1, 1, 32), device=device, dtype=dtype)
    for step, timestep in enumerate(scheduler.timesteps.to(dtype=dtype) / 1000):
        embedding = model.time_text_embed(timestep.reshape(1), dummy, None)
        for layer, block in enumerate(model.transformer_blocks):
            first, _ = block.img_mod(embedding).chunk(2, dim=-1)
            shift, _, _ = first.chunk(3, dim=-1)
            assert torch.equal(executor.projected_shifts[layer, step], block.attn.to_v(shift).reshape(2, 16).float())
    stats = synthetic_statistics(device)
    shared_metadata = executor.prepare_tables(scheduler.timesteps, stats)
    shared_alpha, shared_delta = executor.alpha.clone(), executor.delta.clone()
    repeated = {**stats, "mean": stats["mean"][:, None].expand(-1, 8, -1),
                "variance": stats["variance"][:, None].expand(-1, 8, -1)}
    repeated_metadata = executor.prepare_tables(scheduler.timesteps, repeated)
    assert torch.equal(executor.alpha, shared_alpha) and torch.equal(executor.delta, shared_delta)
    assert shared_metadata["statistics_float32_content_sha256"] == repeated_metadata["statistics_float32_content_sha256"]
    assert repeated_metadata["caller_statistics_source_hash"] == stats["source_hash"]
    ptrs = [executor.alpha.data_ptr(), executor.delta.data_ptr()]
    executor.clear()
    assert ptrs == [executor.alpha.data_ptr(), executor.delta.data_ptr()]
    return dict(default_metadata=metadata, shared_and_timestep_statistics_equal=True,
                statistics_api_uses_synthetic_data_only=True, tables_persist_after_clear=True)


def trajectory(device, dtype, full):
    results = []
    cases = [("none", None, "none"), ("affine", None, "no_sample_affine"),
             ("affine", synthetic_statistics(device), "synthetic_statistics_affine")]
    for mode, statistics, label in cases:
        model, executor, raw, text, mask, scheduler, metadata = fixture(device, dtype, statistics)
        executor.mode = mode
        assert executor.correction_mode == mode
        executor.begin(6, 4, [4, 1], list(range(8)) if full else [0, 7], per_step_indices=SELECTIONS)
        native_read, audits, projection_sizes, projected_values, image_inputs = executor.read_value_cache, [], [], {}, []
        handles = [model.img_in.register_forward_pre_hook(lambda module, args: image_inputs.append(args[0].clone()))]

        def capture(layer):
            def hook(module, args, output):
                projection_sizes.append((layer, args[0].shape[1]))
                projected_values[layer] = output.clone()
            return hook

        for layer, block in enumerate(model.transformer_blocks):
            handles.append(block.attn.to_v.register_forward_hook(capture(layer)))

        def audit(layer, cv, selected):
            before_v, before_k, before_age = cv.clone(), executor.layer_cache[layer][0].clone(), executor.age.clone()
            fresh = projected_values[layer][:, :selected.numel()].reshape(1, selected.numel(), 2, 16)
            assert torch.equal(cv.index_select(1, selected), fresh)
            current = native_read(layer, cv, selected)
            assert torch.equal(cv, before_v) and torch.equal(executor.age, before_age)
            assert torch.equal(executor.layer_cache[layer][0], before_k)
            ages = before_age.clone()
            selected_cpu = executor.all_targets_cpu if executor.step in executor.full_steps else executor.indices_cpu[executor.step]
            ages[selected_cpu] = executor.step
            expected = before_v.clone()
            if mode == "affine" and selected.numel() < 6:
                for index, age in enumerate(ages.tolist()):
                    expected[:, index] = (before_v[:, index].float() * executor.alpha[layer, executor.step, age]
                                          + executor.delta[layer, executor.step, age]).to(dtype)
                assert current.data_ptr() != cv.data_ptr()
            else:
                assert current.data_ptr() == cv.data_ptr()
            assert torch.equal(current, expected)
            assert torch.equal(current.index_select(1, selected), fresh)
            audits.append(dict(step=executor.step + 1, layer=layer, effective_anchor=ages.tolist(),
                               real_cache_and_age_unchanged=True, selected_current_value_exact=True,
                               per_token_channel_formula_exact=True))
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
        try:
            for step, timestep in enumerate(scheduler.timesteps):
                current = torch.cat([latent, condition], dim=1)
                selected_cpu = executor.all_targets_cpu if step in executor.full_steps else executor.indices_cpu[step]
                indices = torch.cat([selected_cpu, torch.arange(6, 10)]).to(device)
                image_inputs.clear()
                projection_sizes.clear()
                kwargs = dict(hidden_states=current, encoder_hidden_states=text, encoder_hidden_states_mask=mask,
                              timestep=timestep.expand(1).to(dtype=dtype) / 1000, img_shapes=SHAPES, return_dict=False)
                output = model(**kwargs)[0].clone()
                assert len(image_inputs) == 1 and torch.equal(image_inputs[0], current.index_select(1, indices))
                assert projection_sizes == [(layer, len(selected_cpu) + 4) for layer in range(2)]
                if full or step in [0, 7]:
                    assert torch.equal(output, executor.original_forward(**kwargs)[0])
                if base_model is not None:
                    assert torch.equal(output, base_model(**kwargs)[0])
                assert len(audits) == (step + 1) * 2
                assert bool(torch.isfinite(output).all())
                rows.append(dict(step=step + 1, full_step=full or step in [0, 7], age_after=executor.age.tolist(),
                                 gathered_image_input_exact=True, target_projection_reduced=len(selected_cpu) < 6,
                                 dense_reference_exact=full or step in [0, 7]))
                latent = scheduler.step(output[:, :6], timestep, latent, return_dict=False)[0]
        finally:
            for handle in handles:
                handle.remove()
        executor.clear()
        assert executor.table_bytes() == metadata["table_bytes"]
        results.append(dict(case=label, mode=mode, steps=rows, cache_audits=audits,
                            no_training_or_calibration_in_this_test=True))
    return dict(full_selection=full, cases=results)


def guards(device):
    model, executor, _, _, _, scheduler, _ = fixture(device, torch.float32)
    errors = {}
    for label, statistics in [
        ("missing_variance", {"mean": torch.zeros((2, 32))}),
        ("wrong_shape", {"mean": torch.zeros((2, 7, 32)), "variance": torch.ones((2, 32))}),
        ("negative_variance", {"mean": torch.zeros((2, 32)), "variance": -torch.ones((2, 32))}),
        ("nonfinite_mean", {"mean": torch.full((2, 32), float("nan")), "variance": torch.ones((2, 32))}),
        ("nonfinite_variance", {"mean": torch.zeros((2, 32)), "variance": torch.full((2, 32), float("inf"))}),
    ]:
        errors[label] = rejected(lambda statistics=statistics: executor.prepare_tables(scheduler.timesteps, statistics))
    executor.mode = "shift"
    errors["unsupported_shift_mode"] = rejected(lambda: executor.begin(6, 4, [1], [0, 7]))
    executor.mode = "affine"
    executor.begin(6, 4, [1], [0, 7])
    executor.step = 1
    cached = torch.zeros((1, 6, 2, 16), device=device)
    errors["uninitialized_anchor"] = rejected(lambda: executor.read_value_cache(0, cached, torch.tensor([1], device=device)))
    executor.age.fill_(0)
    executor.mode = "none"
    errors["mode_changed_mid_request"] = rejected(lambda: executor.read_value_cache(0, cached, torch.tensor([1], device=device)))
    model.zero_cond_t = True
    errors["zero_cond_t"] = rejected(lambda: executor.prepare_tables(scheduler.timesteps))
    model.zero_cond_t = False
    errors["wrong_schedule_length"] = rejected(lambda: executor.prepare_tables(scheduler.timesteps[:4]))
    bare_model = QwenImageTransformer2DModel(**CONFIG).to(device)
    bare = ValueChannelExecutor(bare_model)
    bare.mode = "affine"
    errors["missing_tables"] = rejected(lambda: bare.begin(6, 4, [1], [0, 7]))
    return errors


def main():
    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    FORWARD_COUNTS.clear()
    output = ROOT / "data" / "channel_tiny_correctness.json"
    source_paths = [Path(__file__), Path(__file__).with_name("value_channel_correction.py"),
                    STAGE4_SCRIPTS / "local_executor.py", STAGE4_SCRIPTS / "value_time_correction.py"]
    report = dict(status="running", device=str(device), torch_version=torch.__version__,
                  started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), tiny_model_config=CONFIG,
                  real_20b_model_forwards=0, hypothesis_only=True, no_train8_statistics_used=True,
                  actual_tiny_forward_calls=forward_accounting(), expected_completed_tiny_forward_calls=172,
                  forward_accounting_scope="Whole tiny Transformer calls; coefficient, theory and precompute submodules excluded",
                  sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}, tests=[])
    save(output, report)
    tests = [("closed_form_nonzero_mean_variance_bias_anchor_center", lambda: closed_form(device)),
             ("monte_carlo_ridge_optimal_mse", lambda: monte_carlo(device)),
             ("collinear_ridge_bias_and_zero_variance", lambda: collinear_and_degenerate(device)),
             ("unsupported_states_and_statistics", lambda: guards(device))]
    for dtype, label in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        tests.extend([(f"{label}_tables_statistics_schedule", lambda dtype=dtype: table_checks(device, dtype)),
                      (f"{label}_eight_full_steps_all_modes", lambda dtype=dtype: trajectory(device, dtype, True)),
                      (f"{label}_dynamic_real_anchors_no_pollution", lambda dtype=dtype: trajectory(device, dtype, False))])
    started = time.perf_counter()
    with torch.inference_mode():
        for name, function in tests:
            row, before_counts = dict(name=name), forward_accounting()
            test_started = time.perf_counter()
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
                **{field: after_counts[field] - before_counts[field] for field in ["invoked", "completed", "failed"]})
            report["tests"].append(row)
            report["actual_tiny_forward_calls"] = after_counts
            save(output, report)
            print(json.dumps(dict(name=name, status=row["status"], error=row.get("error")), ensure_ascii=False), flush=True)
    failures = [row["name"] for row in report["tests"] if row["status"] != "pass"]
    if not failures and forward_accounting()["completed"] != report["expected_completed_tiny_forward_calls"]:
        failures.append("actual_tiny_forward_call_count")
    report.update(status="fail" if failures else "pass", failed_tests=failures,
                  elapsed_seconds=time.perf_counter() - started,
                  completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    save(output, report)
    assert not failures, f"Channel correction tiny checks failed: {failures}"


if __name__ == "__main__":
    main()
