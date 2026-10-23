import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from attribution_executor import AttributionExecutor, mix_prediction

ROOT = Path(__file__).resolve().parent.parent
PREVIOUS = Path("/workspace/diffusers-fix/research/local_executor_boundary_20261022")
if not PREVIOUS.exists():
    PREVIOUS = ROOT.parent / "局部执行与边界验证_20261022"
sys.path.append(str(PREVIOUS / "scripts"))
from check_tiny_executor import TINY_CONFIG


TARGETS = 6
CONDITIONS = 4
SHAPES = [[(1, 2, 3), (1, 2, 2)]]
SELECTIONS = {0: [0, 1, 2, 3, 4, 5], 1: [4, 1], 2: [1], 3: [4, 1], 4: [0, 3], 5: [3], 6: [2, 5], 7: [0, 1, 2, 3, 4, 5]}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def cpu_copy(tensor):
    return tensor.detach().cpu().clone()


def snapshot(executor):
    return dict(
        step=executor.step,
        state_spec=copy.deepcopy(executor.state_spec),
        age=cpu_copy(executor.age),
        trace=copy.deepcopy(executor.trace),
        velocity=None if executor.velocity is None else cpu_copy(executor.velocity),
        cache={index: (cpu_copy(key), cpu_copy(value)) for index, (key, value) in executor.layer_cache.items()},
    )


def assert_snapshot(before, executor):
    assert before["step"] == executor.step
    assert before["state_spec"] == executor.state_spec
    assert before["trace"] == executor.trace
    assert torch.equal(before["age"], cpu_copy(executor.age))
    if before["velocity"] is None:
        assert executor.velocity is None
    else:
        assert torch.equal(before["velocity"], cpu_copy(executor.velocity))
    assert set(before["cache"]) == set(executor.layer_cache)
    for index, (key, value) in executor.layer_cache.items():
        old_key, old_value = before["cache"][index]
        assert torch.equal(old_key, cpu_copy(key))
        assert torch.equal(old_value, cpu_copy(value))


def tensor_sha(tensor):
    return hashlib.sha256(cpu_copy(tensor).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def mix_and_orthogonality(device, dtype):
    local = (torch.arange(80, dtype=torch.float32).reshape(1, 10, 8) / 8).to(device=device, dtype=dtype)
    dense = (local.float() * -0.75 + 0.25).to(dtype)
    selected = torch.tensor([4, 1], device=device)
    inactive = torch.tensor([0, 2, 3, 5], device=device)
    local_original = local.clone()
    dense_original = dense.clone()
    predictions = {mode: mix_prediction(local, dense, selected, TARGETS, mode) for mode in ["D", "B", "V", "K"]}
    assert torch.equal(predictions["B"], local)
    assert torch.equal(predictions["D"], dense)
    assert torch.equal(predictions["V"].index_select(1, selected), dense.index_select(1, selected))
    assert torch.equal(predictions["V"].index_select(1, inactive), local.index_select(1, inactive))
    assert torch.equal(predictions["K"].index_select(1, selected), local.index_select(1, selected))
    assert torch.equal(predictions["K"].index_select(1, inactive), dense.index_select(1, inactive))
    assert torch.equal(predictions["V"][:, TARGETS:], dense[:, TARGETS:])
    assert torch.equal(predictions["K"][:, TARGETS:], dense[:, TARGETS:])
    residuals = {mode: predictions[mode][:, :TARGETS].float() - dense[:, :TARGETS].float() for mode in ["B", "V", "K"]}
    assert torch.equal(residuals["B"], residuals["V"] + residuals["K"])
    assert torch.count_nonzero(residuals["V"].index_select(1, selected)) == 0
    assert torch.count_nonzero(residuals["K"].index_select(1, inactive)) == 0
    cross = float((residuals["V"] * residuals["K"]).sum())
    assert cross == 0.0
    energies = {mode: float(value.square().sum()) for mode, value in residuals.items()}
    assert abs(energies["B"] - energies["V"] - energies["K"]) <= max(1e-5, energies["B"] * 1e-5)
    for mode, output in predictions.items():
        assert output.data_ptr() not in {local.data_ptr(), dense.data_ptr()}
        other_hashes = {key: tensor_sha(value) for key, value in predictions.items() if key != mode}
        output[0, 0, 0] += 1
        assert torch.equal(local, local_original)
        assert torch.equal(dense, dense_original)
        assert other_hashes == {key: tensor_sha(value) for key, value in predictions.items() if key != mode}
    try:
        mix_prediction(local, dense, selected, TARGETS, "invalid")
    except ValueError:
        invalid_mode_rejected = True
    else:
        raise AssertionError("Invalid attribution mode accepted")
    return dict(noncontiguous_selected=[4, 1], residual_energy=energies, cross_term=cross,
                target_only_decomposition=True, no_alias_after_mutation=True,
                invalid_mode_rejected=invalid_mode_rejected)


def flow_scheduler(device):
    scheduler = FlowMatchEulerDiscreteScheduler(base_shift=math.log(3), max_shift=math.log(3),
        base_image_seq_len=256, max_image_seq_len=8192, use_dynamic_shifting=True,
        time_shift_type="exponential", shift_terminal=None)
    scheduler.set_timesteps(sigmas=[1 - step / 8 for step in range(8)], device=device, mu=math.log(3))
    return scheduler


def actual_tiny_trajectory(device, dtype, mode):
    torch.manual_seed(1873)
    model = QwenImageTransformer2DModel(**TINY_CONFIG).to(device=device, dtype=dtype).eval()
    model.set_attention_backend("native")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(5817)
    raw = torch.randn((1, TARGETS + CONDITIONS, 8), generator=generator).to(device=device, dtype=dtype)
    text = torch.randn((1, 5, 24), generator=generator).to(device=device, dtype=dtype)
    mask = torch.tensor([[True, False, True, True, False]], device=device)
    scheduler = flow_scheduler(device)
    executor = AttributionExecutor(model)
    executor.install()
    executor.begin_attribution(TARGETS, CONDITIONS, [4, 1], mode, scheduler.sigmas)
    for step, indices in SELECTIONS.items():
        executor.indices_cpu[step] = torch.tensor(indices, dtype=torch.long)
        executor.indices[step] = executor.indices_cpu[step].to(device)
    native_forward = executor.original_forward
    oracle_records = []

    def audited_dense(*args, **kwargs):
        before = snapshot(executor)
        input_copies = {key: cpu_copy(kwargs[key]) for key in ["hidden_states", "encoder_hidden_states", "encoder_hidden_states_mask", "timestep"]}
        result = native_forward(*args, **kwargs)
        assert_snapshot(before, executor)
        for key, value in input_copies.items():
            assert torch.equal(value, cpu_copy(kwargs[key]))
        dense = result.sample if kwargs.get("return_dict", True) else result[0]
        oracle_records.append(dict(dense=cpu_copy(dense), checked_layers=len(executor.layer_cache),
                                   step=executor.step, complete_cache_unchanged=True))
        return result

    executor.original_forward = audited_dense
    latent = raw[:, :TARGETS].clone()
    condition = raw[:, TARGETS:].clone()
    expected_history = None
    dense_token_four_after_step_two = None
    local_token_four_after_step_two = None
    rows = []
    for step, timestep in enumerate(scheduler.timesteps):
        selected_cpu = torch.tensor(SELECTIONS[step], dtype=torch.long)
        inactive_cpu = torch.tensor([index for index in range(TARGETS) if index not in SELECTIONS[step]], dtype=torch.long)
        previous_history = None if expected_history is None else expected_history.clone()
        current = torch.cat([latent, condition], dim=1)
        result = model(hidden_states=current, encoder_hidden_states=text, encoder_hidden_states_mask=mask,
                       timestep=timestep.expand(1).to(device=device, dtype=dtype) / 1000,
                       img_shapes=SHAPES, return_dict=False)[0]
        output = cpu_copy(result)
        capture = executor.captures[-1]
        local = capture["local"]
        dense = capture["dense"]
        assert torch.equal(capture["selected"], selected_cpu)
        assert torch.equal(capture["current_latent"], cpu_copy(latent))
        assert torch.equal(capture["returned"], output[:, :TARGETS])
        assert len(oracle_records) == step + 1
        dense_full = oracle_records[-1]["dense"]
        assert torch.equal(dense, dense_full[:, :TARGETS])
        if mode == "B":
            expected_output = local.clone()
        elif mode == "V":
            expected_output = local.clone()
            expected_output.index_copy_(1, selected_cpu, dense.index_select(1, selected_cpu))
            assert torch.equal(output[:, TARGETS:], dense_full[:, TARGETS:])
        else:
            expected_output = dense.clone()
            expected_output.index_copy_(1, selected_cpu, local.index_select(1, selected_cpu))
            assert torch.equal(output[:, TARGETS:], dense_full[:, TARGETS:])
        assert torch.equal(output[:, :TARGETS], expected_output)
        fresh_history = dense if mode == "V" else local
        if previous_history is None:
            expected_history = fresh_history.clone()
        else:
            expected_history = previous_history.clone()
            expected_history.index_copy_(1, selected_cpu, fresh_history.index_select(1, selected_cpu))
            assert torch.equal(local.index_select(1, inactive_cpu), previous_history.index_select(1, inactive_cpu))
        assert torch.equal(cpu_copy(executor.velocity[:, :TARGETS]), expected_history)
        if mode == "V":
            assert torch.equal(cpu_copy(executor.velocity[:, TARGETS:]), dense_full[:, TARGETS:])
        combined = local.float() - dense.float()
        velocity_error = combined.clone()
        velocity_error.index_fill_(1, selected_cpu, 0)
        kv_error = combined.clone()
        kv_error.index_fill_(1, inactive_cpu, 0)
        assert torch.equal(combined, velocity_error + kv_error)
        cross = float((velocity_error * kv_error).sum())
        assert cross == 0.0
        energies = dict(combined=float(combined.square().sum()), velocity=float(velocity_error.square().sum()), kv=float(kv_error.square().sum()))
        assert abs(energies["combined"] - energies["velocity"] - energies["kv"]) <= max(1e-5, energies["combined"] * 1e-5)
        assert abs(executor.records[-1]["combined_error"]["squared_error_sum"] - energies["combined"]) <= max(1e-5, energies["combined"] * 1e-5)
        assert executor.records[-1]["orthogonality_cross_term"] == cross == 0.0
        assert executor.records[-1]["factorization_max_abs"] == 0.0
        full = step in [0, 7]
        if full:
            assert torch.equal(local, dense)
            assert torch.equal(output[:, :TARGETS], dense)
        if step == 1:
            dense_token_four_after_step_two = dense[:, 4].clone()
            local_token_four_after_step_two = local[:, 4].clone()
        cross_step_proved = None
        if mode == "V" and step == 2:
            assert torch.equal(output[:, 4], dense_token_four_after_step_two)
            assert torch.equal(cpu_copy(executor.velocity[:, 4]), dense_token_four_after_step_two)
            distinct = not torch.equal(dense_token_four_after_step_two, local_token_four_after_step_two)
            assert distinct, "V writeback test became vacuous because prior dense/local active predictions were equal"
            cross_step_proved = True
        previous_latent = latent.clone()
        latent = scheduler.step(result[:, :TARGETS], timestep, latent, return_dict=False)[0]
        assert not torch.equal(previous_latent, latent)
        assert torch.equal(cpu_copy(latent), capture["expected_post_latent"]), "Analytic Euler prediction differs from actual scheduler.step"
        rows.append(dict(step=step + 1, selected=SELECTIONS[step], full_step=full,
                         dense_oracle_checked_all_cache_layers=oracle_records[-1]["checked_layers"],
                         orthogonality_cross_term=cross, error_energy=energies,
                         expected_post_latent_matches_real_scheduler=True,
                         velocity_history_matches_independent_expected=True,
                         prior_dense_active_reused_after_becoming_inactive=cross_step_proved,
                         output_sha256=tensor_sha(output[:, :TARGETS])))
    assert executor.step == 8
    assert executor.local_calls == executor.oracle_calls == 8
    assert executor.native_calls == 0
    assert len(executor.records) == len(executor.captures) == len(oracle_records) == 8
    assert all(row["complete_cache_unchanged"] and row["checked_layers"] == len(model.transformer_blocks) for row in oracle_records)
    return dict(mode=mode, device=str(device), dtype=str(dtype), steps=rows,
                local_calls=executor.local_calls, oracle_calls=executor.oracle_calls,
                native_calls=executor.native_calls, actual_tiny_model_forwards=16,
                complete_dense_no_side_effect_audits=8, final_latent_sha256=tensor_sha(latent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = ROOT / "data" / "attribution_tiny_correctness.json"
    packages = {}
    for name in ["torch", "diffusers", "peft", "transformers", "huggingface_hub", "torchao"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = dict(status="running", started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  device=str(device), packages=packages, python=sys.version, diffusers_source=diffusers.__file__,
                  tiny_model_config=TINY_CONFIG, target_count=TARGETS, condition_count=CONDITIONS,
                  text_mask=[[1, 0, 1, 1, 0]], dynamic_noncontiguous_selections=SELECTIONS,
                  source_hashes={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                 for name in ["check_attribution_tiny.py", "attribution_executor.py", "local_executor.py"]},
                  cpu_synchronization_noop_for_cpu_tensors=args.device == "cpu",
                  planned_actual_tiny_model_forwards=96, real_20b_model_forwards=0, tests=[])
    save(output, report)
    if args.device == "cuda" and not torch.cuda.is_available():
        report.update(status="fail", failed_tests=["cuda_available"], error="CUDA requested but unavailable")
        save(output, report)
        raise AssertionError(report["error"])
    if args.device == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name()
    tests = []
    for dtype, label in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        tests.append((f"{label}_mix_no_alias_and_orthogonality", lambda dtype=dtype: mix_and_orthogonality(device, dtype)))
        for mode in ["B", "V", "K"]:
            tests.append((f"{label}_{mode}_actual_tiny_eight_steps", lambda dtype=dtype, mode=mode: actual_tiny_trajectory(device, dtype, mode)))
    started = time.perf_counter()
    synchronization = mock.patch("torch.cuda.synchronize", lambda *args, **kwargs: None) if args.device == "cpu" else contextlib.nullcontext()
    with synchronization, torch.inference_mode():
        for name, function in tests:
            test_started = time.perf_counter()
            row = dict(name=name)
            try:
                row.update(status="pass", result=function())
            except Exception as error:
                row.update(status="fail", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            row["seconds"] = time.perf_counter() - test_started
            report["tests"].append(row)
            save(output, report)
            print(json.dumps(dict(name=name, status=row["status"], error=row.get("error"), seconds=row["seconds"]), ensure_ascii=False), flush=True)
    failures = [row["name"] for row in report["tests"] if row["status"] != "pass"]
    report.update(status="fail" if failures else "pass", failed_tests=failures, test_count=len(tests),
                  passed_tests=len(tests) - len(failures), elapsed_seconds=time.perf_counter() - started,
                  completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  completed_actual_tiny_model_forwards=sum(row.get("result", {}).get("actual_tiny_model_forwards", 0) for row in report["tests"]))
    save(output, report)
    print(json.dumps(dict(status=report["status"], output=str(output), failed_tests=failures), ensure_ascii=False), flush=True)
    assert not failures, f"Attribution tiny correctness failed: {failures}"


if __name__ == "__main__":
    main()
