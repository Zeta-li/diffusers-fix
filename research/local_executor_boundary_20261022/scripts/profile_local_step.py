import hashlib
import importlib.metadata
import io
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image

import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline

from local_executor import LocalTargetExecutor

PILOT = Path("/workspace/diffusers-fix/research/lightning8_preflight_20261022")
sys.path.insert(0, str(PILOT / "scripts"))
from run_pilot import BASE, LORA, LORA_FILE, SEED, SHIFT3, tensor_hash


ROOT = Path(__file__).resolve().parent.parent


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def numeric_attribute(event, names):
    for name in names:
        try:
            value = getattr(event, name)
        except AttributeError:
            continue
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value, name
    return None, None


def operator_category(name):
    if not name.startswith("aten::"):
        return "non_aten_observer_or_kernel"
    if "attention" in name:
        return "attention"
    if name in {"aten::linear", "aten::addmm", "aten::mm", "aten::bmm", "aten::matmul", "aten::_scaled_mm", "aten::addbmm", "aten::baddbmm"}:
        return "matmul_linear"
    if name.startswith("aten::index_copy"):
        return "index_copy"
    if name.startswith("aten::index_select"):
        return "index_select"
    if name == "aten::cat":
        return "cat"
    return "other_aten"


def collect_key_averages(profiler):
    rows = []
    grouped = {}
    for event in profiler.key_averages(group_by_input_shape=True):
        key = str(event.key)
        self_device, self_device_source = numeric_attribute(event, ["self_device_time_total", "self_cuda_time_total"])
        device_total, device_total_source = numeric_attribute(event, ["device_time_total", "cuda_time_total"])
        self_cpu, self_cpu_source = numeric_attribute(event, ["self_cpu_time_total"])
        cpu_total, cpu_total_source = numeric_attribute(event, ["cpu_time_total"])
        row = dict(
            key=key,
            count=int(event.count),
            input_shapes=getattr(event, "input_shapes", None),
            category=operator_category(key),
            self_device_time_total_us=self_device,
            device_time_total_us=device_total,
            self_cpu_time_total_us=self_cpu,
            cpu_time_total_us=cpu_total,
            device_type=str(getattr(event, "device_type", "unavailable")),
            attribute_sources=dict(self_device=self_device_source, device_total=device_total_source,
                                   self_cpu=self_cpu_source, cpu_total=cpu_total_source),
        )
        rows.append(row)
        if row["category"] == "non_aten_observer_or_kernel":
            continue
        group = grouped.setdefault(row["category"], dict(count=0, shape_groups=0, self_device_time_total_us=0.0,
                                                        self_cpu_time_total_us=0.0, missing_self_device_groups=0, keys=set()))
        group["count"] += row["count"]
        group["shape_groups"] += 1
        group["keys"].add(key)
        group["self_cpu_time_total_us"] += self_cpu or 0.0
        if self_device is None:
            group["missing_self_device_groups"] += 1
        else:
            group["self_device_time_total_us"] += self_device
    for value in grouped.values():
        value["keys"] = sorted(value["keys"])
    rows.sort(key=lambda row: row["self_device_time_total_us"] or 0.0, reverse=True)
    assert rows and any(row["input_shapes"] for row in rows)
    return dict(
        grouping="profiler.key_averages(group_by_input_shape=True)",
        time_unit="microseconds",
        category_aggregation="Sum operator self times for aten events only; non-aten kernels and observer scopes are excluded to avoid counting the same kernels twice",
        aggregation_limit="CPU and GPU work may overlap; sums are not request wall time. Inclusive operator totals are retained per row and are not summed by category.",
        categories=grouped,
        shape_group_count=len(rows),
        device_timing_available=any((row["self_device_time_total_us"] or 0.0) > 0 for row in rows),
        rows=rows,
    )


def profile_request(pipe, executor, case, sample, config, region_indices, directory, record):
    prefix = directory / config
    request_path = prefix.with_name(prefix.name + "_request.json")
    trace_path = prefix.with_name(prefix.name + "_chrome_trace.json")
    averages_path = prefix.with_name(prefix.name + "_key_averages.json")
    output_path = prefix.with_name(prefix.name + "_output.png")
    record.update(status="running", started_at=timestamp(), config=config, kind="observer_profiler_diagnostic",
                  included_in_main_performance=False, case_id=case["case_id"], seed=SEED,
                  initial_latent_sha256=sample["latent_hash"], target_hw=[sample["height"], sample["width"]],
                  target_count=sample["target_count"], condition_count=sample["condition_count"],
                  partial_target_count=len(region_indices) if config == "R0" else sample["target_count"],
                  full_steps_1based=[1, 8] if config == "R0" else list(range(1, 9)),
                  profiled_transformer_call_1based=2, planned_pipeline_requests=1, planned_dit_calls=8,
                  observer_overhead=True, timing_valid_for_main_comparison=False)
    write_json(request_path, record)
    original_forward = pipe.transformer.forward
    calls = 0
    captured = {}

    def wrapped(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls != 2:
            return original_forward(*args, **kwargs)
        assert executor.step == 1
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            with torch.profiler.record_function(f"diagnostic_{config}_transformer_call_2"):
                output = original_forward(*args, **kwargs)
                torch.cuda.synchronize()
        captured["profiler"] = profiler
        return output

    pipe.transformer.forward = wrapped
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(SHIFT3)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        executor.begin(sample["target_count"], sample["condition_count"], region_indices,
                       full_steps=list(range(8)) if config == "F" else [0, 7], native=config == "F")
        with torch.inference_mode():
            image = pipe(
                image=sample["image"],
                prompt=case["prompt"],
                negative_prompt=" ",
                true_cfg_scale=1.0,
                num_inference_steps=8,
                height=sample["height"],
                width=sample["width"],
                generator=torch.Generator(device="cuda").manual_seed(SEED),
                latents=sample["latent"].clone(),
            ).images[0]
        execution_trace = executor.trace.copy()
        cache_bytes = executor.cache_bytes()
        assert calls == executor.step == 8
        assert "profiler" in captured
        if config == "R0":
            step_two = [row for row in execution_trace if row["step"] == 2]
            assert step_two and all(row["selected_target"] == len(region_indices) for row in step_two)
        executor.clear()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        profiler = captured.pop("profiler")
        export_started = time.perf_counter()
        profiler.export_chrome_trace(str(trace_path))
        averages = collect_key_averages(profiler)
        averages.update(config=config, case_id=case["case_id"], profiled_call_1based=2,
                        observer_overhead=True, not_main_performance=True,
                        record_shapes=True, profile_memory=False)
        write_json(averages_path, averages)
        export_seconds = time.perf_counter() - export_started
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        output_path.write_bytes(encoded.getvalue())
        output_sha = file_hash(output_path)
        main_rows = [row for row in read_rows(ROOT / "data" / "runs.jsonl")
                     if row.get("status") == "complete" and row.get("kind") == "measurement"
                     and row.get("config") == config and row.get("case_id") == case["case_id"]]
        assert main_rows
        reference = main_rows[0]
        reference_image = Image.open(ROOT / reference["output"]).convert("RGB")
        actual_image = image.convert("RGB")
        record.update(
            status="complete",
            completed_at=timestamp(),
            actual_pipeline_requests=1,
            actual_dit_calls=calls,
            profiled_dit_calls=1,
            nonprofiled_dit_calls=7,
            observer_e2e_seconds=elapsed,
            profiler_export_seconds_outside_request=export_seconds,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            cache_bytes=cache_bytes,
            execution_trace=execution_trace,
            profiled_transformer_step_trace=[row for row in execution_trace if row["step"] == 2],
            output=str(output_path.relative_to(ROOT)),
            output_sha256=output_sha,
            reference_main_run_id=reference["run_id"],
            reference_main_output_sha256=reference["output_sha256"],
            output_matches_main_sha256=output_sha == reference["output_sha256"],
            rgb_pixels_equal_to_main=actual_image.size == reference_image.size and actual_image.tobytes() == reference_image.tobytes(),
            chrome_trace=str(trace_path.relative_to(ROOT)),
            chrome_trace_sha256=file_hash(trace_path),
            key_averages=str(averages_path.relative_to(ROOT)),
            key_averages_sha256=file_hash(averages_path),
            operator_category_summary=averages["categories"],
            scheduler_config=dict(pipe.scheduler.config),
            timesteps=pipe.scheduler.timesteps.cpu().tolist(),
            sigmas=pipe.scheduler.sigmas.cpu().tolist(),
        )
    except Exception as error:
        record.update(status="failed", failed_at=timestamp(), actual_pipeline_requests=1,
                      actual_dit_calls=calls, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        pipe.transformer.forward = original_forward
        executor.clear()
        write_json(request_path, record)
        with (ROOT / "diagnostics" / "profiler_requests.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(dict(config=config, status=record["status"], calls=calls,
                          observer_e2e_seconds=record["observer_e2e_seconds"], trace=record["chrome_trace"]), ensure_ascii=False), flush=True)


def main():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    assert torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities()
    status = json.loads((ROOT / "status.json").read_text())
    assert status["phase"] == "complete", "Wait for the main local pilot to finish before profiling"
    performance_plan = json.loads((ROOT / "data" / "performance_plan.json").read_text())
    main_rows = read_rows(ROOT / "data" / "runs.jsonl")
    primary_ids = {row["run_id"] for row in main_rows if row.get("status") == "complete"
                   and row.get("config") in performance_plan["configs"]
                   and row.get("kind") in {"warmup", "measurement"}}
    assert len(primary_ids) == performance_plan.get("primary_requests", performance_plan.get("planned_requests"))
    assert json.loads((ROOT / "data" / "real_correctness.json").read_text())["status"] == "passed"
    case = next(case for case in json.loads((ROOT / "cases.json").read_text())["cases"] if case["case_id"] == "remove_hair")
    manifest_path = ROOT / "masks" / "regions.json"
    manifest = json.loads(manifest_path.read_text())
    region = manifest["cases"][case["case_id"]]
    r0 = region["regions"]["R0"]
    assert r0["target_active_fraction"] == min(value["regions"]["R0"]["target_active_fraction"] for value in manifest["cases"].values())
    reference_rows = [row for row in read_rows(PILOT / "data" / "runs.jsonl")
                      if row.get("status") == "complete" and row.get("run_id") == "E_lightning8_fused__remove_hair__measurement_0"]
    assert len(reference_rows) == 1
    reference = reference_rows[0]
    latent_path = PILOT / "latents" / "remove_hair.pt"
    latent_cpu = torch.load(latent_path, map_location="cpu", weights_only=True)
    latent_sha = tensor_hash(latent_cpu)
    assert latent_sha == reference["initial_latent_sha256"]
    image_path = ROOT / case["image"]
    assert file_hash(image_path) == region["source_sha256"]
    target_count = latent_cpu.shape[1]
    condition_count = reference["first_cond_shapes"]["hidden_states"][1] - target_count
    assert target_count == region["n_target_tokens"]
    assert condition_count == region["n_condition_tokens_always_computed"]
    assert reference["target_hw"] == region["target_hw"]
    directory = ROOT / "diagnostics" / ("profile_local_step_" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S"))
    directory.mkdir(parents=True, exist_ok=False)
    packages = {}
    for name in ["torch", "transformers", "peft", "huggingface_hub", "torchao", "safetensors", "accelerate"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    model_source = Path(diffusers.__file__)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=model_source.parent, text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    report = dict(
        status="running", started_at=timestamp(), kind="two_request_operator_diagnostic",
        included_in_main_performance=False, observer_overhead=True,
        timing_interpretation="Observer e2e includes profiler setup, synchronization and collection; do not use it to estimate native/local speedup. Trace export and JSON/PNG writing occur outside request timing.",
        profiler_scope="Only transformer call 2 of each 8-step request; calls 1 and 3-8 execute without a profiler",
        synchronization_policy="Synchronize before entering the call-2 profiler to drain call 1; synchronize inside the observer scope after call 2 to finish captured GPU kernels",
        planned_extra_requests=2, planned_extra_dit_calls=16, planned_profiled_dit_calls=2,
        planned_nonprofiled_dit_calls=14, warmup_requests=0, main_primary_requests_already_complete=len(primary_ids),
        main_runs_sha256_before=file_hash(ROOT / "data" / "runs.jsonl"),
        case_id=case["case_id"], prompt=case["prompt"], image_sha256=file_hash(image_path),
        region_manifest_sha256=file_hash(manifest_path), r0_fraction=r0["target_active_fraction"],
        initial_latent_sha256=latent_sha, seed=SEED, base=BASE, lora=LORA, lora_file=LORA_FILE,
        environment=dict(python=sys.version, packages=packages, diffusers_source=diffusers.__file__,
                         diffusers_version=diffusers.__version__, git_commit=commit,
                         gpu_name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                         dtype="bfloat16", fused_lora=True, vae_tiling=True, attention="native PyTorch SDPA",
                         cfg=1.0, steps=8, cpu_offload=False, tf32=torch.backends.cuda.matmul.allow_tf32,
                         cross_request_condition_cache=False),
        source_hashes=dict(profile_script=file_hash(Path(__file__)), executor=file_hash(Path(__file__).with_name("local_executor.py"))),
        requests=[],
    )
    report_path = directory / "summary.json"
    write_json(report_path, report)
    try:
        started = time.perf_counter()
        pipe = QwenImageEditPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.to("cuda")
        pipe.vae.enable_tiling()
        pipe.transformer.set_attention_backend("native")
        pipe.set_progress_bar_config(disable=True)
        for module in [pipe.transformer, pipe.text_encoder, pipe.vae]:
            module.eval()
        pipe.load_lora_weights(LORA, weight_name=LORA_FILE, adapter_name="lightning8", local_files_only=True)
        adapter_dtypes = dict(Counter(str(parameter.dtype) for name, parameter in pipe.transformer.named_parameters() if "lora_" in name))
        assert adapter_dtypes and set(adapter_dtypes) == {"torch.bfloat16"}
        pipe.fuse_lora(adapter_names=["lightning8"], safe_fusing=True)
        pipe.unload_lora_weights()
        assert not any("lora_" in name for name, _ in pipe.transformer.named_parameters())
        assert not pipe.transformer.zero_cond_t
        torch.cuda.synchronize()
        report["model_load_and_fuse_seconds"] = time.perf_counter() - started
        report["adapter_dtypes_before_fusion"] = adapter_dtypes
        report["environment"]["attention_processors"] = sorted({type(processor).__name__ for processor in pipe.transformer.attn_processors.values()})
        write_json(report_path, report)
        sample = dict(latent=latent_cpu.to("cuda"), latent_hash=latent_sha,
                      image=Image.open(image_path).convert("RGB"), height=reference["target_hw"][0],
                      width=reference["target_hw"][1], target_count=target_count, condition_count=condition_count)
        executor = LocalTargetExecutor(pipe.transformer)
        executor.install()
        for config in ["F", "R0"]:
            request_record = dict(diagnostic_attempt=directory.name)
            report["requests"].append(request_record)
            profile_request(pipe, executor, case, sample, config, r0["indices"], directory, request_record)
            write_json(report_path, report)
        assert sum(row["actual_pipeline_requests"] for row in report["requests"]) == 2
        assert sum(row["actual_dit_calls"] for row in report["requests"]) == 16
        assert sum(row["profiled_dit_calls"] for row in report["requests"]) == 2
        report.update(status="complete", completed_at=timestamp(), actual_extra_requests=2,
                      actual_extra_dit_calls=16, actual_profiled_dit_calls=2,
                      main_runs_sha256_after=file_hash(ROOT / "data" / "runs.jsonl"))
        assert report["main_runs_sha256_after"] == report["main_runs_sha256_before"]
    except Exception as error:
        report.update(status="failed", failed_at=timestamp(), error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write_json(report_path, report)
        raise
    write_json(report_path, report)
    print(json.dumps(dict(status=report["status"], summary=str(report_path),
                          actual_extra_requests=2, actual_extra_dit_calls=16), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
