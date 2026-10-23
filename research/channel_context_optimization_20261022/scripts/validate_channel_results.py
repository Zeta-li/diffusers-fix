import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ["F", "R2", "CHANNEL"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def find_root(root, supplied, names):
    candidates = [supplied] if supplied else []
    candidates += [root.parent / name for name in names]
    for value in candidates:
        if value is not None and Path(value).is_dir():
            return Path(value).resolve()
    raise FileNotFoundError(f"Cannot resolve experiment root from {names}")


def source_path(value, root, stage4):
    path = Path(value)
    if not path.is_absolute() and len(path.parts) == 1:
        return root / "scripts" / path.name
    if "rotation_context_optimization_20261022" in path.parts or "背景轮转与上下文优化_20261022" in path.parts:
        return stage4 / "scripts" / path.name
    if "channel_context_optimization_20261022" in path.parts or root.name in path.parts:
        return root / "scripts" / path.name
    raise AssertionError(f"Unrecognized source provenance path: {value}")


def verify_sources(items, root, stage4):
    verified = []
    for original, expected in items.items():
        path = source_path(original, root, stage4)
        assert path.is_file() and sha(path) == expected, f"Source mismatch: {original} -> {path}"
        verified.append(dict(recorded_path=original, resolved_local_path=str(path), sha256=expected))
    return verified


def bf16_round(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def validate_table(table):
    shapes = {"alpha_shape": [60, 8, 8, 24, 128], "delta_shape": [60, 8, 8, 24, 128],
              "projected_shift_shape": [60, 8, 24, 128], "rho_shape": [60, 8, 8]}
    assert table["status"] == "prepared" and table["finite"] is True
    assert all(table[key] == value for key, value in shapes.items())
    expected_bytes = 4 * sum(math.prod(shape) for shape in shapes.values())
    assert table["table_bytes"] == expected_bytes == 100285440
    assert table["table_dtype"] == "torch.float32" and table["model_dtype"] == "torch.bfloat16"
    assert math.isfinite(table["prepare_seconds"]) and table["prepare_seconds"] > 0
    for key in ["alpha_min", "alpha_max", "alpha_causal_min", "alpha_causal_max", "rho_causal_min", "rho_causal_max", "delta_min", "delta_max", "regularizer_min", "regularizer_max"]:
        assert math.isfinite(table[key]), f"Nonfinite table metadata: {key}"
    assert table["regularizer_min"] > 0 and table["regularizer_max"] >= table["regularizer_min"]
    assert table["ridge_relative"] == 1e-6 and table["ridge_floor"] == 1e-12
    assert table["alpha_clipped"] is False and table["hyperparameter_search_performed"] is False
    assert table["diagonal_alpha"] == 1 and table["diagonal_delta"] == 0
    assert table["shift_projection_includes_linear_bias"] is True and table["feature_mean_projection_has_bias"] is False
    assert table["statistics_mode"] == "no_sample_mean_zero_variance_one"
    assert table["input_shapes"] is None and table["calibration_performed_here"] is False
    assert table["caller_statistics_source"] is None and table["caller_statistics_source_hash"] is None
    n_statistics = 60 * 8 * 3072
    digest = hashlib.sha256()
    digest.update(b"\x00" * (4 * n_statistics))
    digest.update(struct.pack("<f", 1.0) * n_statistics)
    assert digest.hexdigest() == table["statistics_float32_content_sha256"], "Default mean/variance content SHA mismatch"
    raw, model = table["raw_scheduler_timesteps"], table["model_timesteps"]
    assert len(raw) == len(model) == 8 and all(math.isfinite(value) for value in raw + model)
    assert [bf16_round(bf16_round(value) / 1000) for value in raw] == model
    assert table["preparation_timestep_calls"] == 8
    return dict(bytes=expected_bytes, MiB=expected_bytes / 2**20,
        prepare_seconds=table["prepare_seconds"], statistics_prior_content_independently_verified=True,
        model_timestep_conversion_independently_verified=True,
        finite_validation_scope="Producer finite flag and numeric metadata checked; actual static table tensors were not exported for an independent elementwise scan.")


def validate_age_and_shapes(row, policy):
    if row["config"] == "F":
        assert not row["age_trace"] and not row["trace"]
        return
    n, condition = row["target_count"], row["condition_count"]
    assert len(row["age_trace"]) == 8
    last = [-1] * n
    for step, entry in enumerate(row["age_trace"]):
        selected = policy["per_step_indices"][str(step)]
        for index in selected:
            last[index] = step
        age = [step - value for value in last]
        histogram = {str(key): value for key, value in sorted(Counter(age).items())}
        assert entry["step"] == step + 1 and entry["selected_target"] == len(selected)
        assert entry["max_read_age"] == max(age) and entry["age_histogram"] == histogram
        assert math.isclose(entry["mean_read_age"], sum(age) / n, abs_tol=1e-6)
        traced = [value for value in row["trace"] if value["step"] == step + 1]
        assert len(traced) == 3
        for value in traced:
            assert value["selected_target"] == len(selected) and value["condition"] == condition
            assert value["image_qkv_tokens"] == value["image_mlp_tokens"] == len(selected) + condition
            assert value["joint_query_tokens"] == value["text_tokens"] + len(selected) + condition
            assert value["joint_kv_tokens"] == value["text_tokens"] + n + condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage4-root", type=Path)
    parser.add_argument("--previous-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    stage4 = find_root(root, args.stage4_root, ["背景轮转与上下文优化_20261022", "rotation_context_optimization_20261022"])
    previous = find_root(root, args.previous_root, ["局部执行与边界验证_20261022", "local_executor_boundary_20261022"])
    data = root / "data"
    write_json(data / "final_validation_summary.json", dict(status="running",
        started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        validator_sha256=sha(Path(__file__)),
        meaning="A failed rerun must not leave an older PASS summary looking current."))
    prereg = read_json(data / "preregistration_optimization.json")
    environment = read_json(data / "environment_optimization.json")
    summary = read_json(data / "summary_optimization.json")
    plan_path = root / "masks/rotation_plan.json"
    rotation = read_json(plan_path)
    protocol_path = root / "01_通道增益优化协议.md"
    assert sha(protocol_path) == prereg["protocol_sha256"]
    prereg_sources = verify_sources(prereg["scripts"], root, stage4)
    assert prereg["configs"] == CONFIGS and prereg["requests"] == 36 and prereg["dit_forwards"] == 288
    assert prereg["oracle_forwards"] == prereg["training_images_used"] == 0
    assert prereg["warmups_per_case_config"] == 1 and prereg["measurements_per_case_config"] == 3
    assert prereg["statistics_mode"] == "mean_zero_unit_variance_prior"
    assert sha(stage4 / "data/summary_optimization.json") == prereg["previous_scalar_summary_sha256"]
    for name, expected in prereg["scripts"].items():
        assert environment["scripts"][name] == expected
    assert environment["configs"] == CONFIGS and environment["repeats"] == 3 and environment["planned_requests"] == 36
    assert environment["extra_dense_oracles_in_performance"] == 0
    assert environment["region_plan_sha256"] == sha(plan_path)
    table = environment["static_correction_tables"]
    table_validation = validate_table(table)
    tiny_path = data / "channel_tiny_correctness.json"
    tiny = read_json(tiny_path)
    assert tiny["status"] == "pass" and not tiny["failed_tests"] and len(tiny["tests"]) == 10
    assert all(value["status"] == "pass" for value in tiny["tests"])
    assert tiny["real_20b_model_forwards"] == 0 and tiny["no_train8_statistics_used"] is True
    calls = tiny["actual_tiny_forward_calls"]
    assert calls["completed"] == calls["invoked"] == tiny["expected_completed_tiny_forward_calls"] == 172
    assert calls["failed"] == 0
    for field in ["completed", "invoked", "failed"]:
        assert sum(value[field] for value in calls["by_path"].values()) == calls[field]
    tiny_sources = verify_sources(tiny["sources"], root, stage4)
    assert summary["status"] == "complete" and not summary["warnings"]
    assert not summary["failed_records"] and not summary["duplicate_successful_run_ids"]
    assert summary["runtime_age_audits_all_passed"] and summary["repeated_output_sha_audit_all_passed"]
    assert summary["execution_order_audit"]["complete_order_matches"]
    assert len(summary["baseline_F_R2_audits"]) == 6
    assert all(value["png_sha256_equal"] and value["pixel_identical"] for value in summary["baseline_F_R2_audits"])
    assert len(summary["quality_rows"]) == 9 and all(value["quality_status"] == "complete" for value in summary["quality_rows"])
    cases = [value["case_id"] for value in read_json(previous / "cases.json")["cases"]]
    old = read_rows(previous / "data/runs.jsonl")
    old_lookup = {(row["case_id"], row["config"]): row for row in old if row.get("status") == "complete" and row.get("kind") == "measurement" and row.get("repeat") == 0 and row["config"] in ["F", "R2"]}
    raw_path = data / "runs_optimization.jsonl"
    raw = read_rows(raw_path)
    assert len(raw) == len({row["run_id"] for row in raw}) == 36
    assert all(row["status"] == "complete" for row in raw)
    assert sum(row["kind"] == "warmup" for row in raw) == 9
    assert sum(row["kind"] == "measurement" for row in raw) == 27
    assert sum(row["dit_calls"] for row in raw) == 288
    expected_order = [(cid, config, "warmup", 0) for cid in cases for config in CONFIGS]
    for repeat in range(3):
        for case_index, cid in enumerate(cases):
            offset = (repeat + case_index) % len(CONFIGS)
            expected_order += [(cid, config, "measurement", repeat) for config in CONFIGS[offset:] + CONFIGS[:offset]]
    assert [(row["case_id"], row["config"], row["kind"], row["repeat"]) for row in raw] == expected_order
    controls = 0
    for row in raw:
        assert row["case_id"] in cases and row["config"] in CONFIGS and row["dit_calls"] == 8
        assert sha(root / row["output"]) == row["output_sha256"]
        assert row["model_timestep_contract_exact"] is True
        assert row["observed_model_timesteps"] == table["model_timesteps"]
        assert row["timesteps"] == table["raw_scheduler_timesteps"] and len(row["sigmas"]) == 9
        assert math.isfinite(row["e2e_seconds"]) and row["e2e_seconds"] > 0
        policy = rotation["cases"][row["case_id"]]["configs"]["F" if row["config"] == "F" else "R2"]
        assert row["full_steps_1based"] == policy["full_steps_1based"]
        validate_age_and_shapes(row, policy)
        if row["config"] in ["F", "R2"]:
            control = old_lookup[(row["case_id"], row["config"])]
            assert row["old_output_exact"] is True
            assert sha(previous / control["output"]) == control["output_sha256"] == row["output_sha256"]
            assert row["initial_latent_sha256"] == control["initial_latent_sha256"]
            assert row["sigmas"] == control["sigmas"]
            controls += 1
    assert controls == 24
    for cid in cases:
        f = next(row for row in raw if row["case_id"] == cid and row["config"] == "F")
        for config in CONFIGS:
            group = [row for row in raw if row["case_id"] == cid and row["config"] == config]
            assert len(group) == 4 and len({row["output_sha256"] for row in group}) == 1
            assert sum(row["kind"] == "warmup" for row in group) == 1
            assert {row["repeat"] for row in group if row["kind"] == "measurement"} == {0, 1, 2}
            assert all(row["initial_latent_sha256"] == f["initial_latent_sha256"] and row["sigmas"] == f["sigmas"] for row in group)
    result = dict(status="pass", scope="Artifact integrity and execution correctness, not algorithm quality acceptance",
        validated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), total_requests=36, measured_requests=27, warmup_requests=9,
        total_real_dit_forwards=288, extra_dense_oracles_in_performance=0, training_images_used=0,
        prior_F_R2_exact_output_controls=controls, model_timestep_contract_checks=36,
        independent_request_age_contract_checks=36,
        local_requests_with_8_step_age_checks=sum(row["config"] != "F" for row in raw),
        native_F_empty_cache_trace_checks=sum(row["config"] == "F" for row in raw),
        measured_summary_age_contract_checks=len(summary["runtime_age_audits"]),
        independent_execution_order_matches=True, all_output_hashes_verified=True, all_group_repeated_hashes_identical=True,
        tiny_model_forwards=172, tiny_test_count=10, failed_real_model_requests=0, failed_tiny_tests=0,
        table_validation=table_validation, statistics_mode=table["statistics_mode"],
        prior_sources=prereg_sources, tiny_sources_with_remote_to_local_mapping=tiny_sources,
        peak_allocated_GiB=max(row["peak_allocated_bytes"] for row in raw) / 2**30,
        request_seconds_sum=sum(row["e2e_seconds"] for row in raw), model_load_and_fuse_seconds=environment["model_load_and_fuse_seconds"],
        cold_table_preparation_not_a_DiT=True, candidate_numeric_quality_pass_all_cases=summary["configs"]["CHANNEL"]["quality_gate_pass_all_cases"],
        deployment_joint_target_inferred=False, runs_sha256=sha(raw_path), preregistration_sha256=sha(data / "preregistration_optimization.json"),
        validator_sha256=sha(Path(__file__)), artifact_manifest_scope="Snapshot of current local files; rerun after any further document edits or synchronization.")
    write_json(data / "final_validation_summary.json", result)
    artifact_manifest = {str(path.relative_to(root)): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json" and "__pycache__" not in path.parts and path.name != ".DS_Store"}
    write_json(data / "artifact_manifest.json", artifact_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
