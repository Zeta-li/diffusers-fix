import argparse
import hashlib
import json
import math
import statistics
import struct
import tokenize
import traceback
from collections import Counter
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ["F", "R2", "B125", "B250", "B333", "B500"]
BUDGETS = {"F": None, "R2": Fraction(0), "B125": Fraction(1, 8), "B250": Fraction(1, 4),
           "B333": Fraction(1, 3), "B500": Fraction(1, 2)}
PLANNER_SEED = 2026102301


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def close(first, second, name, absolute=1e-10, relative=1e-10):
    assert isinstance(first, (float, int)) and isinstance(second, (float, int)), name
    assert math.isfinite(first) and math.isfinite(second), name
    assert math.isclose(first, second, abs_tol=absolute, rel_tol=relative), f"{name}: {first} != {second}"


def locate(root, supplied, names):
    candidates = [Path(supplied)] if supplied else []
    candidates += [root.parent / name for name in names]
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError(f"Cannot locate {names}; supply its explicit root")


def histogram(values):
    return {str(key): count for key, count in sorted(Counter(values).items())}


def bf16_round(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def independent_blocks(case_id, height, width, core):
    result = []
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            indices = [row * width + col for row in range(y, min(height, y + 2))
                       for col in range(x, min(width, x + 2)) if row * width + col not in core]
            if indices:
                ty, tx = y // 2, x // 2
                digest = hashlib.sha256(f"{PLANNER_SEED}:{case_id}:{ty}:{tx}".encode()).hexdigest()
                result.append(dict(block_id=f"{ty}:{tx}", tile_row=ty, tile_col=tx,
                                   indices=indices, n_tokens=len(indices), sha256=digest))
    return sorted(result, key=lambda row: (row["sha256"], row["tile_row"], row["tile_col"]))


def independent_policy(n, core, blocks, config):
    core, background = sorted(core), n - len(core)
    fraction = BUDGETS[config]
    cap = background if fraction is None else background * fraction.numerator // fraction.denominator
    previous, per_step, audits = [-1] * n, {}, []
    for step in range(8):
        before = [step - value if value >= 0 else -1 for value in previous]
        full = config == "F" or step in [0, 7]
        if full:
            chosen = blocks.copy()
            selected = list(range(n))
            current_cap = background
        else:
            current_cap, remaining, chosen = cap, cap, []
            ordered = sorted(blocks, key=lambda block: (previous[block["indices"][0]], block["sha256"], block["tile_row"], block["tile_col"]))
            for block in ordered:
                assert len({previous[index] for index in block["indices"]}) == 1
                if len(block["indices"]) <= remaining:
                    chosen.append(block)
                    remaining -= len(block["indices"])
            selected = sorted(core + [index for block in chosen for index in block["indices"]])
        assert len(selected) == len(set(selected)) and set(core).issubset(selected)
        for index in selected:
            previous[index] = step
        assert min(previous) >= 0
        ages = [step - value for value in previous]
        used = len(selected) - len(core)
        assert used <= current_cap
        per_step[str(step)] = selected
        audits.append(dict(step_1based=step + 1, full=full, selected_target_count=len(selected),
            selected_background_count=used, background_cap=current_cap, nominal_partial_background_cap=cap,
            budget_enforced=not full, unused_background_cap=current_cap - used,
            selected_background_block_ids=[block["block_id"] for block in chosen],
            actual_background_fraction=used / background if background else None,
            target_active_fraction=len(selected) / n, max_age_before_update=max(before, default=0),
            age_before_update_histogram=histogram(before), max_age_at_attention_read=max(ages, default=0),
            age_at_attention_read_histogram=histogram(ages)))
    partial = audits[1:7]
    return dict(native=config == "F", indices=per_step["1"],
        full_steps_1based=list(range(1, 9)) if config == "F" else [1, 8], per_step_indices=per_step,
        per_step_audit=audits, budgetmeta=dict(requested_b_fraction=str(fraction) if fraction is not None else None,
            requested_b=float(fraction) if fraction is not None else None, partial_background_cap=cap,
            core_target_fraction=len(core) / n,
            partial_actual_background_fractions=[row["actual_background_fraction"] for row in partial],
            partial_target_active_fractions=[row["target_active_fraction"] for row in partial],
            partial_unused_background_caps=[row["unused_background_cap"] for row in partial],
            full8_target_fraction=sum(row["selected_target_count"] for row in audits) / (8 * n),
            full8_background_fraction=sum(row["selected_background_count"] for row in audits) / (8 * background) if background else None,
            core_preserved_all_steps=True, whole_blocks_only=True,
            fraction_scope="target tokens only; excludes condition/text and is not latency"))


def validate_sources(recorded, root, allow_analysis_archive=False):
    result = []
    for name, expected in recorded.items():
        assert Path(name).name == name and name.endswith(".py"), f"Unexpected source path: {name}"
        path = root / "scripts" / name
        if allow_analysis_archive and name == "analyze_budget.py" and sha(path) != expected:
            folder = root / "attempts/analysis_text_v1"
            archive = folder / "analyze_budget.py.txt"
            provenance = read_json(folder / "provenance.json")
            assert sha(archive) == provenance["original_sha256"] == expected
            assert sha(path) == provenance["current_sha256"]
            assert provenance["model_forwards_added"] == 0 and provenance["metrics_or_audit_logic_changed"] is False
            streams = []
            for item in [archive, path]:
                with item.open("rb") as stream:
                    streams.append([(token.type, token.string) for token in tokenize.tokenize(stream.readline)
                        if token.type not in [tokenize.STRING, tokenize.ENCODING, tokenize.ENDMARKER]])
            assert streams[0] == streams[1], "Historical/current analysis differs beyond explanatory string constants"
            result.append(dict(path=str(archive), sha256=expected, scope="Runtime-start historical analysis source",
                               current_path=str(path), current_sha256=sha(path), explanatory_strings_only=True))
            continue
        assert path.is_file() and sha(path) == expected, f"Source SHA mismatch: {name}"
        result.append(dict(path=str(path), sha256=expected))
    return result


def validate_runtime(row, policy):
    n, condition = row["target_count"], row["condition_count"]
    if row["config"] == "F":
        assert row["age_trace"] == row["trace"] == []
        assert row["cache_bytes"] == 0
        return
    assert len(row["age_trace"]) == 8 and len(row["trace"]) == 24
    assert [value["step"] for value in row["age_trace"]] == list(range(1, 9))
    previous = [-1] * n
    for step, entry in enumerate(row["age_trace"]):
        selected = policy["per_step_indices"][str(step)]
        newly = sum(previous[index] >= 0 and previous[index] < step - 1 for index in selected) if step else 0
        for index in selected:
            previous[index] = step
        ages = [step - value for value in previous]
        assert entry["selected_target"] == len(selected) and entry["max_read_age"] == max(ages)
        assert entry["age_histogram"] == histogram(ages) and entry["newly_activated"] == newly
        close(entry["mean_read_age"], sum(ages) / n, row["run_id"] + ":mean_age", absolute=1e-6)
        traces = [value for value in row["trace"] if value["step"] == step + 1]
        assert len(traces) == 3 and {value["layer"] for value in traces} == {0, 30, 59}
        for value in traces:
            assert value["selected_target"] == len(selected) and value["condition"] == condition
            assert value["image_qkv_tokens"] == value["image_mlp_tokens"] == len(selected) + condition
            assert value["joint_query_tokens"] == value["text_tokens"] + len(selected) + condition
            assert value["joint_kv_tokens"] == value["text_tokens"] + n + condition
    assert row["cache_bytes"] > 0


def validate_summary(summary, raw, policies, cases, root, previous, plan_path):
    assert summary["status"] == "complete" and summary["phase"] == "primary"
    assert not summary["warnings"] and not summary["failed_records"]
    assert not summary["duplicate_successful_run_ids"] and not summary["duplicate_measurement_keys"]
    counts = summary["counts"]
    for key, expected in dict(raw_records=72, successful_unique_relevant_records=72, expected_successful_records=72,
        measurements=54, expected_measurements=54, warmups=18, expected_warmups=18,
        actual_recorded_dit_calls=576, expected_dit_calls=576).items():
        assert counts[key] == expected, key
    for key in ["budget_plan_audits_all_passed", "runtime_age_audits_all_passed", "planner_runtime_metadata_all_recorded",
                "repeated_output_sha_audit_all_passed", "output_file_sha_audits_all_passed"]:
        assert summary[key] is True, key
    assert summary["execution_order_audit"]["complete_order_matches"] is True
    assert len(summary["runtime_age_audits"]) == len(summary["planner_runtime_rows"]) == len(summary["output_file_sha_audits"]) == 72
    assert len(summary["baseline_F_R2_audits"]) == 6
    assert all(value["png_sha256_equal"] and value["pixel_identical"] for value in summary["baseline_F_R2_audits"])
    assert len(summary["all_request_F_R2_audits"]) == 24 and all(value["output_matches_old"] for value in summary["all_request_F_R2_audits"])
    assert len(summary["quality_rows"]) == len(summary["budget_summaries"]) == 18
    lookup = {(row["case_id"], row["config"]): row for row in summary["quality_rows"]}
    assert len(lookup) == 18
    budget_lookup = {(row["case_id"], row["config"]): row for row in summary["budget_summaries"]}
    for cid in cases:
        f = {row["repeat"]: row for row in raw if row["case_id"] == cid and row["config"] == "F" and row["kind"] == "measurement"}
        for config in CONFIGS:
            row = lookup[cid, config]
            selected = [value for value in raw if value["case_id"] == cid and value["config"] == config and value["kind"] == "measurement"]
            values = [value["e2e_seconds"] for value in selected]
            ratio = math.exp(statistics.mean(math.log(value["e2e_seconds"] / f[value["repeat"]]["e2e_seconds"]) for value in selected))
            assert row["quality_status"] == "complete" and row["timing_complete"] is True
            assert row["timing_n"] == row["expected_repeats"] == row["paired_F_n"] == 3
            assert row["baseline_config"] == "F" and row["timing_pairing_status"] == "complete"
            close(row["e2e_median_seconds"], statistics.median(values), "median")
            close(row["e2e_mean_seconds"], statistics.mean(values), "mean")
            close(row["e2e_cv"], statistics.stdev(values) / statistics.mean(values), "CV")
            close(row["paired_geometric_latency_ratio_to_F"], ratio, "paired ratio")
            close(row["budget_selection_median_seconds"], statistics.median(value["budget_selection_seconds"] for value in selected), "selection median")
            assert row["budget_selection_time_count"] == 3
            zero = next(value for value in selected if value["repeat"] == 0)
            assert row["quality_run_id"] == zero["run_id"] and row["quality_output_sha256"] == zero["output_sha256"]
            assert row["quality_output"] == zero["output"]
            for key in ["lpips_mapmean_to_F", "mae_to_F", "band2_mae_to_F", "band2_gradient_mae_to_F", "extra_background_mae_over_F"]:
                assert math.isfinite(row[key]), key
            gate = row["lpips_mapmean_to_F"] <= .02 and row["extra_background_mae_over_F"] <= 1 / 255
            assert row["predefined_quality_gate_pass"] == gate
            assert row["gate_lpips_le_0p02"] == (row["lpips_mapmean_to_F"] <= .02)
            assert row["gate_extra_background_le_1over255"] == (row["extra_background_mae_over_F"] <= 1 / 255)
            policy, budget = policies[cid, config], budget_lookup[cid, config]
            partial = policy["per_step_audit"][1:7]
            close(row["mean_partial_background_fraction"], statistics.mean(value["actual_background_fraction"] for value in partial), "actual background")
            close(row["mean_partial_active_target_fraction"], statistics.mean(value["target_active_fraction"] for value in partial), "active target")
            close(row["full8_target_fraction"], policy["budgetmeta"]["full8_target_fraction"], "full8 target")
            close(row["mean_unused_partial_background_cap"], statistics.mean(value["unused_background_cap"] for value in partial), "unused cap")
            assert row["max_unused_partial_background_cap"] == max(value["unused_background_cap"] for value in partial)
            assert row["requested_background_fraction"] == policy["budgetmeta"]["requested_b_fraction"]
            assert budget["all_plan_audits_passed"] is True and budget["max_read_age"] == max(value["max_age_at_attention_read"] for value in policy["per_step_audit"])
    for config in CONFIGS:
        rows = [lookup[cid, config] for cid in cases]
        aggregate = summary["configs"][config]
        ratio = math.exp(statistics.mean(math.log(row["paired_geometric_latency_ratio_to_F"]) for row in rows))
        close(aggregate["paired_geometric_latency_ratio_to_F_across_cases"], ratio, "aggregate ratio")
        close(aggregate["descriptive_net_latency_reduction_fraction"], 1 - ratio, "aggregate reduction")
        assert aggregate["quality_gate_pass_all_cases"] == all(row["predefined_quality_gate_pass"] for row in rows)
        assert aggregate["formal_Go_established"] is False
    metrics_env = read_json(root / "data/metrics_environment_primary.json")
    assert metrics_env["network"] == "alex" and metrics_env["spatial"] is True and metrics_env["normalize"] is True
    assert metrics_env["regions_sha256"] == sha(previous / "masks/regions.json")
    assert metrics_env["budget_plan_sha256"] == sha(plan_path)
    assert metrics_env["raw_runs_sha256"] == sha(root / "data/runs_primary.jsonl")
    assert metrics_env["script_sha256"] == sha(root / "scripts/analyze_budget.py")


def validate(root, previous, rotation):
    data = root / "data"
    prereg_path, plan_path = data / "preregistration.json", root / "masks/budget_plan.json"
    prereg, plan = read_json(prereg_path), read_json(plan_path)
    protocol = root / "01_规则预算实验协议.md"
    assert sha(protocol) == prereg["protocol_sha256"]
    assert sha(plan_path) == prereg["policy_plan_sha256"]
    assert prereg["configs"] == CONFIGS
    assert [prereg[key] for key in ["requests", "dit_forwards", "warmups", "measurements", "repeats"]] == [72, 576, 18, 54, 3]
    assert prereg["oracle_forwards"] == prereg["training_images_used"] == 0
    assert prereg["value_correction"] is False and prereg["per_request_selection_for_budget_candidates"] is True
    assert prereg["static_controls"] == ["F", "R2"]
    prereg_sources = validate_sources(prereg["scripts"], root)
    cpu_path = data / "budget_policy_checks.json"
    cpu = read_json(cpu_path)
    assert sha(cpu_path) == prereg["cpu_checks_sha256"]
    assert cpu["status"] == "PASS" and cpu["policy_checks"] == len(cpu["checks"]) == 48
    assert len(set(cpu["checks"])) == 48 and sum(value.startswith("real:") for value in cpu["checks"]) == 18
    assert cpu["model_calls"] == cpu["dit_forwards"] == 0 and cpu["stdlib_only"] is True
    assert cpu["plan_sha256"] == sha(plan_path)
    assert plan["seed"] == PLANNER_SEED and plan["configs"] == CONFIGS
    assert plan["full_steps_1based_local"] == [1, 8] and plan["partial_steps_1based"] == list(range(2, 8))
    source_plan_path = rotation / "masks/rotation_plan.json"
    assert sha(source_plan_path) == plan["source_plan_sha256"]
    source_plan = read_json(source_plan_path)
    for name, expected in plan["generation_source_sha256"].items():
        assert sha(root / "scripts" / name) == expected
    cases = [row["case_id"] for row in read_json(previous / "cases.json")["cases"]]
    regions = read_json(previous / "masks/regions.json")
    assert len(cases) == 3 and set(cases) == set(plan["cases"])
    policies = {}
    for cid in cases:
        case, source = plan["cases"][cid], source_plan["cases"][cid]
        h, w = source["target_grid_hw"]
        core = set(regions["cases"][cid]["regions"]["R2"]["indices"])
        assert sorted(core) == source["core_indices"] == case["core_indices"]
        assert case["target_grid_hw"] == [h, w] and case["n_target_tokens"] == h * w
        assert case["n_core_tokens"] == len(core) and case["n_background_tokens"] == h * w - len(core)
        assert case["seed"] == PLANNER_SEED and case["case_id"] == cid
        blocks = independent_blocks(cid, h, w, core)
        assert blocks == case["blocks"], f"Independent 2x2/hash block reconstruction failed: {cid}"
        assert set(case["configs"]) == set(CONFIGS)
        for config in CONFIGS:
            expected = independent_policy(h * w, core, blocks, config)
            assert expected == case["configs"][config], f"Independent oldest-first greedy policy mismatch: {cid}/{config}"
            policies[cid, config] = expected
    environment = read_json(data / "environment_primary.json")
    assert environment["configs"] == CONFIGS and environment["planned_requests"] == 72 and environment["repeats"] == 3
    assert environment["region_plan_sha256"] == sha(plan_path)
    environment_sources = validate_sources(environment["scripts"], root, allow_analysis_archive=True)
    for name, expected in prereg["scripts"].items():
        assert environment["scripts"][name] == expected
    assert environment["steps"] == 8 and environment["cfg"] == 1 and environment["dtype"] == "bfloat16"
    assert environment["seed"] == 2026102201 and environment["vae_tiling"] is True and environment["backend"] == "native SDPA"
    assert environment["extra_dense_oracles_in_performance"] == environment["training_images_used"] == 0
    assert environment["value_correction"] is False and environment["adaptive_quality_risk"] is False
    assert environment["full_fallback_other_than_endpoints"] is False
    assert environment["selection_precomputation_reused_for_candidates"] is False
    assert environment["static_policies_used_only_for_F_R2"] is True and environment["block_partition_reused"] is True
    assert environment["background_budgets"] == {config: [fraction.numerator, fraction.denominator] for config, fraction in BUDGETS.items() if fraction is not None}
    raw_times = environment["expected_raw_timesteps"]
    assert len(raw_times) == 8 and all(math.isfinite(value) for value in raw_times)
    expected_times = [bf16_round(bf16_round(value) / 1000) for value in raw_times]
    assert environment["expected_model_timesteps"] == expected_times
    raw_path = data / "runs_primary.jsonl"
    raw = read_rows(raw_path)
    assert len(raw) == len({row["run_id"] for row in raw}) == 72, "Full 72-request ledger is required"
    assert all(row["status"] == "complete" for row in raw)
    assert sum(row["dit_calls"] for row in raw) == 576
    assert Counter(row["kind"] for row in raw) == {"warmup": 18, "measurement": 54}
    expected_order = [(cid, config, "warmup", 0) for cid in cases for config in CONFIGS]
    for repeat in range(3):
        for index, cid in enumerate(cases):
            shift = (repeat + index) % len(CONFIGS)
            expected_order += [(cid, config, "measurement", repeat) for config in CONFIGS[shift:] + CONFIGS[:shift]]
    assert [(row["case_id"], row["config"], row["kind"], row["repeat"]) for row in raw] == expected_order
    old_rows = read_rows(previous / "data/runs.jsonl")
    old_lookup = {(row["case_id"], row["config"]): row for row in old_rows if row.get("status") == "complete"
        and row.get("kind") == "measurement" and row.get("repeat") == 0 and row["config"] in ["F", "R2"]}
    controls, rebuilt, native = 0, 0, 0
    for row in raw:
        cid, config = row["case_id"], row["config"]
        policy, case = policies[cid, config], plan["cases"][cid]
        assert row["run_id"] == f"{config}__{cid}__{row['kind']}_{row['repeat']}"
        assert row["phase"] == "primary" and row["dit_calls"] == 8 and row["seed"] == environment["seed"]
        assert row["target_count"] == case["n_target_tokens"] and row["target_hw"] == case["target_hw"]
        assert row["condition_count"] == old_lookup[cid, "F"]["condition_count"]
        assert math.isfinite(row["e2e_seconds"]) and row["e2e_seconds"] > 0
        assert sha(root / row["output"]) == row["output_sha256"]
        assert row["full_steps_1based"] == policy["full_steps_1based"]
        assert row["policy_matches_frozen"] is True and row["budget_selection_metadata"] == policy["budgetmeta"]
        assert row["per_step_selected_target_counts"] == [len(policy["per_step_indices"][str(step)]) for step in range(8)]
        selection = row["budget_selection_seconds"]
        assert math.isfinite(selection) and 0 <= selection < row["e2e_seconds"]
        if config in ["F", "R2"]:
            assert row["budget_selection_rebuilt"] is False and selection == 0
            old = old_lookup[cid, config]
            assert sha(previous / old["output"]) == old["output_sha256"] == row["output_sha256"]
            assert row["old_output_exact"] is True
            assert row["initial_latent_sha256"] == old["initial_latent_sha256"] and row["sigmas"] == old["sigmas"]
            controls += 1
        else:
            assert row["budget_selection_rebuilt"] is True and selection > 0
            assert row["old_output_exact"] is None
            rebuilt += 1
        assert row["model_timestep_contract_exact"] is True and row["observed_model_timesteps"] == expected_times
        assert row["timesteps"] == raw_times and len(row["sigmas"]) == 9
        assert row["sigmas"] == old_lookup[cid, "F"]["sigmas"]
        assert row["initial_latent_sha256"] == old_lookup[cid, "F"]["initial_latent_sha256"]
        validate_runtime(row, policy)
        native += config == "F"
    assert controls == 24 and rebuilt == 48 and native == 12
    for cid in cases:
        for config in CONFIGS:
            group = [row for row in raw if row["case_id"] == cid and row["config"] == config]
            assert len(group) == 4 and len({row["output_sha256"] for row in group}) == 1
            assert len({row["initial_latent_sha256"] for row in group}) == 1
            assert sum(row["kind"] == "warmup" for row in group) == 1
            assert {row["repeat"] for row in group if row["kind"] == "measurement"} == {0, 1, 2}
    summary = read_json(data / "summary_primary.json")
    validate_summary(summary, raw, policies, cases, root, previous, plan_path)
    return dict(status="pass", scope="Integrity, numerical contracts and execution accounting; not semantic or algorithm-quality acceptance",
        validated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), total_requests=72, measured_requests=54, warmup_requests=18,
        total_real_dit_forwards=576, extra_dense_oracles_in_performance=0, training_images_used=0, value_correction=False,
        policy_cpu_checks=48, policy_cpu_model_calls=0, policy_cpu_dit_forwards=0,
        independent_block_reconstructions=3, independent_whole_tile_greedy_schedules=18,
        producer_policy_equals_frozen_checks=72, candidate_in_request_policy_rebuilds=48, static_F_R2_policy_requests=24,
        F_R2_old_PNG_exact_controls=controls, model_timestep_contract_checks=72,
        independent_request_age_shape_checks=72, native_F_empty_cache_checks=12, local_requests_with_8_step_checks=60,
        whole_execution_order_matches=True, all_output_hashes_verified=True, all_warmup_measurement_group_hashes_identical=True,
        failed_real_model_requests=0, tiny_model_forwards=0,
        peak_allocated_GiB=max(row["peak_allocated_bytes"] for row in raw) / 2**30,
        request_seconds_sum=sum(row["e2e_seconds"] for row in raw),
        candidate_selection_seconds_sum=sum(row["budget_selection_seconds"] for row in raw if row["config"] not in ["F", "R2"]),
        model_load_and_fuse_seconds=environment["model_load_and_fuse_seconds"],
        candidate_numeric_quality_pass_all_cases={config: summary["configs"][config]["quality_gate_pass_all_cases"] for config in CONFIGS[2:]},
        deployment_joint_target_inferred=False, budget_is_not_latency_or_quality_guarantee=True,
        preregistration_sources=prereg_sources, environment_sources=environment_sources,
        runs_sha256=sha(raw_path), plan_sha256=sha(plan_path), preregistration_sha256=sha(prereg_path),
        cpu_checks_sha256=sha(cpu_path), summary_sha256=sha(data / "summary_primary.json"),
        validator_sha256=sha(Path(__file__)),
        provenance_scope="Producer asserts full rebuilt-policy equality after timed requests; metadata/count/age/shape checks and independently reconstructed frozen policies are verified here; full per-request rebuilt index lists were not separately exported")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--previous-root", type=Path)
    parser.add_argument("--rotation-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "data/final_validation_summary.json"
    save(output, dict(status="running", started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        validator_sha256=sha(Path(__file__)), stale_PASS_prevented=True))
    try:
        previous = locate(root, args.previous_root, ["局部执行与边界验证_20261022", "local_executor_boundary_20261022"])
        rotation = locate(root, args.rotation_root, ["背景轮转与上下文优化_20261022", "rotation_context_optimization_20261022"])
        result = validate(root, previous, rotation)
    except Exception as error:
        save(output, dict(status="fail", validated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(), validator_sha256=sha(Path(__file__)),
            scope="No PASS is emitted without the complete raw ledger and every required artifact"))
        raise
    save(output, result)
    manifest = {str(path.relative_to(root)): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json" and "__pycache__" not in path.parts and path.name != ".DS_Store"}
    save(root / "data/artifact_manifest.json", manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
