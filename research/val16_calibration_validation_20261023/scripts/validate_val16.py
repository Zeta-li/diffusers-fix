import argparse
import collections
import hashlib
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path


CONFIGS = ["F", "R2", "PRIOR", "A", "B"]
CAL_SHA = "6df60e49ca8d618c5458063a80ce5bf3a5b02b0c28c7e618b04b76733964518f"
LORA_SHA = "36ee4ff860856559b9c581eaa69414900488f3de8be2f2448586daec77e201b9"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def located(value, roots):
    path = Path(value)
    if path.is_absolute():
        return path
    return next((root / path for root in roots if (root / path).exists()), roots[0] / path)


def bf16(value):
    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def group(case):
    value = case.get("original_img_id", case.get("img_id", case.get("case_id")))
    if value is None:
        raise ValueError("Missing source group identity")
    return str(value).lstrip("0") or "0"


def png_hw(path):
    with path.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or len(header) != 24 or header[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return [height, width]


def validate(args):
    root = args.root.resolve()
    train = args.train_root.resolve()
    errors, warnings = [], []
    def check(condition, message):
        if not condition:
            errors.append(message)
        return bool(condition)
    cases_path = located(args.cases, [root])
    prereg_path = located(args.prereg, [root])
    runs_path = located(args.runs, [root])
    calibration_path = located(args.calibration, [train, root])
    selection_path = located(args.selection, [train, root])
    manifest = read(cases_path)
    cases = manifest["cases"]
    prereg = read(prereg_path)
    train_cases = read(train / "cases_train8.json")["cases"]
    selection = read(selection_path)
    raw = []
    for index, line in enumerate(runs_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw.append(json.loads(line))
        except json.JSONDecodeError as error:
            errors.append(f"Malformed run ledger line {index}: {error}")
    complete = [row for row in raw if row.get("status") == "complete"]
    failed = [row for row in raw if row.get("status") == "failed"]
    check(len(raw) == len(complete) + len(failed), "Unknown run status in ledger")
    case_ids = [case["case_id"] for case in cases]
    case_map = {case["case_id"]: case for case in cases}
    check(len(case_ids) == len(set(case_ids)) == 16, "Exactly 16 unique validation sources are required")
    check(all(case.get("status") == "PASS" for case in cases), "A validation source did not pass frozen QC")
    check(prereg.get("case_ids") == case_ids, "Preregistered case order differs from manifest")
    check(prereg.get("configs") == CONFIGS, "Configuration order differs from F/R2/PRIOR/A/B")
    check(prereg.get("planned_requests") == 320 and prereg.get("planned_dit_forwards") == 2560,
          "Preregistered request/DiT budget differs from 320/2560")
    check(prereg.get("repeats") == 3 and prereg.get("quality_repeat") == 0, "Repeat contract changed")
    check(prereg.get("warmups") == 80 and prereg.get("measurements") == 240, "Warmup/measurement budget changed")
    check(prereg.get("cases_sha256") == sha(cases_path), "Validation manifest changed after preregistration")
    check(sha(calibration_path) == CAL_SHA == prereg.get("calibration_sha256"), "Frozen Train8 calibration digest mismatch")
    check(prereg.get("fitting_on_val") is False and prereg.get("target_column_used") is False
          and prereg.get("test_used") is False and prereg.get("rl") is False
          and prereg.get("new_gradient_steps") == 0, "No-fit/no-target/no-Test/no-RL contract changed")
    check(LORA_SHA in prereg.get("model_identity", ""), "Model identity does not contain actual frozen LoRA digest")
    gate = prereg.get("quality_gate", {})
    check(gate.get("lpips_to_F_max") == 0.02 and gate.get("extra_far_background_over_F_max") == 1 / 255,
          "Executed numeric quality gate was changed")
    roots = [root, train]
    source_audits = []
    for filename, digest in prereg.get("scripts", {}).items():
        path = located(filename, roots)
        check(path.exists() and sha(path) == digest, f"Preregistered source code changed or missing: {filename}")
    eval_digest = prereg.get("evaluation_regions_sha256")
    if eval_digest:
        eval_path = located(prereg.get("evaluation_regions_path", prereg.get("evaluation_regions_file", "masks/evaluation_regions.json")), roots)
        check(eval_path.exists() and sha(eval_path) == eval_digest, "Frozen evaluation-region file changed or missing")
    else:
        errors.append("Missing preregistered evaluation-region digest")
    expected_val_groups = {group(case) for case in selection["selections"]["val16"]}
    train_groups = {group(case) for case in selection["selections"]["train32"]}
    test_groups = {group(case) for case in selection["selections"]["test64"]}
    val_groups = {group(case) for case in cases}
    check(val_groups == expected_val_groups and len(val_groups) == 16, "Validation source groups differ from frozen Val16")
    check(not (val_groups & train_groups) and not (val_groups & test_groups), "Source-group leakage across Train/Val/Test")
    train_ids = {case["case_id"] for case in train_cases}
    train_raw_hashes = {case.get("source_sha256") for case in train_cases}
    train_rgb_hashes = {case.get("signature", {}).get("rgb_sha256") for case in train_cases}
    train_rgb_hashes.discard(None)
    seen_raw, seen_rgb = set(), set()
    for case in cases:
        cid = case["case_id"]
        source_path = located(case["image"], roots)
        source_ok = source_path.exists() and sha(source_path) == case.get("source_sha256")
        check(source_ok, f"Source content digest mismatch: {cid}")
        check(case.get("official_split") == "dev", f"Source is not from the frozen official dev split: {cid}")
        check(case.get("target_column_read") is False and "target_img" not in case.get("columns_read", []),
              f"Ground-truth target read contract failed: {cid}")
        indices = case["indices"]
        n = int(case.get("n_target_tokens", (case["target_hw"][0] // 16) * (case["target_hw"][1] // 16)))
        check(bool(indices) and len(indices) == len(set(indices)) and all(type(v) is int and 0 <= v < n for v in indices),
              f"Invalid frozen core indices: {cid}")
        core_sha = hashlib.sha256(json.dumps(indices).encode()).hexdigest()
        registered = prereg.get("sources", {}).get(cid, {})
        check(registered.get("source_sha256") == case.get("source_sha256")
              and registered.get("mask_sha256") == case.get("mask_sha256")
              and registered.get("core_indices_sha256") == core_sha, f"Registered source/mask/core differs: {cid}")
        for field, digest_field in [("original_mask", "mask_sha256"), ("binary_mask", "binary_mask_sha256")]:
            if field in case:
                path = located(case[field], roots)
                check(path.exists() and sha(path) == case[digest_field], f"Mask asset digest mismatch: {cid}:{field}")
        raw_hash = case.get("source_sha256")
        rgb_hash = case.get("signature", {}).get("rgb_sha256")
        check(raw_hash not in train_raw_hashes and raw_hash not in seen_raw, f"Exact source-content duplicate: {cid}")
        if rgb_hash:
            check(rgb_hash not in train_rgb_hashes and rgb_hash not in seen_rgb, f"Decoded-RGB duplicate: {cid}")
            seen_rgb.add(rgb_hash)
        seen_raw.add(raw_hash)
        comparisons = case.get("duplicate_comparisons", [])
        check(all(item.get("verdict") == "PASS" for item in comparisons), f"Unresolved near-duplicate QC: {cid}")
        if not comparisons:
            warnings.append(f"No per-source near-duplicate receipt in case manifest: {cid}")
        source_audits.append(dict(case_id=cid, source_sha256=raw_hash, source_file_verified=source_ok,
                                 source_group=group(case), core_indices_sha256=core_sha))
    expected_keys = {(cid, config, "warmup", 0) for cid in case_ids for config in CONFIGS}
    expected_keys |= {(cid, config, "measurement", repeat) for cid in case_ids for config in CONFIGS for repeat in range(3)}
    keys = [(r.get("case_id"), r.get("config"), r.get("kind"), r.get("repeat")) for r in complete]
    check(len(complete) == 320 and set(keys) == expected_keys and len(set(keys)) == len(keys),
          "Missing, extra or duplicate successful execution keys")
    run_ids = [row.get("run_id") for row in complete]
    check(len(run_ids) == len(set(run_ids)), "Duplicate successful run IDs")
    planned_order = prereg.get("planned_execution_order", [])
    check(len(planned_order) == 320 and run_ids == planned_order, "Successful execution order differs from preregistration")
    cursor = 0
    for row in raw:
        check(cursor < len(planned_order) and row.get("run_id") == planned_order[cursor], "Failure/resume sequence diverges from next planned request")
        if row.get("status") == "complete":
            cursor += 1
    environments = {}
    for path in (root / "data").glob("environment_*.json"):
        environment = read(path)
        environments[environment.get("attempt")] = (path, environment)
    expected_raw_times = None
    expected_sigmas = None
    table_audits = []
    for attempt in {row.get("attempt") for row in raw}:
        if not check(attempt in environments, f"Missing environment for attempt: {attempt}"):
            continue
        path, environment = environments[attempt]
        check(environment.get("preregistration_sha256") == sha(prereg_path), f"Environment/preregistration mismatch: {attempt}")
        check(environment.get("calibration_sha256") == CAL_SHA and environment.get("model_identity") == prereg.get("model_identity"),
              f"Environment calibration/model identity mismatch: {attempt}")
        check(environment.get("steps") == 8 and environment.get("cfg") == 1
              and environment.get("model_dtype") == "bfloat16", f"Model precision/step/CFG contract changed: {attempt}")
        check(environment.get("new_teacher_or_shadow_forwards") == 0 and environment.get("val_refit") is False,
              f"Extra teacher/refit during validation: {attempt}")
        check(environment.get("all_correction_tables_resident_for_all_configs") is True,
              f"Static table residency contract changed: {attempt}")
        times = environment.get("expected_raw_timesteps", [])
        check(len(times) == 8 and all(math.isfinite(v) for v in times), f"Invalid raw timestep contract: {attempt}")
        if expected_raw_times is None:
            expected_raw_times = times
        else:
            check(times == expected_raw_times, f"Timestep schedule changed across attempts: {attempt}")
        model_times = [bf16(bf16(value) / 1000) for value in times]
        check(environment.get("expected_model_timesteps") == model_times, f"BF16 cast-before-division timestep contract differs: {attempt}")
        tables = environment.get("tables", {})
        check(set(tables) == {"PRIOR", "A", "B"}, f"Missing separate frozen correction tables: {attempt}")
        prior, a, b = [tables.get(name, {}) for name in ["PRIOR", "A", "B"]]
        check(prior.get("statistics_mode") == "no_sample_mean_zero_variance_one", f"PRIOR is not the mu0/S1 control: {attempt}")
        check(a.get("caller_statistics_source_hash") == CAL_SHA, f"A statistics source mismatch: {attempt}")
        check(b.get("source_hash") == CAL_SHA and set(b.get("training_image_ids", [])) == train_ids,
              f"B coefficients did not come exclusively from frozen Train8: {attempt}")
        contract = b.get("contract", {})
        check(contract.get("model_identity") == prereg.get("model_identity")
              and contract.get("raw_scheduler_timesteps") == times and contract.get("model_timesteps") == model_times,
              f"B fitted contract mismatch: {attempt}")
        check(b.get("unseen_pair_fallback") == "identity" and b.get("diagonal_alpha") == 1
              and b.get("diagonal_delta") == 0, f"B unseen/same-time identity contract changed: {attempt}")
        shapes = [table.get("alpha_shape") for table in [prior, a, b]]
        check(all(shape == shapes[0] for shape in shapes) and bool(shapes[0]), f"Correction table geometry differs: {attempt}")
        table_audits.append(dict(attempt=attempt, environment_sha256=sha(path),
                                A_statistics_sha256=a.get("statistics_float32_content_sha256"),
                                B_effective_coefficient_sha256=b.get("effective_coefficient_sha256")))
    for field in ["A_statistics_sha256", "B_effective_coefficient_sha256"]:
        check(bool(table_audits) and len({audit[field] for audit in table_audits}) == 1
              and all(audit[field] for audit in table_audits), f"Frozen table content differs across attempts: {field}")
    output_sha_cache = {}
    by_case, by_case_config = collections.defaultdict(list), collections.defaultdict(list)
    for row in complete:
        cid, config = row.get("case_id"), row.get("config")
        if cid not in case_map or config not in CONFIGS:
            continue
        case = case_map[cid]
        label = row.get("run_id")
        check(row.get("dit_calls") == 8 and row.get("oracle_calls") == 0, f"Non-eight-step or teacher call count: {label}")
        check(row.get("request_reset") is True and row.get("cleanup_all_executors_empty") is True, f"Cache lifecycle audit failed: {label}")
        check(math.isfinite(row.get("e2e_seconds", float("nan"))) and row["e2e_seconds"] > 0, f"Invalid latency: {label}")
        check(row.get("seed") == prereg.get("seed"), f"Seed changed: {label}")
        check(row.get("source_sha256") == case.get("source_sha256")
              and row.get("core_indices_sha256") == prereg["sources"][cid]["core_indices_sha256"], f"Source/core changed: {label}")
        check(row.get("target_hw") == case["target_hw"], f"Target geometry changed: {label}")
        n = (case["target_hw"][0] // 16) * (case["target_hw"][1] // 16)
        check(row.get("target_count") == n, f"Target token count differs: {label}")
        expected_class = {"F": "native", "R2": "LocalTargetExecutor", "PRIOR": "ValueChannelExecutor", "A": "ValueChannelExecutor", "B": "ValueFittedExecutor"}[config]
        check(row.get("executor_class") == expected_class and row.get("correction_mode") == ("none" if config in ["F", "R2"] else "affine"),
              f"Executor or correction mode mismatch: {label}")
        check(row.get("coefficient_sha256") == (CAL_SHA if config in ["A", "B"] else None), f"Wrong per-request coefficient source: {label}")
        check(row.get("timesteps") == expected_raw_times and row.get("model_timestep_contract_exact") is True,
              f"Raw timestep contract failed: {label}")
        check(row.get("observed_model_timesteps") == [bf16(bf16(value) / 1000) for value in expected_raw_times or []],
              f"Observed model timestep differs: {label}")
        sigmas = row.get("sigmas", [])
        check(len(sigmas) == 9 and all(math.isfinite(v) for v in sigmas) and sigmas[-1] == 0,
              f"Invalid Euler sigma sequence: {label}")
        if expected_sigmas is None:
            expected_sigmas = sigmas
        else:
            check(sigmas == expected_sigmas, f"Euler schedule differs across requests: {label}")
        path = located(row.get("output", "missing-output"), [root])
        if check(path.exists(), f"Missing output: {label}"):
            if path not in output_sha_cache:
                output_sha_cache[path] = sha(path)
            check(output_sha_cache[path] == row.get("output_sha256"), f"Output file digest differs: {label}")
            check(png_hw(path) == case["target_hw"], f"Output pixel geometry differs: {label}")
        trace = row.get("trace", [])
        age_trace = row.get("age_trace", [])
        if config == "F":
            check(trace == [] and age_trace == [] and row.get("full_steps_1based") == list(range(1, 9)), f"Native F executor state was populated: {label}")
        else:
            check({item.get("step") for item in trace} == set(range(1, 9)), f"Missing per-step operator trace: {label}")
            check(row.get("full_steps_1based") == [1, 8] and len(age_trace) == 8, f"Full-step/age trace contract differs: {label}")
            for item in trace:
                selected = n if item.get("step") in [1, 8] else len(case["indices"])
                check(item.get("selected_target") == selected and item.get("condition") == row.get("condition_count"), f"Selection/condition changed: {label}")
                check(item.get("image_qkv_tokens") == selected + row.get("condition_count", 0)
                      and item.get("image_mlp_tokens") == selected + row.get("condition_count", 0), f"QKV/MLP computation differs from frozen R2: {label}")
            for item in age_trace:
                step = item.get("step")
                selected = n if step in [1, 8] else len(case["indices"])
                expected_age = 0 if step in [1, 8] or selected == n else step - 1
                check(item.get("selected_target") == selected and item.get("max_read_age") == expected_age,
                      f"Actual raw-cache age differs from frozen R2: {label}")
        by_case[cid].append(row)
        by_case_config[(cid, config)].append(row)
    repeated_audits = []
    for (cid, config), rows in by_case_config.items():
        identical = len(rows) == 4 and len({row.get("output_sha256") for row in rows}) == 1
        check(identical, f"Warmup and three repeats are not pixel-file identical: {cid}:{config}")
        repeated_audits.append(dict(case_id=cid, config=config, requests=len(rows), identical_png_sha256=identical))
    for cid, rows in by_case.items():
        for field in ["initial_latent_sha256", "condition_count"]:
            check(all(field in row for row in rows) and len({str(row.get(field)) for row in rows}) == 1,
                  f"Cross-configuration {field} mismatch: {cid}")
    failed_completed = sum(row.get("dit_calls_completed", row.get("dispatch_completed", 0)) for row in failed)
    failed_invoked = sum(row.get("dit_calls_invoked", row.get("dispatch_invoked", row.get("dit_calls_completed", 0))) for row in failed)
    successful_dit = sum(row.get("dit_calls", 0) for row in complete)
    oracle_calls = sum(row.get("oracle_calls", 0) for row in raw)
    check(successful_dit == 2560, "Successful request DiT count differs from 2560")
    check(oracle_calls == 0, "Unexpected teacher/shadow call recorded in ledger")
    check(0 <= failed_completed <= failed_invoked, "Failed invocation/completion accounting is inconsistent")
    status_path = root / "status.json"
    if check(status_path.exists(), "Missing final runner status"):
        final_status = read(status_path)
        check(final_status.get("phase") == "complete", "Runner has not recorded final completion")
        if final_status.get("phase") == "complete":
            check(final_status.get("successful_requests") == len(complete)
                  and final_status.get("successful_dit_forwards") == successful_dit
                  and final_status.get("failed_requests") == len(failed)
                  and final_status.get("failed_dit_completed") == failed_completed
                  and final_status.get("failed_dit_invoked") == failed_invoked
                  and final_status.get("oracle_calls") == oracle_calls
                  and final_status.get("val_sources") == 16
                  and final_status.get("calibration_sha256") == CAL_SHA,
                  "Final status differs from independent successful/failed accounting")
    if failed:
        warnings.append("Failed attempts are preserved; completed DiT and incomplete invocations are reported separately, not hidden in the successful budget")
    check(sha(calibration_path) == CAL_SHA, "Calibration file changed during independent validation")
    return dict(status="PASS" if not errors else "FAIL", validated_at=datetime.now(timezone.utc).isoformat(),
        scope="Independent data/accounting/provenance/determinism audit only; no semantic, speed significance, noninferiority or formal Go claim",
        planned_sources=16, successful_requests=len(complete), warmups=sum(r.get("kind") == "warmup" for r in complete),
        measurements=sum(r.get("kind") == "measurement" for r in complete), successful_dit_forwards=successful_dit,
        failed_attempts=len(failed), failed_completed_dit_forwards=failed_completed,
        failed_invoked_dit_forwards=failed_invoked, completed_dit_including_failed_attempts=successful_dit + failed_completed,
        incomplete_failed_invocations=max(0, failed_invoked - failed_completed), extra_teacher_or_shadow_calls=oracle_calls,
        calibration_sha256=CAL_SHA, cases_sha256=sha(cases_path), preregistration_sha256=sha(prereg_path),
        source_audits=source_audits, table_audits=table_audits, repeated_output_audits=repeated_audits,
        source_group_disjointness="Frozen Train32/Val16/Test64 metadata checked; no Test media opened by this validator",
        decoded_content_scope="Manifest RGB hashes and declared near-duplicate receipts; independent raw source file digests checked",
        formal_Go_established=False, errors=errors, warnings=warnings,
        validator_sha256=sha(Path(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--cases", default="cases_val16.json")
    parser.add_argument("--prereg", default="data/preregistration.json")
    parser.add_argument("--runs", default="data/runs_primary.jsonl")
    parser.add_argument("--calibration", default="data/calibration_train8.npz")
    parser.add_argument("--selection", default="preprocess_alpha_v2/data/frozen_selection.json")
    parser.add_argument("--output", default="data/independent_validation.json")
    args = parser.parse_args()
    if args.train_root is None:
        candidates = [args.root.parent / "fewshot_calibration_start_20261023", args.root.parent / "小样本校准启动_20261023"]
        args.train_root = next((path for path in candidates if path.exists()), candidates[0])
    try:
        result = validate(args)
    except Exception as error:
        result = dict(status="FAIL", scope="Validator could not complete", errors=[f"{type(error).__name__}: {error}"], formal_Go_established=False)
    destination = located(args.output, [args.root])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ["source_audits", "table_audits", "repeated_output_audits"]}, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
