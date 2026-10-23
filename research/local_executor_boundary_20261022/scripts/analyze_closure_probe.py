import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from PIL import Image

from analyze_local_pilot import (
    EXTRA_BACKGROUND_GATE,
    LPIPS_GATE,
    comparison_panel,
    config_id,
    evaluation_masks,
    load_image,
    metric_environment,
    quality_metrics,
    save_csv,
    sha256,
    timing_detail,
    timing_summary,
    write_json,
)


CASE_ID = "remove_hair"
CONFIGS = ["F_closure", "R4", "R5"]
EXPECTED_REPEATS = 3
EXPECTED_REQUESTS = 12


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def successful_unique(records):
    result = {}
    for row in records:
        if row.get("status") != "complete":
            continue
        assert row["run_id"] not in result, f"Duplicate successful run_id: {row['run_id']}"
        assert row["case_id"] == CASE_ID, "Closure records must contain only remove_hair"
        assert config_id(row) in CONFIGS, "Unexpected closure config"
        result[row["run_id"]] = row
    return result


def verify_regions(root, primary, probe):
    assert probe["primary_regions_json_sha256"] == sha256(root / "masks/regions.json")
    primary_case = primary["cases"][CASE_ID]
    case = probe["cases"][CASE_ID]
    assert case["target_hw"] == primary_case["target_hw"]
    assert case["target_grid_hw"] == primary_case["target_grid_hw"]
    assert case["n_target_tokens"] == primary_case["n_target_tokens"]
    core_path = root / primary_case["regions"]["R0"]["mask_npy"]
    assert sha256(core_path) == case["R0_mask_sha256"]
    core = np.load(core_path, allow_pickle=False).astype(bool)
    masks = {}
    for config in ["R4", "R5"]:
        region = case["regions"][config]
        path = root / region["mask_npy"]
        assert sha256(path) == region["mask_sha256"]
        mask = np.load(path, allow_pickle=False)
        assert mask.dtype == np.bool_ and mask.shape == core.shape
        assert np.all(mask[core]), f"{config} must include the original R0 core"
        assert int(mask.sum()) == region["n_active_target_tokens"]
        assert np.flatnonzero(mask).tolist() == region["indices"]
        masks[config] = mask
    assert int(masks["R4"].sum()) == int(masks["R5"].sum()), "R4/R5 budgets must match exactly"
    assert not np.array_equal(masks["R4"], masks["R5"]), "Equal budgets must retain distinct placement"
    source_path = root / case["source_image"]
    assert sha256(source_path) == case["source_sha256"]
    return primary_case, case


def relative_improvement(reference, candidate):
    if reference is None or candidate is None or abs(reference) < 1e-12:
        return None
    return (reference - candidate) / reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    data, figures = root / "data", root / "figures"
    data.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    primary_path = root / "masks/regions.json"
    probe_path = root / "masks/closure_probe.json"
    primary = json.loads(primary_path.read_text())
    probe = json.loads(probe_path.read_text())
    primary_case, case = verify_regions(root, primary, probe)
    plan = probe["intended_experiment"]
    assert plan["configs"] == CONFIGS and plan["case_ids"] == [CASE_ID]
    assert int(plan["repeats"]) == EXPECTED_REPEATS
    assert int(plan["warmups_per_config"]) == 1
    assert int(plan["planned_requests"]) == EXPECTED_REQUESTS
    assert plan["full_steps_1based_local"] == [1, 8]
    records_path = data / "closure_runs.jsonl"
    records = read_rows(records_path)
    successful = successful_unique(records)
    failures = [row for row in records if row.get("status") == "failed"]
    summary = {
        "status": "validating",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "analysis_group": "posthoc_single_case_carrier_closure",
        "case_id": CASE_ID,
        "configs": CONFIGS,
        "quality_reference_config": "F_closure",
        "timing_reference_config": "F_closure",
        "posthoc": True,
        "preregistered": False,
        "automatic_region_locator": False,
        "primary_or_fallback_results_replaced": False,
        "primary_regions_sha256": sha256(primary_path),
        "closure_probe_sha256": sha256(probe_path),
        "closure_runs_sha256": sha256(records_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "shared_analysis_script_sha256": sha256(Path(__file__).with_name("analyze_local_pilot.py")),
        "closure_manifest_status": probe["status"],
        "annotation_provenance": probe["annotation_provenance"],
        "expected_requests": EXPECTED_REQUESTS,
        "expected_dit_calls": EXPECTED_REQUESTS * 8,
        "successful_unique_requests": len(successful),
        "failed_record_count": len(failures),
        "failed_records": failures,
        "known_successful_dit_calls": sum(int(row.get("dit_calls", 0)) for row in successful.values()),
        "failed_attempt_dit_cost": "Retain reported failed records; missing partial-call counts are unknown, not zero.",
        "successful_request_e2e_seconds_sum": sum(float(row["e2e_seconds"]) for row in successful.values()),
        "cost_scope": "Twelve additional generation requests including three warmups; LPIPS and figure generation are separate analysis costs, not deployment timing. No whole-workflow wall-clock estimate.",
        "evaluation_masks": "Unchanged primary R0 core and its common band1/band2/band4; far background uses the unchanged primary allowed box. No carrier-specific evaluation area is substituted.",
        "quality_protocol": "Repeat0 only. Shared quality_metrics names ending to_F/over_F refer here to same-round F_closure, independently checked against primary F. F is a computational reference, not editing ground truth. Full-image unmasked RGB LPIPS Alex spatial map is averaged in fixed masks; feature receptive fields cross mask boundaries.",
        "predefined_gates": {
            "lpips_mapmean_to_reference_max": LPIPS_GATE,
            "extra_background_mae_over_reference_max": EXTRA_BACKGROUND_GATE,
            "thresholds_changed": False,
            "meaning": "Original provisional numeric retention gates only; no semantic acceptance, losslessness or formal boundary Go follows from this one posthoc case.",
        },
        "confounds": {
            "R4_R5_target_budget_exact": True,
            "perimeter_grid_edges": {key: case["regions"][key]["perimeter_grid_edges"] for key in ["R4", "R5"]},
            "connected_components_4": {key: case["regions"][key]["connected_components_4"] for key in ["R4", "R5"]},
            "placement_perimeter_compactness_topology_not_jointly_controlled": True,
            "posthoc_region_selected_after_observing_primary_artifact": True,
            "generalization_or_causal_boundary_claim_supported": False,
            "manifest_limitations": probe["limitations"],
        },
    }
    summary_path = data / "closure_summary.json"
    write_json(summary_path, summary)
    assert len(successful) == EXPECTED_REQUESTS, "Complete the independent twelve-request probe before analysis"
    measurements = [row for row in successful.values() if row.get("kind") == "measurement"]
    warmups = [row for row in successful.values() if row.get("kind") == "warmup"]
    assert len(measurements) == 9 and len(warmups) == 3
    lookup = {}
    for row in measurements:
        key = (config_id(row), int(row["repeat"]))
        assert key not in lookup, f"Duplicate closure config/repeat: {key}"
        lookup[key] = row
    baseline = {repeat: lookup[("F_closure", repeat)] for repeat in range(EXPECTED_REPEATS)}
    for config in CONFIGS:
        selected = [row for row in measurements if config_id(row) == config]
        assert {int(row["repeat"]) for row in selected} == set(range(EXPECTED_REPEATS))
        assert len([row for row in warmups if config_id(row) == config]) == 1
    height, width = case["target_hw"]
    zero = baseline[0]
    for row in successful.values():
        assert int(row["dit_calls"]) == 8
        assert row["target_hw"] == [height, width]
        assert row["initial_latent_sha256"] == zero["initial_latent_sha256"]
        assert row["timesteps"] == zero["timesteps"] and row["sigmas"] == zero["sigmas"]
        expected_steps = list(range(1, 9)) if config_id(row) == "F_closure" else [1, 8]
        assert row["full_steps_1based"] == expected_steps
        expected_count = case["n_target_tokens"] if config_id(row) == "F_closure" else case["regions"][config_id(row)]["n_active_target_tokens"]
        assert int(row["selected_partial_target_count"]) == expected_count
        load_image(root, row, (width, height))
    primary_records = read_rows(data / "runs.jsonl")
    primary_f = [row for row in primary_records if row.get("status") == "complete" and row.get("case_id") == CASE_ID and config_id(row) == "F" and row.get("kind") == "measurement" and int(row.get("repeat", -1)) == 0]
    assert len(primary_f) == 1, "Require one unambiguous primary F/remove_hair/repeat0 reference"
    primary_f = primary_f[0]
    reference = load_image(root, zero, (width, height))
    primary_image = load_image(root, primary_f, reference.size)
    assert primary_f["initial_latent_sha256"] == zero["initial_latent_sha256"]
    assert primary_f["timesteps"] == zero["timesteps"] and primary_f["sigmas"] == zero["sigmas"]
    difference = np.abs(np.asarray(reference, dtype=np.float32) - np.asarray(primary_image, dtype=np.float32)) / 255
    check = {
        "primary_F_run_id": primary_f["run_id"],
        "F_closure_run_id": zero["run_id"],
        "primary_F_output_sha256": primary_f["output_sha256"],
        "F_closure_output_sha256": zero["output_sha256"],
        "png_sha256_equal": primary_f["output_sha256"] == zero["output_sha256"],
        "pixel_identical": bool(np.array_equal(np.asarray(reference), np.asarray(primary_image))),
        "mae": float(difference.mean()),
        "max_channel_error": float(difference.max()),
    }
    summary["primary_F_reference_check"] = check
    summary["status"] = "reference_checked" if check["png_sha256_equal"] and check["pixel_identical"] else "invalid_primary_reference"
    write_json(summary_path, summary)
    assert check["png_sha256_equal"] and check["pixel_identical"], "F_closure differs from primary F; inspect the recorded discrepancy before interpreting the probe"
    assert torch.cuda.is_available(), "Closure LPIPS analysis requires CUDA after generation has finished"
    import lpips
    network = lpips.LPIPS(net="alex", spatial=True).eval().to(args.device)
    source = Image.open(root / case["source_image"]).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    masks = evaluation_masks(primary_case, root, height, width)
    rows, timing_rows = [], []
    images = {"INPUT": source}
    for config in CONFIGS:
        selected = sorted([row for row in measurements if config_id(row) == config], key=lambda row: int(row["repeat"]))
        measurement_zero = lookup[(config, 0)]
        output = load_image(root, measurement_zero, reference.size)
        images[config] = output
        row = {
            "analysis_group": "posthoc_single_case_carrier_closure",
            "posthoc": True,
            "case_id": CASE_ID,
            "config": config,
            "quality_reference_config": "F_closure",
            "semantic_acceptance": "not_automatically_assessed_manual_review_required",
        }
        row.update(timing_summary(selected, baseline, expected_repeats=EXPECTED_REPEATS, baseline_config="F_closure"))
        assert row["timing_complete"] and row["timing_pairing_status"] == "complete"
        n = case["n_target_tokens"]
        region = case["regions"].get(config)
        count = n if region is None else region["n_active_target_tokens"]
        row.update(n_target_tokens=n, n_active_partial_target_tokens=count, partial_target_fraction=count / n,
            all_query_active_fraction=1.0 if region is None else region["all_query_active_fraction"],
            perimeter_grid_edges=None if region is None else region["perimeter_grid_edges"],
            connected_components_4=None if region is None else region["connected_components_4"],
            region_mask_sha256=None if region is None else region["mask_sha256"],
            region_indices_sha256=None if region is None else region["indices_sha256"],
            quality_run_id=measurement_zero["run_id"], quality_output=measurement_zero["output"],
            quality_output_sha256=measurement_zero["output_sha256"],
            initial_latent_sha256=measurement_zero["initial_latent_sha256"],
            full_steps_1based=json.dumps(measurement_zero["full_steps_1based"]),
            measurement_dit_calls=sum(int(item["dit_calls"]) for item in selected),
            warmup_requests=1, warmup_dit_calls=8,
            measurement_request_e2e_seconds_sum=sum(float(item["e2e_seconds"]) for item in selected))
        row.update(quality_metrics(output, reference, source, masks, network, args.device))
        rows.append(row)
        timing_rows.extend(timing_detail(item, baseline, "posthoc_single_case_carrier_closure", baseline_config="F_closure") for item in selected)
    assert len(rows) == 3
    labels = ["INPUT", "F_closure", "R4", "R5"]
    comparison_path = figures / "remove_hair_closure_comparison.png"
    crop_path = figures / "remove_hair_closure_core_crop.png"
    comparison_panel([images[label] for label in labels], labels, comparison_path)
    ys, xs = np.nonzero(masks["core"])
    crop = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    comparison_panel([images[label] for label in labels], labels, crop_path, crop=crop)
    save_csv(data / "closure_quality_metrics.csv", rows)
    save_csv(data / "closure_timings.csv", timing_rows)
    by_config = {row["config"]: row for row in rows}
    r4, r5 = by_config["R4"], by_config["R5"]
    summary.update(status="complete", measurement_requests=9, warmup_requests=3,
        quality_rows=3, quality_device=args.device, quality_repeats_evaluated=[0],
        repeat_count_limit="Three timing repeats of one seed are not three independent quality samples, confidence intervals or a stable P95 estimate.",
        timings={config: {key: value for key, value in row.items() if key.startswith(("timing_", "e2e_", "paired_"))} for config, row in by_config.items()},
        metrics=rows,
        R4_vs_equal_budget_R5={
            "target_token_count_equal": r4["n_active_partial_target_tokens"] == r5["n_active_partial_target_tokens"],
            "band2_mae_relative_improvement_R4": relative_improvement(r5["band2_mae_to_F"], r4["band2_mae_to_F"]),
            "band2_gradient_relative_improvement_R4": relative_improvement(r5["band2_gradient_mae_to_F"], r4["band2_gradient_mae_to_F"]),
            "R4_latency_ratio_to_R5": r4["e2e_median_seconds"] / r5["e2e_median_seconds"],
            "formal_boundary_go_inferred": False,
        },
        artifacts={"metrics_csv": "data/closure_quality_metrics.csv", "timings_csv": "data/closure_timings.csv",
            "comparison": str(comparison_path.relative_to(root)), "common_R0_core_crop": str(crop_path.relative_to(root))},
        manual_visual_review_required=True,
        completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    write_json(summary_path, summary)
    environment = metric_environment(lpips, args.device, root)
    environment["shared_analysis_script_sha256"] = environment.pop("script_sha256")
    environment.update(script_sha256=sha256(Path(__file__)), closure_probe_sha256=sha256(probe_path),
        posthoc=True, case_id=CASE_ID, baseline_config="F_closure", thresholds_changed=False)
    write_json(data / "closure_metrics_environment.json", environment)
    print(json.dumps({"status": "complete", "posthoc": True, "case_id": CASE_ID,
        "quality_rows": len(rows), "successful_requests": len(successful),
        "known_successful_dit_calls": summary["known_successful_dit_calls"],
        "reference_png_and_pixels_identical": True, "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
