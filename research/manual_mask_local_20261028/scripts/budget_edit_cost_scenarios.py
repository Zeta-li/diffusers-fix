"""Recompute design scenarios from archived latency; performs no model inference."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    latency_path = ROOT / "runs/latency_temporal_v1/analysis/latency.json"
    overhead_path = ROOT / "runs/latency_temporal_v1/analysis/overheads.json"
    latency = json.loads(latency_path.read_text())["aggregate"]
    overhead = json.loads(overhead_path.read_text())
    baseline = latency["F"]["mean_source_median_seconds"]
    dit = latency["F"]["component_mean_source_medians_seconds"]["dit"]
    residual = baseline - dit
    target = condition = 4096
    text = 1536  # Illustrative assumption, not a measured sample mean.
    layers, width, mlp_width, rank = 60, 3072, 12288, 8
    profiles = [
        ("C", [1, 2, 5, 8], 0.5),
        ("B", [1, 2, 8], 0.25),
        ("S", [1, 2], 0.25),
    ]
    rows = []
    for name, anchors, fraction in profiles:
        local = len(range(8)) - len(anchors)
        full_rows = target + condition + text
        queries = len(anchors) * full_rows + local * fraction * target
        rows.append({
            "profile": name,
            "anchors_one_based": anchors,
            "requested_local_target_fraction": fraction,
            "mean_query_row_fraction": queries / (8 * full_rows),
            "assumes_core_fits_requested_budget": True,
            "is_latency_prediction": False,
        })
    scenarios = []
    for ratio, shared in [(0.55, residual), (0.45, residual),
                          (0.35, residual), (0.45, 2.2)]:
        small = 0.05
        estimate = ratio * dit + shared + small
        scenarios.append({
            "assumed_dit_time_fraction": ratio,
            "assumed_shared_seconds": shared,
            "assumed_student_extra_seconds": small,
            "scenario_total_seconds": estimate,
            "scenario_speedup_vs_archived_F": baseline / estimate,
            "scenario_speedup_vs_equally_optimized_F": (dit + shared) / estimate,
        })
    result = {
        "date": "2026-10-30",
        "status": "DESIGN_SCENARIOS_NOT_GPU_MEASUREMENTS",
        "new_requests": 0,
        "new_dit_forwards": 0,
        "new_training_updates": 0,
        "new_rl_updates": 0,
        "inputs": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (latency_path, overhead_path)
        },
        "measured": {
            "F_seconds": baseline,
            "FIX32_seconds": latency["FIX32"]["mean_source_median_seconds"],
            "F_dit_seconds": dit,
            "FIX32_query_row_fraction": overhead["mean_joint_query_row_fraction"],
        },
        "derived_baseline_residual_seconds": residual,
        "decomposition_limitation": "Aggregate wall-clock minus aggregated CUDA interval; approximate Amdahl design input.",
        "illustrative_geometry": {
            "target_tokens": target,
            "condition_tokens": condition,
            "text_tokens_assumed_not_measured": text,
            "layers": layers,
            "hidden_width": width,
        },
        "profiles": rows,
        "raw_context_kv_GiB": 2 * layers * (target + condition + text) * width * 2 / 2**30,
        "lora_parameters_if_geometry_matches": layers * 2 * width * rank + 12 * (mlp_width + width) * rank,
        "latency_scenarios": scenarios,
        "preflight_dit_budget": {
            "contracts": 32,
            "vae_check": 32,
            "main": 256,
            "repeat": 32,
            "warmup": 48,
            "total": 400,
        },
        "training_forward_budget": {
            "optimizer_updates": 1000,
            "microbatches": 4000,
            "student_gradient_window_forwards": 5000,
            "teacher_window_forwards": 5000,
            "student_prefix_forwards_upper_bound": 28000,
            "initial_full_teacher_reference_forwards": 4096,
            "listed_forward_total_upper_bound": 42096,
            "excludes": ["additional full-student consistency calls", "calibration generation",
                         "export validation", "backward compute", "quality evaluator compute"],
        },
    }
    output = ROOT / "data/budget_edit_design_scenarios_v1.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "query_rows": rows,
                      "raw_cache_GiB": result["raw_context_kv_GiB"],
                      "latency_scenarios": scenarios}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
