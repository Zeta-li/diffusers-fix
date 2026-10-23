import argparse
import importlib.util
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


DEFAULT_CONFIGS = {"primary": ["F", "R2", "FIX0", "FIX1", "FIX2", "ROT"], "optimization": ["F", "R2", "CHANNEL"]}


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def config_id(row):
    value = row.get("config", "")
    return value.get("id", "") if isinstance(value, dict) else str(value)


def display(value):
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return "N/A"
    return f"{value:.5g}" if isinstance(value, (float, int)) else str(value)


def resolve(value, root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def configuration_summaries(rows, configs, case_count):
    summaries = {}
    for config in configs:
        selected = [row for row in rows if row["config"] == config]
        ratios = [row["paired_geometric_latency_ratio_to_baseline"] for row in selected if row.get("paired_geometric_latency_ratio_to_baseline") is not None]
        quality = [row for row in selected if row.get("quality_status") == "complete"]
        ratio = math.exp(statistics.mean(math.log(value) for value in ratios)) if ratios else None
        gates_all = all(row.get("predefined_quality_gate_pass") for row in quality) if len(quality) == case_count else None
        summaries[config] = {
            "baseline_config": "F",
            "case_median_seconds": {row["case_id"]: row["e2e_median_seconds"] for row in selected},
            "case_CV": {row["case_id"]: row["e2e_cv"] for row in selected},
            "all_expected_repeats_complete": len(selected) == case_count and all(row["timing_complete"] for row in selected),
            "all_timing_pairs_complete": len(selected) == case_count and all(row["timing_pairing_status"] == "complete" for row in selected),
            "paired_geometric_latency_ratio_to_F_across_cases": ratio,
            "descriptive_net_latency_reduction_fraction": 1 - ratio if ratio is not None else None,
            "quality_cases_complete": len(quality),
            "quality_gate_pass_cases": [row["case_id"] for row in quality if row.get("predefined_quality_gate_pass")],
            "quality_gate_pass_all_cases": gates_all,
            "mean_lpips_mapmean_to_F": statistics.mean(row["lpips_mapmean_to_F"] for row in quality) if quality else None,
            "max_extra_background_mae_over_F": max((row["extra_background_mae_over_F"] for row in quality if row.get("extra_background_mae_over_F") is not None), default=None),
            "all_repeated_output_hashes_identical": len(selected) == case_count and all(row["distinct_output_hashes"] == 1 and row["timing_complete"] for row in selected),
            "max_allocated_GiB": max((row["peak_allocated_GiB"] for row in selected if row.get("peak_allocated_GiB") is not None), default=None),
            "exploratory_numeric_target_met": bool(gates_all and ratio <= 0.9) if gates_all is not None and ratio is not None else None,
            "formal_Go_established": False,
        }
    return summaries


def fixed_comparisons(rows, cases):
    lookup = {(row["case_id"], row["config"]): row for row in rows}
    result = []
    for case in cases:
        cid = case["case_id"]
        fixed = [lookup.get((cid, config)) for config in ["FIX0", "FIX1", "FIX2"]]
        rotation = lookup.get((cid, "ROT"))
        comparison = {"case_id": cid}
        if rotation is None or any(row is None or row.get("quality_status") != "complete" for row in fixed) or rotation.get("quality_status") != "complete":
            comparison["status"] = "incomplete"
            result.append(comparison)
            continue
        comparison["status"] = "complete"
        comparison["fixed_quality_gate_pass_configs"] = [row["config"] for row in fixed if row.get("predefined_quality_gate_pass")]
        comparison["ROT_quality_gate_pass"] = rotation.get("predefined_quality_gate_pass")
        for metric in ["lpips_mapmean_to_F", "mae_to_F", "band2_mae_to_F", "band2_gradient_mae_to_F", "extra_background_mae_over_F", "e2e_median_seconds"]:
            values = [row.get(metric) for row in fixed]
            current = rotation.get(metric)
            if any(value is None for value in values) or current is None:
                continue
            mean = statistics.mean(values)
            best = min(fixed, key=lambda row: row[metric])
            comparison[f"{metric}_FIX_mean"] = mean
            comparison[f"{metric}_FIX_min"] = best[metric]
            comparison[f"{metric}_FIX_max"] = max(values)
            comparison[f"{metric}_best_FIXED_config"] = best["config"]
            comparison[f"{metric}_ROT"] = current
            comparison[f"{metric}_improvement_vs_FIX_mean"] = mean - current
            comparison[f"{metric}_improvement_vs_best_FIXED"] = best[metric] - current
            if metric != "extra_background_mae_over_F":
                comparison[f"{metric}_relative_improvement_vs_FIX_mean"] = (mean - current) / mean if mean else None
        result.append(comparison)
    return result



def audit_age_trace(row, rotation_plan):
    config = config_id(row)
    trace = row.get("age_trace", [])
    result = {"case_id": row["case_id"], "config": config, "repeat": int(row["repeat"]), "run_id": row["run_id"]}
    if config == "F":
        result.update(native=True, age_trace_count=len(trace), all_steps_match=len(trace) == 0)
        return result
    expected_config = config if config in ["R2", "FIX0", "FIX1", "FIX2", "ROT"] else "R2" if config in ["SHIFT", "AFFINE", "CHANNEL"] else None
    expected = rotation_plan.get("cases", {}).get(row["case_id"], {}).get("configs", {}).get(expected_config, {})
    expected_steps = {value["step_1based"]: value for value in expected.get("per_step_audit", [])}
    audits = []
    for entry in trace:
        step = int(entry["step"])
        planned = expected_steps.get(step)
        item = {"step": step, "selected_target": entry.get("selected_target"), "max_read_age": entry.get("max_read_age"), "mean_read_age": entry.get("mean_read_age"), "age_histogram": entry.get("age_histogram"), "newly_activated": entry.get("newly_activated")}
        if planned:
            item.update(expected_selected_target=planned["selected_target_count"], expected_max_read_age=planned["max_age_at_attention_read"],
                selected_count_matches=entry.get("selected_target") == planned["selected_target_count"], max_read_age_matches=entry.get("max_read_age") == planned["max_age_at_attention_read"],
                histogram_matches=entry.get("age_histogram") == planned["age_at_attention_read_histogram"])
        audits.append(item)
    result.update(native=False, expected_schedule_config=expected_config, age_trace_count=len(trace), steps=audits,
        max_observed_read_age=max((entry.get("max_read_age", 0) for entry in trace), default=None),
        all_steps_match=len(trace) == 8 and {int(entry["step"]) for entry in trace} == set(range(1, 9)) and all(value.get("selected_count_matches") and value.get("max_read_age_matches") and value.get("histogram_matches") for value in audits))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--previous-root", type=Path)
    parser.add_argument("--phase", choices=["primary", "optimization"], default="primary")
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    assert args.repeats > 0
    root, phase = args.root.resolve(), args.phase
    data = root / "data"
    figure_dir = root / "figures" / phase
    figure_dir.mkdir(parents=True, exist_ok=True)
    configs = args.configs or DEFAULT_CONFIGS[phase]
    assert configs[0] == "F" and len(configs) == len(set(configs))
    environment_path = data / f"environment_{phase}.json"
    environment = json.loads(environment_path.read_text()) if environment_path.exists() else {}
    previous_value = args.previous_root or environment.get("previous_root") or environment.get("local_pilot_root")
    previous = resolve(previous_value, root) if previous_value else root.parent / "局部执行与边界验证_20261022"
    helper_path = previous / "scripts/analyze_local_pilot.py"
    spec = importlib.util.spec_from_file_location("rotation_quality_helpers", helper_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    cases_path = previous / "cases.json"
    regions_path = previous / "masks/regions.json"
    regions = json.loads(regions_path.read_text())
    cases = json.loads(cases_path.read_text())["cases"] if cases_path.exists() else [{"case_id": cid, "image": info["source_image"]} for cid, info in regions["cases"].items()]
    runs_path = args.runs.resolve() if args.runs else data / f"runs_{phase}.jsonl"
    raw = load_lines(runs_path)
    successful = {}
    duplicate_run_ids = []
    for row in raw:
        if row.get("status") != "complete":
            continue
        if row["run_id"] in successful:
            duplicate_run_ids.append(row["run_id"])
        successful[row["run_id"]] = row
    relevant = [row for row in successful.values() if config_id(row) in configs]
    measurements = [row for row in relevant if row.get("kind") == "measurement"]
    lookup = {(row["case_id"], config_id(row), int(row["repeat"])): row for row in measurements}
    old_runs = load_lines(previous / "data/runs.jsonl")
    old_lookup = {(row["case_id"], config_id(row), int(row.get("repeat", 0))): row for row in old_runs if row.get("status") == "complete" and row.get("kind") == "measurement"}
    warnings, metrics, timing_details, baseline_audits, age_traces = [], [], [], [], []
    rotation_plan_path = root / "masks/rotation_plan.json"
    rotation_plan = json.loads(rotation_plan_path.read_text()) if rotation_plan_path.exists() else {}
    age_audits = []
    torch.set_num_threads(8)
    import lpips
    network = lpips.LPIPS(net="alex", spatial=True).eval().to(args.device)
    for case in cases:
        cid = case["case_id"]
        geometry = regions["cases"][cid]
        height, width = geometry["target_hw"]
        source = Image.open(previous / case.get("image", geometry["source_image"])).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        masks = helpers.evaluation_masks(geometry, previous, height, width)
        f_rows = {repeat: row for (key, config, repeat), row in lookup.items() if key == cid and config == "F"}
        f_zero = f_rows.get(0)
        reference = None
        if f_zero:
            try:
                reference = helpers.load_image(root, f_zero, (width, height))
            except Exception as error:
                warnings.append(f"{cid}: invalid same-phase F reference: {type(error).__name__}: {error}")
        images = {"INPUT": source}
        for config in configs:
            selected = sorted([row for (key, name, repeat), row in lookup.items() if key == cid and name == config], key=lambda row: int(row["repeat"]))
            result = {"phase": phase, "case_id": cid, "config": config}
            result.update(helpers.timing_summary(selected, f_rows, expected_repeats=args.repeats, baseline_config="F"))
            if selected and result["distinct_output_hashes"] != 1:
                warnings.append(f"Repeated output SHA differs: {cid}/{config}")
            zero = lookup.get((cid, config, 0))
            if zero is None or reference is None:
                result.update(quality_status="missing_repeat0_or_F", predefined_quality_gate_pass=None)
            else:
                try:
                    assert zero.get("initial_latent_sha256") == f_zero.get("initial_latent_sha256"), "Initial latent differs from F"
                    for key in ["timesteps", "sigmas"]:
                        if key in zero or key in f_zero:
                            assert zero.get(key) == f_zero.get(key), f"{key} differs from F"
                    output = helpers.load_image(root, zero, reference.size)
                    result.update(helpers.quality_metrics(output, reference, source, masks, network, args.device))
                    result.update(quality_output=zero["output"], quality_output_sha256=zero.get("output_sha256"), quality_run_id=zero["run_id"])
                    images[config] = output
                except Exception as error:
                    result.update(quality_status="invalid_output", quality_error=f"{type(error).__name__}: {error}", predefined_quality_gate_pass=None)
                    warnings.append(f"{cid}/{config}: {result['quality_error']}")
            metrics.append(result)
            for row in selected:
                timing_details.append(helpers.timing_detail(row, f_rows, phase, baseline_config="F"))
                age_traces.append({"case_id": cid, "config": config, "repeat": int(row["repeat"]), "run_id": row["run_id"], "age_trace": row.get("age_trace"), "trace": row.get("trace")})
                age_audit = audit_age_trace(row, rotation_plan)
                age_audits.append(age_audit)
                if not age_audit["all_steps_match"]:
                    warnings.append(f"Runtime age/selection differs from frozen plan or trace missing: {row['run_id']}")
            if config in ["F", "R2"]:
                old = old_lookup.get((cid, config, 0))
                audit = {"case_id": cid, "config": config, "phase": phase}
                if zero is None or old is None:
                    audit["status"] = "missing_reference"
                else:
                    try:
                        new_image = helpers.load_image(root, zero, (width, height))
                        old_image = helpers.load_image(previous, old, (width, height))
                        audit.update(status="complete", old_output_sha256=old.get("output_sha256"), new_output_sha256=zero.get("output_sha256"), png_sha256_equal=zero.get("output_sha256") == old.get("output_sha256"), pixel_identical=bool(np.array_equal(np.asarray(new_image), np.asarray(old_image))))
                        if not audit["png_sha256_equal"] or not audit["pixel_identical"]:
                            warnings.append(f"{phase}/{cid}/{config}: output differs from the old baseline")
                    except Exception as error:
                        audit.update(status="invalid_reference", error=f"{type(error).__name__}: {error}")
                baseline_audits.append(audit)
        labels = ["INPUT"] + configs
        helpers.comparison_panel([images.get(label) for label in labels], labels, figure_dir / f"{cid}_comparison.png")
        ys, xs = np.nonzero(masks["core"])
        crop = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        helpers.comparison_panel([images.get(label) for label in labels], labels, figure_dir / f"{cid}_core_crop.png", crop=crop)
    expected_measurements = len(cases) * len(configs) * args.repeats
    expected_warmups = len(cases) * len(configs)
    complete = len(measurements) == expected_measurements and all(row["timing_complete"] and row["quality_status"] == "complete" for row in metrics)
    config_summaries = configuration_summaries(metrics, configs, len(cases))
    comparisons = fixed_comparisons(metrics, cases) if phase == "primary" and all(name in configs for name in ["FIX0", "FIX1", "FIX2", "ROT"]) else []
    order_audits = []
    planned_execution = [(case["case_id"], config, "warmup", 0) for case in cases for config in configs]
    for repeat in range(args.repeats):
        for case_index, case in enumerate(cases):
            shift = (repeat + case_index) % len(configs)
            order = [config_id(row) for row in measurements if row["case_id"] == case["case_id"] and int(row["repeat"]) == repeat]
            expected = configs[shift:] + configs[:shift]
            order_audits.append({"case_id": case["case_id"], "case_index": case_index, "repeat": repeat, "rotation_shift": shift, "expected_cyclic_order": expected, "observed_log_order": order, "matches": order == expected})
            planned_execution.extend((case["case_id"], config, "measurement", repeat) for config in expected)
    observed_execution = [(row["case_id"], config_id(row), row.get("kind"), int(row.get("repeat", 0))) for row in relevant]
    execution_order_audit = {"case_order_from_previous_cases_json": [case["case_id"] for case in cases],
        "rule": "All case/config warmups first; then repeat outer loop, case order from previous cases.json inner loop, configuration shift=(repeat+case_index)%len(configs).",
        "planned_execution": [list(value) for value in planned_execution], "observed_execution": [list(value) for value in observed_execution],
        "complete_order_matches": planned_execution == observed_execution}
    summary = {
        "phase": phase,
        "status": "complete" if complete else "partial",
        "status_scope": "Data aggregation completeness only; not acceptance. Inspect warnings, runtime-age, hash, execution-order, quality and manual visual audits independently.",
        "configs": config_summaries,
        "counts": {"raw_records": len(raw), "successful_unique_relevant_records": len(relevant), "expected_successful_records": expected_measurements + expected_warmups, "measurements": len(measurements), "expected_measurements": expected_measurements, "warmups": sum(row.get("kind") == "warmup" for row in relevant), "expected_warmups": expected_warmups, "actual_recorded_dit_calls": sum(row.get("dit_calls", 0) for row in relevant), "expected_dit_calls": 8 * (expected_measurements + expected_warmups)},
        "failed_records": [row for row in raw if row.get("status") == "failed"],
        "duplicate_successful_run_ids": duplicate_run_ids,
        "warnings": warnings,
        "baseline_F_R2_audits": baseline_audits,
        "repeated_output_sha_audit_all_passed": complete and all(row["distinct_output_hashes"] == 1 for row in metrics),
        "measurement_order_audits": order_audits,
        "execution_order_audit": execution_order_audit,
        "rotation_vs_fixed_controls": comparisons,
        "recorded_age_and_shape_traces": age_traces,
        "runtime_age_audits": age_audits,
        "runtime_age_audits_all_passed": bool(age_audits) and all(row["all_steps_match"] for row in age_audits),
        "quality_rows": metrics,
        "paths": {"runs": str(runs_path), "environment": str(environment_path), "previous_root": str(previous), "regions": str(regions_path), "helpers": str(helper_path)},
        "protocol": {"case_count": len(cases), "repeats": args.repeats, "quality_repeat": 0, "timing_baseline": "same-phase F, same case/repeat cycle", "execution_order": execution_order_audit["rule"], "phase_separation": "Primary and optimization use different raw files, baselines, summaries, CSVs, Markdown and figure folders. Never pool repetitions across phases.",
            "quality": "Unmasked RGB LPIPS Alex spatial map, averaged globally or over common old R0 bands; source resized with LANCZOS; far background from old allowed_box. F is a computational reference, not true editing ground truth.",
            "extra_background": "MAE(candidate,source,far)-MAE(same-phase F,source,far), not MAE(candidate,F,far). Negative values can occur and do not prove successful editing.",
            "frozen_gate": {"lpips_mapmean_to_F_max": .02, "extra_background_mae_over_F_max": 1 / 255},
            "formal_target": "Quality gate plus >=10% end-to-end net latency reduction relative to F. Three previously viewed demos and three timing repetitions only screen this target; no formal Go, significance, noninferiority, CI or independent-seed success-rate claim.",
            "fixed_controls": "Compare ROT with the arithmetic mean of three FIX metrics and also disclose each FIX and metric-specific best FIX. Per-metric best names may differ; these are hindsight diagnostic controls, not an automatically selectable deployable oracle.",
            "optimization_controls": "CHANNEL is the projection-aware per-output-channel Value correction using zero mean and unit variance priors, without sample calibration. Compare with contemporaneous F/R2 and count all correction work. Earlier scalar and rotation results remain separate cohorts.",
            "visual_review": "Manual visual review is performed separately. Numeric gate passing does not establish absence of brightness blocks, seams, text errors or identity changes."},
    }
    helpers.save_csv(data / f"quality_{phase}.csv", metrics)
    helpers.save_csv(data / f"timings_{phase}.csv", timing_details)
    (data / f"summary_{phase}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    metric_environment = helpers.metric_environment(lpips, args.device, previous)
    metric_environment.update(phase=phase, script_sha256=helpers.sha256(Path(__file__)), raw_runs_sha256=helpers.sha256(runs_path))
    (data / f"metrics_environment_{phase}.json").write_text(json.dumps(metric_environment, ensure_ascii=False, indent=2))
    title = "轮转实验数据摘要" if phase == "primary" else "优化数据摘要"
    lines = [f"# {title}", "", f"阶段：`{phase}`；数据汇总状态：**{summary['status']}**。三个已查看的演示输入，每配置 {args.repeats} 次计时，质量仅取 repeat0。complete仅表示数据汇总齐备，不代表所有验收通过。", "",
        "实际交错：先完成全部case/config预热，再按repeat外循环、源cases.json中的case顺序内循环，配置循环位移为 `(repeat+case_index)%配置数`；同case/repeat配对F。该描述核对实际runner，不修改已冻结的主协议。", "",
        f"完成 {len(relevant)}/{expected_measurements + expected_warmups} 条相关请求，记录 {summary['counts']['actual_recorded_dit_calls']}/{summary['counts']['expected_dit_calls']} 次 DiT。预热、失败和重复成功记录分别统计，原始失败行不删除。", "",
        "## 1 同期 F 配对结果", "", "| 输入 | 配置 | 中位秒 | CV | 配对几何耗时比/F | LPIPS/F | 公共band2 MAE/F | extraBG/F | 数值门槛 |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in metrics:
        values = [row["case_id"], row["config"], row["e2e_median_seconds"], row["e2e_cv"], row["paired_geometric_latency_ratio_to_baseline"], row.get("lpips_mapmean_to_F"), row.get("band2_mae_to_F"), row.get("extra_background_mae_over_F"), row.get("predefined_quality_gate_pass")]
        lines.append("| " + " | ".join(display(value) for value in values) + " |")
    lines += ["", "extraBG 是候选相对源图的远背景 MAE 减去同轮 F 的对应 MAE。数值门槛为 LPIPS≤0.02 且 extraBG≤1/255；通过不等于语义正确或没有局部接缝。CV 描述计时重复，不是质量或泛化不确定性。", ""]
    if comparisons:
        lines += ["## 2 ROT 与等预算固定组", "", "| 输入 | FIX平均LPIPS | 最佳FIX LPIPS（组） | ROT LPIPS | FIX平均band2 | 最佳FIX band2（组） | ROT band2 |", "|---|---:|---|---:|---:|---|---:|"]
        for row in comparisons:
            if row["status"] != "complete":
                lines.append(f"| {row['case_id']} | N/A | 未完整 | N/A | N/A | 未完整 | N/A |")
                continue
            best_lpips = f"{display(row['lpips_mapmean_to_F_FIX_min'])} ({row['lpips_mapmean_to_F_best_FIXED_config']})"
            best_band = f"{display(row['band2_mae_to_F_FIX_min'])} ({row['band2_mae_to_F_best_FIXED_config']})"
            values = [row["case_id"], row["lpips_mapmean_to_F_FIX_mean"], best_lpips, row["lpips_mapmean_to_F_ROT"], row["band2_mae_to_F_FIX_mean"], best_band, row["band2_mae_to_F_ROT"]]
            lines.append("| " + " | ".join(display(value) for value in values) + " |")
        lines += ["", "JSON 同时保存三个 FIX 各项指标的均值、范围、最佳组及 ROT 差值。最佳组按指标可能不同，不能拼成虚构的最佳输出。联合刷新 K/V 与速度、覆盖位置和轨迹历史仍耦合，不能将差异当纯 KV 年龄因果效应。", ""]
    elif phase == "optimization":
        lines += ["## 2 逐通道 Value 校正", "", "CHANNEL 与同轮 F/R2 比较，逐通道查表和调制成本已在请求计时内。本轮使用均值0、方差1的先验，未使用Train8校准；此前标量与轮转结果不并入本阶段重复。是否具备质量收益需结合完整结果和单独视觉检查。", ""]
    lines += ["## 3 一致性与结论边界", "", "| 输入 | 配置 | 旧基线SHA一致 | RGB一致 |", "|---|---|---|---|"]
    for audit in baseline_audits:
        lines.append("| " + " | ".join(display(value) for value in [audit["case_id"], audit["config"], audit.get("png_sha256_equal"), audit.get("pixel_identical")]) + " |")
    lines += ["", f"所有计时重复的输出 SHA 一致：**{display(summary['repeated_output_sha_audit_all_passed'])}**；运行时选区/读缓存年龄与冻结计划一致：**{display(summary['runtime_age_audits_all_passed'])}**；预热及测量完整交错顺序一致：**{display(execution_order_audit['complete_order_matches'])}**。", "",
        "正式工程目标是质量过门并相对 F 净延迟下降≥10%；三案例探索不建立正式 Go。若仅有0–10%正收益，记为未达目标/Revise。没有执行显著性检验、置信区间或非劣性论证；后续必须使用隔离 Train/Val，Test 继续封存。", ""]
    if warnings:
        lines += ["审计提示：", ""] + [f"- {warning}" for warning in warnings] + [""]
    lines += ["## 4 产物", ""]
    for case in cases:
        cid = case["case_id"]
        lines += [f"- [{cid} 完整对照](figures/{phase}/{cid}_comparison.png)", f"- [{cid} 原分辨率核心裁剪](figures/{phase}/{cid}_core_crop.png)"]
    lines += [f"- [耗时图](figures/{phase}/latency.png)", f"- [指标CSV](data/quality_{phase}.csv)", f"- [计时CSV](data/timings_{phase}.csv)", f"- [完整JSON](data/summary_{phase}.json)", "", "人工视觉结论由实际检查另行补充，本脚本不自动填入编辑成功或算法已解决问题的判断。", ""]
    filename = "02_轮转实验数据摘要.md" if phase == "primary" else "03_优化数据摘要.md"
    (root / filename).write_text("\n".join(lines))
    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.25), 4.3))
    width_bar = .23
    metric_lookup = {(row["case_id"], row["config"]): row for row in metrics}
    for index, case in enumerate(cases):
        values = [metric_lookup[(case["case_id"], config)]["e2e_median_seconds"] for config in configs]
        values = [value if value is not None else float("nan") for value in values]
        positions = np.arange(len(configs)) + (index - (len(cases) - 1) / 2) * width_bar
        ax.bar(positions, values, width=width_bar, label=case["case_id"])
        for position, config in zip(positions, configs):
            samples = [row["e2e_seconds"] for row in measurements if row["case_id"] == case["case_id"] and config_id(row) == config]
            ax.scatter([position] * len(samples), samples, color="black", s=10, alpha=.5, zorder=3)
    ax.set_xticks(np.arange(len(configs)), configs)
    ax.set_ylabel(f"End-to-end seconds (median of {args.repeats}; dots = repeats)")
    ax.set_title(f"{phase}: same-round F baseline, supplied region, fresh request cache")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "latency.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"phase": phase, "status": summary["status"], "counts": summary["counts"], "configs": config_summaries, "rotation_vs_fixed_controls": comparisons, "warnings": warnings}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
