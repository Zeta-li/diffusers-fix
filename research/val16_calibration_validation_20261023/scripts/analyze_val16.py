import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image


CONFIGS = ["F", "R2", "PRIOR", "A", "B"]
REPEATS = 3
BOOTSTRAP_SEED = 2026102304
BOOTSTRAP_DRAWS = 10000


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve(root, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def config_id(row):
    value = row.get("config", "")
    return str(value.get("id", "")) if isinstance(value, dict) else str(value)


def dilate(mask, radius):
    height, width = mask.shape
    padded = np.pad(mask, radius, constant_values=False)
    result = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result |= padded[radius + dy:radius + dy + height, radius + dx:radius + dx + width]
    return result


def upsample(mask, height, width):
    if height % mask.shape[0] or width % mask.shape[1]:
        raise ValueError("Evaluation grid does not divide output canvas")
    return np.repeat(np.repeat(mask, height // mask.shape[0], axis=0), width // mask.shape[1], axis=1)


def case_geometry(case):
    height, width = [int(value) for value in case["target_hw"]]
    gh, gw = [int(value) for value in case["target_grid_hw"]]
    if (height, width) != (gh * 16, gw * 16):
        raise ValueError(f"Unexpected Qwen token geometry: {case['case_id']}")
    bbox = [int(value) for value in case["target_mask_bbox_xyxy_exclusive"]]
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Invalid source annotation bbox: {case['case_id']}")
    token_bbox = [x0 // 16, y0 // 16, (x1 + 15) // 16, (y1 + 15) // 16]
    core = np.zeros((gh, gw), dtype=bool)
    xx0, yy0, xx1, yy1 = token_bbox
    core[yy0:yy1, xx0:xx1] = True
    r2 = dilate(core, 2)
    actual = np.flatnonzero(r2).tolist()
    if actual != [int(value) for value in case["indices"]]:
        raise ValueError(f"R2 indices differ from annotation bbox plus halo2: {case['case_id']}")
    if not core.any() or core.all() or r2.all():
        raise ValueError(f"Empty/full core or missing far background: {case['case_id']}")
    return core, r2, token_bbox


def prepare_regions(root, cases_path, cases):
    destination = root / "masks/evaluation_regions.json"
    if destination.exists():
        regions = load_regions(root, cases_path, cases)
        return dict(status="already_frozen_and_verified", regions_sha256=sha256(destination), cases=len(regions["cases"]))
    if any((root / "data").glob("runs*.jsonl")):
        raise ValueError("Refusing to freeze evaluation regions after a run ledger exists")
    records = {}
    for case in cases:
        cid = case["case_id"]
        core, r2, bbox = case_geometry(case)
        folder = root / "masks" / cid
        folder.mkdir(parents=True, exist_ok=True)
        assets = {}
        for name, grid in [("R0", core), ("R2", r2)]:
            path = folder / f"{name}.npy"
            np.save(path, grid, allow_pickle=False)
            assets[name] = dict(path=str(path.relative_to(root)), sha256=sha256(path), tokens=int(grid.sum()))
        source = resolve(root, case["image"])
        binary = resolve(root, case["binary_mask"])
        binary_mask = np.asarray(Image.open(binary).convert("L")) > 0
        hh, ww = case["target_hw"]
        if binary_mask.shape != (hh, ww):
            raise ValueError(f"Binary mask canvas differs from frozen output canvas: {cid}")
        ys, xs = np.nonzero(binary_mask)
        actual_bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        if actual_bbox != case["target_mask_bbox_xyxy_exclusive"]:
            raise ValueError(f"Binary mask and frozen bbox differ: {cid}")
        if case.get("source_sha256") and sha256(source) != case["source_sha256"]:
            raise ValueError(f"Source SHA differs: {cid}")
        if case.get("binary_mask_sha256") and sha256(binary) != case["binary_mask_sha256"]:
            raise ValueError(f"Binary mask SHA differs: {cid}")
        records[cid] = dict(target_hw=case["target_hw"], target_grid_hw=case["target_grid_hw"],
            annotation_target_bbox_xyxy_exclusive=case["target_mask_bbox_xyxy_exclusive"],
            R0_token_bbox_xyxy_exclusive=bbox, assets=assets,
            source=case["image"], source_sha256=sha256(source), binary_mask=case["binary_mask"],
            binary_mask_sha256=sha256(binary), far_background_tokens=int((~r2).sum()),
            far_background_fraction=float((~r2).mean()),
            band_tokens={f"band{radius}": int((dilate(core, radius) & dilate(~core, radius)).sum()) for radius in [1, 2, 4]})
    value = dict(protocol_version="val16_annotation_bbox_v1", configs=CONFIGS,
        cases_file=str(cases_path.relative_to(root)), cases_sha256=sha256(cases_path),
        case_ids=[case["case_id"] for case in cases], cases=records,
        protocol=dict(R0="Annotation binary positive bbox projected to target 16px grid by floor/ceil; no halo",
            R2="L-infinity dilation of R0 by 2 target tokens, finite canvas; equals frozen execution indices",
            bands="dilate_Linf(R0,r) & dilate_Linf(~R0,r), r=1,2,4; finite grid, no artificial image exterior",
            far_background="Complement of R2; differs from historical allowed-box far background",
            unmasked_lpips="Compute full RGB LPIPS Alex spatial map before region averaging; receptive fields cross masks",
            gate=dict(lpips_mapmean_to_F_max=.02, extra_background_mae_over_F_max=1 / 255),
            frozen_before_generation=True, adaptive_regions=False))
    save_json(destination, value)
    return dict(status="frozen", regions_sha256=sha256(destination), cases=len(records))


def load_regions(root, cases_path, cases):
    path = root / "masks/evaluation_regions.json"
    value = read_json(path)
    if value["cases_sha256"] != sha256(cases_path) or value["case_ids"] != [case["case_id"] for case in cases]:
        raise ValueError("Cases changed after evaluation geometry freeze")
    for case in cases:
        record = value["cases"][case["case_id"]]
        core, r2, _ = case_geometry(case)
        for name, expected in [("R0", core), ("R2", r2)]:
            asset = record["assets"][name]
            mask_path = root / asset["path"]
            if sha256(mask_path) != asset["sha256"] or not np.array_equal(np.load(mask_path, allow_pickle=False), expected):
                raise ValueError(f"Frozen mask changed: {case['case_id']}/{name}")
        for asset_key, hash_key in [("source", "source_sha256"), ("binary_mask", "binary_mask_sha256")]:
            if sha256(resolve(root, record[asset_key])) != record[hash_key]:
                raise ValueError(f"Frozen input changed: {case['case_id']}/{asset_key}")
    return value


def evaluation_masks(root, record):
    height, width = record["target_hw"]
    core = np.load(root / record["assets"]["R0"]["path"], allow_pickle=False)
    r2 = np.load(root / record["assets"]["R2"]["path"], allow_pickle=False)
    result = dict(core=upsample(core, height, width), far_background=upsample(~r2, height, width))
    for radius in [1, 2, 4]:
        result[f"band{radius}"] = upsample(dilate(core, radius) & dilate(~core, radius), height, width)
    return result


def load_helpers(root, supplied):
    candidates = [supplied] if supplied else []
    candidates += [root.parent / "local_executor_boundary_20261022", root.parent / "局部执行与边界验证_20261022"]
    for directory in candidates:
        path = Path(directory) / "scripts/analyze_local_pilot.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("val16_quality_helpers", path)
            helpers = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helpers)
            return helpers, path.resolve()
    raise FileNotFoundError("Use --previous-root for analyze_local_pilot.py")


def wilson(success, count):
    if not count:
        return None
    z = 1.959963984540054
    p = success / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0., center - margin), min(1., center + margin)]


def runtime_audit(row, case, calibration_sha256):
    config = config_id(row)
    n, k = int(case["n_target_tokens"]), len(case["indices"])
    condition = row.get("condition_count", 0)
    checks = dict(target_count=row.get("target_count") == n,
        condition_count=isinstance(condition, int) and condition > 0,
        dit_calls=row.get("dit_calls") == 8, no_oracles=row.get("oracle_calls") == 0,
        request_reset=row.get("request_reset") is True,
        cleanup=row.get("cleanup_all_executors_empty") is True,
        source=row.get("source_sha256") == case["source_sha256"],
        indices=row.get("core_indices_sha256") == hashlib.sha256(json.dumps(case["indices"]).encode()).hexdigest(),
        correction_mode=row.get("correction_mode") == ("none" if config in ["F", "R2"] else "affine"),
        coefficient_hash=row.get("coefficient_sha256") == (calibration_sha256 if config in ["A", "B"] else None),
        full_steps=row.get("full_steps_1based") == (list(range(1, 9)) if config == "F" else [1, 8]))
    traces, ages = row.get("trace", []), row.get("age_trace", [])
    steps = []
    if config == "F":
        checks.update(native=row.get("executor_class") == "native", no_cache=row.get("cache_bytes") == 0,
            no_trace=not traces and not ages)
    else:
        checks.update(cache=isinstance(row.get("cache_bytes"), int) and row["cache_bytes"] > 0,
            trace_count=len(traces) == 24, age_count=len(ages) == 8,
            age_step_set={entry.get("step") for entry in ages} == set(range(1, 9)))
        for step in range(8):
            active = n if step in [0, 7] else k
            stale = 0 if step in [0, 7] else step
            histogram = {"0": active} if stale == 0 else {"0": active, str(stale): n - active}
            newly = n - k if step == 7 else 0
            matches = [entry for entry in ages if entry.get("step") == step + 1]
            age = matches[0] if len(matches) == 1 else {}
            shapes = [entry for entry in traces if entry.get("step") == step + 1]
            shape_checks = len(shapes) == 3 and {entry.get("layer") for entry in shapes} == {0, 30, 59}
            for entry in shapes:
                text = entry.get("text_tokens", -1)
                shape_checks = shape_checks and text > 0 and all(entry.get(key) == expected for key, expected in {
                    "selected_target": active, "condition": condition,
                    "image_qkv_tokens": active + condition, "image_mlp_tokens": active + condition,
                    "joint_query_tokens": text + active + condition, "joint_kv_tokens": text + n + condition,
                    "newly_activated_from_current_latent": newly}.items())
            step_checks = dict(step_1based=step + 1, selected=age.get("selected_target") == active,
                maximum_age=age.get("max_read_age") == stale, histogram=age.get("age_histogram") == histogram,
                mean_age=math.isclose(age.get("mean_read_age", float("inf")), stale * (n - active) / n, abs_tol=1e-6),
                newly_activated=age.get("newly_activated") == newly, qkv_mlp_shapes=bool(shape_checks))
            steps.append(step_checks)
        checks["all_age_and_shape_steps"] = all(all(value for key, value in step.items() if key != "step_1based") for step in steps)
    return dict(run_id=row["run_id"], case_id=row["case_id"], config=config, checks=checks, steps=steps, passed=all(checks.values()))


def bootstrap_mean(values, exponential=False, complement=False):
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        return None
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))
    bootstrap = array[indices].mean(axis=1)
    observed = float(array.mean())
    if exponential:
        bootstrap, observed = np.exp(bootstrap), math.exp(observed)
    if complement:
        bootstrap, observed = 1 - bootstrap, 1 - observed
    return dict(point=observed, percentile95=[float(value) for value in np.quantile(bootstrap, [.025, .975])],
        source_images=len(array), draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, resampling_unit="source image")


def config_summary(metrics, config, case_count):
    selected = [row for row in metrics if row["config"] == config]
    quality = [row for row in selected if row.get("quality_status") == "complete"]
    timing = [row for row in selected if row.get("timing_complete") and row.get("paired_geometric_latency_ratio_to_F") is not None]
    passing = [row for row in quality if row.get("predefined_quality_gate_pass") is True]
    joint = [row for row in passing if row.get("paired_geometric_latency_ratio_to_F", float("inf")) <= .9]
    numeric_count = len(passing)
    return dict(case_count=case_count, complete_quality_cases=len(quality), complete_timing_cases=len(timing),
        median_seconds_across_case_medians=statistics.median(row["e2e_median_seconds"] for row in timing) if timing else None,
        paired_geometric_latency_ratio_to_F=bootstrap_mean([math.log(row["paired_geometric_latency_ratio_to_F"]) for row in timing], exponential=True),
        paired_net_latency_reduction=bootstrap_mean([math.log(row["paired_geometric_latency_ratio_to_F"]) for row in timing], exponential=True, complement=True),
        mean_lpips_to_F=bootstrap_mean([row["lpips_mapmean_to_F"] for row in quality]),
        mean_extra_background_mae_over_F=bootstrap_mean([row["extra_background_mae_over_F"] for row in quality]),
        numeric_gate_pass_count=numeric_count, numeric_gate_pass_denominator=case_count,
        numeric_gate_pass_fraction=numeric_count / case_count,
        numeric_gate_pass_wilson95=wilson(numeric_count, case_count),
        numeric_gate_pass_case_ids=[row["case_id"] for row in passing],
        numeric_quality_and_10percent_speed_pass_count=len(joint),
        numeric_quality_and_10percent_speed_pass_case_ids=[row["case_id"] for row in joint],
        numeric_gate_pass_all_cases=len(quality) == case_count and numeric_count == case_count,
        max_lpips_to_F=max((row["lpips_mapmean_to_F"] for row in quality), default=None),
        max_extra_background_mae_over_F=max((row["extra_background_mae_over_F"] for row in quality), default=None),
        max_timing_cv=max((row["e2e_cv"] for row in timing if row.get("e2e_cv") is not None), default=None),
        peak_allocated_GiB=max((row["peak_allocated_GiB"] for row in timing if row.get("peak_allocated_GiB") is not None), default=None),
        formal_Go_established=False, semantic_acceptance="requires separate nonblind image review")


def route_comparisons(metrics, case_ids):
    lookup = {(row["case_id"], row["config"]): row for row in metrics}
    result = {}
    for first, second in [("PRIOR", "R2"), ("A", "PRIOR"), ("B", "R2"), ("B", "PRIOR"), ("B", "A")]:
        rows = []
        for cid in case_ids:
            a, b = lookup[(cid, first)], lookup[(cid, second)]
            if a.get("quality_status") != "complete" or b.get("quality_status") != "complete":
                continue
            row = dict(case_id=cid, candidate=first, reference=second,
                lpips_delta=a["lpips_mapmean_to_F"] - b["lpips_mapmean_to_F"],
                band2_mae_delta=a["band2_mae_to_F"] - b["band2_mae_to_F"],
                extra_background_mae_delta=a["extra_background_mae_over_F"] - b["extra_background_mae_over_F"],
                numeric_pass_candidate=a["predefined_quality_gate_pass"], numeric_pass_reference=b["predefined_quality_gate_pass"])
            if a.get("paired_geometric_latency_ratio_to_F") and b.get("paired_geometric_latency_ratio_to_F"):
                row["paired_latency_ratio_candidate_over_reference"] = a["paired_geometric_latency_ratio_to_F"] / b["paired_geometric_latency_ratio_to_F"]
            rows.append(row)
        result[f"{first}_vs_{second}"] = dict(rows=rows,
            mean_lpips_delta=bootstrap_mean([row["lpips_delta"] for row in rows]),
            mean_band2_mae_delta=bootstrap_mean([row["band2_mae_delta"] for row in rows]),
            mean_extra_background_mae_delta=bootstrap_mean([row["extra_background_mae_delta"] for row in rows]),
            lpips_improved_cases=sum(row["lpips_delta"] < 0 for row in rows),
            lpips_worsened_cases=sum(row["lpips_delta"] > 0 for row in rows),
            numeric_pass_gains=sum(row["numeric_pass_candidate"] and not row["numeric_pass_reference"] for row in rows),
            numeric_pass_losses=sum(not row["numeric_pass_candidate"] and row["numeric_pass_reference"] for row in rows))
    return result


def box(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Empty evaluation crop")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def make_case_panels(root, case, images, masks, helpers):
    plt = plotting()
    cid = case["case_id"]
    folder = root / "figures/val16"
    labels = ["INPUT"] + CONFIGS
    core_crop, boundary_crop = box(masks["core"]), box(masks["core"] | masks["band4"])
    helpers.comparison_panel([images.get(label) for label in labels], labels, folder / f"{cid}_comparison.png")
    helpers.comparison_panel([images.get(label) for label in labels], labels, folder / f"{cid}_core_crop.png", crop=core_crop)
    helpers.comparison_panel([images.get(label) for label in labels], labels, folder / f"{cid}_boundary_crop.png", crop=boundary_crop)
    helpers.region_visualization(images["INPUT"], masks, folder / f"{cid}_regions.png")
    for label, image in images.items():
        image.crop(core_crop).save(folder / "raw_core" / f"{cid}_{label}.png")
    if "F" not in images:
        return
    reference = np.asarray(images["F"], dtype=np.float32) / 255
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), squeeze=False)
    view = None
    for axis, config in zip(axes[0], CONFIGS[1:]):
        if config in images:
            difference = np.abs(np.asarray(images[config], dtype=np.float32) / 255 - reference).mean(axis=-1)
            view = axis.imshow(difference, vmin=0, vmax=.08, cmap="magma")
            axis.set_title(f"{config}: RGB MAE to F")
        axis.axis("off")
    if view is not None:
        fig.colorbar(view, ax=list(axes[0]), fraction=.02, pad=.02)
    fig.savefig(folder / f"{cid}_error_maps.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_summary_plots(root, metrics, summaries):
    plt = plotting()
    folder = root / "figures/val16"
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharex=True, sharey=True)
    for axis, config in zip(axes, CONFIGS[1:]):
        selected = [row for row in metrics if row["config"] == config and row.get("quality_status") == "complete" and row.get("paired_geometric_latency_ratio_to_F") is not None]
        for row in selected:
            speed = 100 * (1 - row["paired_geometric_latency_ratio_to_F"])
            color = "tab:green" if row["predefined_quality_gate_pass"] else "tab:red"
            axis.scatter(speed, row["lpips_mapmean_to_F"], color=color, s=22)
            axis.annotate(row["case_id"].replace("mb_val_", ""), (speed, row["lpips_mapmean_to_F"]), fontsize=5, xytext=(2, 2), textcoords="offset points")
        axis.axhline(.02, color="black", ls="--", lw=.8)
        axis.axvline(10, color="black", ls="--", lw=.8)
        axis.set_title(config)
        axis.set_xlabel("Paired net latency reduction (%)")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Full RGB LPIPS spatial mean to F")
    fig.suptitle("One point per independent source; green passes both numeric quality gates")
    fig.tight_layout()
    fig.savefig(folder / "quality_latency_scatter.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["gray", "tab:blue", "tab:orange", "tab:green", "tab:red"]
    for position, config, color in zip(range(len(CONFIGS)), CONFIGS, colors):
        selected = [row for row in metrics if row["config"] == config and row.get("timing_complete")]
        axes[0].scatter([position] * len(selected), [row["e2e_median_seconds"] for row in selected], s=15, alpha=.55, color=color)
        value = summaries[config]["paired_net_latency_reduction"]
        if value:
            point, interval = 100 * value["point"], [100 * x for x in value["percentile95"]]
            axes[1].errorbar(position, point, yerr=[[point - interval[0]], [interval[1] - point]], fmt="o", capsize=4, color=color)
    axes[0].set_ylabel("Per-source end-to-end median (s)")
    axes[1].set_ylabel("Paired net latency reduction (%)")
    axes[1].axhline(10, color="black", ls="--", lw=.8)
    axes[1].set_title("Source bootstrap 95% percentile interval")
    for axis in axes:
        axis.set_xticks(range(len(CONFIGS)), CONFIGS)
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / "latency_sources_bootstrap.png", dpi=200)
    plt.close(fig)


def display(value):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "通过" if value else "未通过"
    return f"{value:.6g}" if isinstance(value, (float, int)) else str(value)


def write_md(root, cases, summary):
    metrics = summary["quality_rows"]
    lines = ["# 独立 Val16：冻结校准质量与净延迟数据摘要", "",
        f"汇总状态：**{summary['status']}**。成功请求 {summary['counts']['successful']}/320，预热 {summary['counts']['warmups']}/80，计时 {summary['counts']['measurements']}/240，记录 DiT {summary['counts']['dit_calls']}/2560。质量仅使用每源图每配置 repeat0，共16个独立源图；三个计时重复不是48个质量样本。", "",
        "| 配置 | 净节时 % [source bootstrap 95%] | 数值质量通过/16 [Wilson95%] | 均值 LPIPS/F | 最大 extraBG/F | 数值质量且节时≥10%/16 |",
        "|---|---|---|---:|---:|---:|"]
    for config in CONFIGS:
        row = summary["configs"][config]
        speed = row["paired_net_latency_reduction"]
        speed_text = "N/A" if speed is None else f"{speed['point'] * 100:.2f} [{speed['percentile95'][0] * 100:.2f}, {speed['percentile95'][1] * 100:.2f}]"
        ci = row["numeric_gate_pass_wilson95"]
        pass_text = f"{row['numeric_gate_pass_count']}/16 [{ci[0]:.3f}, {ci[1]:.3f}]"
        lpips = row["mean_lpips_to_F"]
        values = [config, speed_text, pass_text, lpips["point"] if lpips else None, row["max_extra_background_mae_over_F"], row["numeric_quality_and_10percent_speed_pass_count"]]
        lines.append("| " + " | ".join(display(value) for value in values) + " |")
    lines += ["", "节时先在同一源图、同一计时repeat内计算 T配置/TF，再对repeat取对数均值，最后以源图等权聚合。95%区间重采样源图10000次，seed=2026102304；只描述本次16源图的不确定性，不含生成种子、硬件和数据分布变化。F为计算参考，不能替代编辑正确性真值。", "",
        "数值门槛逐例固定为 LPIPS≤0.02 且 extraBG≤1/255。extraBG=MAE(候选,源图,远背景)−MAE(F,源图,远背景)，远背景为本次冻结R2的补集；其负值不表示指令执行正确。该区域与旧实验allowed-box定义不同，不能把数值直接拼接成同一序列。", "",
        "| 源图 | 配置 | 中位秒 | CV | 配对耗时比/F | LPIPS/F | band2 MAE/F | extraBG/F | 数值通过 |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in metrics:
        values = [row["case_id"], row["config"], row["e2e_median_seconds"], row["e2e_cv"], row["paired_geometric_latency_ratio_to_F"], row.get("lpips_mapmean_to_F"), row.get("band2_mae_to_F"), row.get("extra_background_mae_over_F"), row.get("predefined_quality_gate_pass")]
        lines.append("| " + " | ".join(display(value) for value in values) + " |")
    lines += ["", "R0为标注透明编辑区域的bbox投影到16px token，评价band1/2/4均由同一个R0内外膨胀交集得到。R2为R0加2-token halo。所有配置共享评价区；LPIPS先对未遮罩完整RGB图像计算空间距离，再分区平均，感受野可以跨越区域。", "",
        "| 校准对照 | LPIPS更小源图数 | LPIPS更大源图数 | 数值门槛新增通过 | 数值门槛丢失通过 | 平均LPIPS差 [95%] |", "|---|---:|---:|---:|---:|---|"]
    for name, row in summary["route_comparisons"].items():
        delta = row["mean_lpips_delta"]
        text = "N/A" if delta is None else f"{delta['point']:.6f} [{delta['percentile95'][0]:.6f}, {delta['percentile95'][1]:.6f}]"
        lines.append("| " + " | ".join(display(value) for value in [name, row["lpips_improved_cases"], row["lpips_worsened_cases"], row["numeric_pass_gains"], row["numeric_pass_losses"], text]) + " |")
    lines += ["", "候选−对照为负表示该指标更接近F，不自动代表语义编辑更好。该冻结验证不重新选择eta、系数、掩码或门槛；失败也保留，Test未参与本次实验。数字通过不自动宣布正式Go。实际助手视觉非盲复核见 [前8例](05_前8例助手视觉复核.md) 与 [后8例](06_后8例助手视觉复核.md)，不是人工盲评或真人用户研究。", "",
        "| 配置 | 静态校正表预备秒 | 表大小 MiB | 部署时更新系数 |", "|---|---:|---:|---|"]
    for config in ["PRIOR", "A", "B"]:
        table = summary["correction_table_preparation"].get(config, {})
        size = table.get("table_bytes")
        lines.append("| " + " | ".join(display(value) for value in [config, table.get("prepare_seconds"), size / 2**20 if size is not None else None, "否"]) + " |")
    lines += ["", "静态表冷启动耗时单独报告，未加入每请求耗时；运行时系数读取、Value变换、临时张量和缓存清理均包含于端到端计时。所有配置测量时校正表均已常驻显存。模型载入/LoRA融合、源图文件I/O、标注ROI提供、初始latent创建、PNG保存和离线质量指标不在本次请求计时范围内；该结果不是自动掩码检测系统的全链路耗时。", "",
        "| 审计 | 结果 |", "|---|---|"]
    for name in ["all_output_sha_passed", "all_repeats_deterministic", "all_latent_and_scheduler_pairs_match", "execution_order_matches"]:
        lines.append(f"| {name} | {display(summary[name])} |")
    lines.append(f"| runtime_age_QKV_MLP_correction_contract | {display(summary['all_runtime_audits_passed'])} |")
    if summary["warnings"]:
        lines += ["", "审计提示：", ""] + [f"- {value}" for value in summary["warnings"]]
    lines += ["", "图像链接（core/boundary crop保持输出像素尺度，不插值放大）：", ""]
    for case in cases:
        cid = case["case_id"]
        lines += [f"- {cid}: {case.get('prompt', case.get('instruction', ''))}",
            f"  [全图](figures/val16/{cid}_comparison.png) · [核心原像素裁剪](figures/val16/{cid}_core_crop.png) · [边界原像素裁剪](figures/val16/{cid}_boundary_crop.png) · [误差图](figures/val16/{cid}_error_maps.png) · [评价区域](figures/val16/{cid}_regions.png)"]
    lines += ["", "- [质量与净延迟散点](figures/val16/quality_latency_scatter.png)", "- [逐源图延迟与区间](figures/val16/latency_sources_bootstrap.png)", "- [完整质量CSV](data/quality_val16.csv)", "- [逐请求计时CSV](data/timings_val16.csv)", "- [完整JSON汇总](data/summary_val16.json)", ""]
    (root / "03_Val16数据摘要.md").write_text("\n".join(lines), encoding="utf-8")
    template_path = root / "04_人工非盲复核模板.md"
    if not template_path.exists():
        lines = ["# Val16 人工非盲复核模板（未填写）", "", "状态：未填写，仅供后续人工记录；不能作为已有评价证据。实际助手视觉非盲复核已另存 [前8例](05_前8例助手视觉复核.md) 与 [后8例](06_后8例助手视觉复核.md)，其记录者为助手，不能称为真人评价。本模板不代表任何语义通过。", "",
            "每例分别记录F指令执行结果，以及R2/PRIOR/A/B相对F是否漏编辑、内容错误、形状/身份退化、边界接缝、光照块、远背景异常。F本身失败必须单列，不能删除该源图。", ""]
        for case in cases:
            cid = case["case_id"]
            lines += [f"## {cid}", "", case.get("prompt", case.get("instruction", "")), "", f"[全图](figures/val16/{cid}_comparison.png) · [边界原像素](figures/val16/{cid}_boundary_crop.png)", "", "| 配置 | 指令执行/保留对象 | 边界/光照 | 远背景 | 结论与证据 |", "|---|---|---|---|---|"]
            lines += [f"| {config} | 待复核 | 待复核 | 待复核 | 待复核 |" for config in CONFIGS]
            lines += [""]
        template_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--previous-root", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--prepare-regions", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    cases_path = args.cases.resolve() if args.cases else root / "cases_val16.json"
    cases_document = read_json(cases_path)
    cases = cases_document["cases"]
    if len(cases) != 16 or len({case["case_id"] for case in cases}) != 16:
        raise ValueError("Frozen independent validation requires 16 distinct cases")
    if args.prepare_regions:
        print(json.dumps(prepare_regions(root, cases_path, cases), ensure_ascii=False, indent=2))
        return
    regions = load_regions(root, cases_path, cases)
    helpers, helper_path = load_helpers(root, args.previous_root)
    data, figures = root / "data", root / "figures/val16"
    (figures / "raw_core").mkdir(parents=True, exist_ok=True)
    runs_path = args.runs.resolve() if args.runs else data / "runs_primary.jsonl"
    environment_candidates = sorted(data.glob("environment_*.json"))
    environment_path = args.environment.resolve() if args.environment else (environment_candidates[-1] if environment_candidates else data / "environment_val16.json")
    environment = read_json(environment_path) if environment_path.exists() else {}
    preregistration_path = data / "preregistration.json"
    preregistration = read_json(preregistration_path) if preregistration_path.exists() else {}
    raw = [json.loads(line) for line in runs_path.read_text().splitlines() if line.strip()]
    relevant_raw = [row for row in raw if config_id(row) in CONFIGS]
    relevant = [row for row in relevant_raw if row.get("status") == "complete"]
    failures = [row for row in relevant_raw if row.get("status") != "complete"]
    if len({row["run_id"] for row in relevant}) != len(relevant):
        raise ValueError("Duplicate successful run_id; refusing to silently deduplicate")
    lookup = {}
    for row in relevant:
        if row.get("kind") != "measurement":
            continue
        key = row["case_id"], config_id(row), int(row["repeat"])
        if key in lookup:
            raise ValueError(f"Duplicate measurement key: {key}")
        lookup[key] = row
    case_ids = [case["case_id"] for case in cases]
    if any(row["case_id"] not in case_ids for row in relevant):
        raise ValueError("Unexpected source image in validation run ledger")
    warnings, output_audits, repeat_audits, pair_audits, runtime_audits = [], [], [], [], []
    preregistration_checks = dict(cases=preregistration.get("cases_sha256") == sha256(cases_path),
        evaluation_regions=preregistration.get("evaluation_regions_sha256") == sha256(root / "masks/evaluation_regions.json"),
        case_order=preregistration.get("case_ids") == case_ids, configs=preregistration.get("configs") == CONFIGS,
        no_fit=preregistration.get("fitting_on_val") is False, no_target=preregistration.get("target_column_used") is False,
        no_test=preregistration.get("test_used") is False, no_rl=preregistration.get("rl") is False,
        no_gradients=preregistration.get("new_gradient_steps") == 0,
        environment_preregistration=environment.get("preregistration_sha256") == sha256(preregistration_path) if preregistration_path.exists() else False)
    if not all(preregistration_checks.values()):
        warnings.append("Preregistration metadata or its file hashes do not match the frozen protocol")
    case_by_id = {case["case_id"]: case for case in cases}
    for row in relevant:
        audit = runtime_audit(row, case_by_id[row["case_id"]], preregistration.get("calibration_sha256"))
        runtime_audits.append(audit)
        if not audit["passed"]:
            warnings.append(f"Runtime age, shape, cleanup or correction contract mismatch: {row['run_id']}")
        path = resolve(root, row["output"])
        actual = sha256(path) if path.is_file() else None
        valid = actual is not None and actual == row.get("output_sha256")
        output_audits.append(dict(run_id=row["run_id"], matches=valid, actual_sha256=actual, recorded_sha256=row.get("output_sha256")))
        if not valid:
            warnings.append(f"Missing or modified output: {row['run_id']}")
        if row.get("dit_calls") != 8:
            warnings.append(f"Unexpected DiT count: {row['run_id']}")
        if row.get("model_timestep_contract_exact") is not True or row.get("observed_model_timesteps") != environment.get("expected_model_timesteps"):
            warnings.append(f"Model time conversion mismatch: {row['run_id']}")
    import lpips
    import torch
    torch.set_num_threads(8)
    network = lpips.LPIPS(net="alex", spatial=True).eval().to(args.device)
    metrics, timings = [], []
    for case in cases:
        cid = case["case_id"]
        height, width = case["target_hw"]
        source = Image.open(resolve(root, case["image"])).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        masks = evaluation_masks(root, regions["cases"][cid])
        f_rows = {repeat: lookup[(cid, "F", repeat)] for repeat in range(REPEATS) if (cid, "F", repeat) in lookup}
        f_zero = f_rows.get(0)
        reference = helpers.load_image(root, f_zero, (width, height)) if f_zero else None
        images = {"INPUT": source}
        for config in CONFIGS:
            selected = [lookup[(cid, config, repeat)] for repeat in range(REPEATS) if (cid, config, repeat) in lookup]
            result = dict(case_id=cid, img_id=case.get("img_id"), config=config, phase="independent_val16")
            result.update(helpers.timing_summary(selected, f_rows, expected_repeats=REPEATS, baseline_config="F"))
            group = [row for row in relevant if row["case_id"] == cid and config_id(row) == config]
            complete = len(group) == 4 and sum(row["kind"] == "warmup" for row in group) == 1 and result["timing_complete"]
            identical = len({row.get("output_sha256") for row in group}) == 1 if group else False
            repeat_audits.append(dict(case_id=cid, config=config, requests=len(group), complete=complete, output_sha_identical=identical))
            if not complete or not identical:
                warnings.append(f"Incomplete or nondeterministic warmup/measurement group: {cid}/{config}")
            for row in selected:
                ref = f_rows.get(int(row["repeat"]))
                matches = ref is not None and all(row.get(key) is not None and row.get(key) == ref.get(key) for key in ["initial_latent_sha256", "timesteps", "sigmas"])
                pair_audits.append(dict(run_id=row["run_id"], baseline_run_id=ref["run_id"] if ref else None, latent_scheduler_match=matches))
                if not matches:
                    warnings.append(f"Latent or scheduler differs from same-repeat F: {row['run_id']}")
                timings.append(helpers.timing_detail(row, f_rows, "independent_val16", baseline_config="F"))
            zero = lookup.get((cid, config, 0))
            if zero is None or reference is None:
                result.update(quality_status="missing_repeat0_or_F", predefined_quality_gate_pass=None)
            else:
                try:
                    if not all(zero.get(key) is not None and zero.get(key) == f_zero.get(key) for key in ["initial_latent_sha256", "timesteps", "sigmas"]):
                        raise ValueError("Initial latent or scheduler differs from same-repeat F")
                    output = helpers.load_image(root, zero, (width, height))
                    result.update(helpers.quality_metrics(output, reference, source, masks, network, args.device))
                    result.update(quality_output=zero["output"], quality_output_sha256=zero["output_sha256"], quality_run_id=zero["run_id"])
                    images[config] = output
                except Exception as error:
                    result.update(quality_status="invalid_output", quality_error=f"{type(error).__name__}: {error}", predefined_quality_gate_pass=None)
                    warnings.append(f"Quality failure: {cid}/{config}: {result['quality_error']}")
            metrics.append(result)
        make_case_panels(root, case, images, masks, helpers)
    counts = dict(raw_records=len(raw), successful=len(relevant), failed=len(failures),
        warmups=sum(row["kind"] == "warmup" for row in relevant), measurements=len(lookup),
        dit_calls=sum(row.get("dit_calls", 0) for row in relevant), expected_successful=320,
        expected_warmups=80, expected_measurements=240, expected_dit_calls=2560,
        unique_source_quality_images=len({row["case_id"] for row in metrics if row.get("quality_status") == "complete"}),
        unique_quality_outputs=sum(row.get("quality_status") == "complete" for row in metrics),
        failed_dit_calls_completed=sum(row.get("dit_calls_completed", 0) for row in failures),
        failed_dit_calls_invoked=sum(row.get("dit_calls_invoked", 0) for row in failures))
    complete = counts["successful"] == 320 and counts["warmups"] == 80 and counts["measurements"] == 240 and counts["dit_calls"] == 2560
    complete = complete and all(row["complete"] for row in repeat_audits) and all(row.get("quality_status") == "complete" for row in metrics)
    expected_execution = environment.get("expected_execution")
    if expected_execution is None:
        if preregistration_path.exists():
            expected_execution = preregistration.get("expected_execution")
            if expected_execution is None and "planned_execution" in preregistration:
                expected_execution = [[row["case_id"], row["config"], row["kind"], int(row["repeat"])] for row in preregistration["planned_execution"]]
    observed = [[row["case_id"], config_id(row), row["kind"], int(row["repeat"])] for row in relevant]
    order_matches = observed == expected_execution if expected_execution is not None else None
    if order_matches is not True:
        warnings.append("Execution order does not match supplied preregistration, or expected_execution not supplied")
    summaries = {config: config_summary(metrics, config, len(cases)) for config in CONFIGS}
    comparisons = route_comparisons(metrics, case_ids)
    failed_numeric = [row for row in metrics if row.get("predefined_quality_gate_pass") is not True]
    failed_numeric.sort(key=lambda row: row.get("lpips_mapmean_to_F", float("inf")), reverse=True)
    summary = dict(status="complete" if complete else "partial", status_scope="Completeness of requested matrix; not semantic acceptance or formal Go",
        counts=counts, configs=summaries, quality_rows=metrics, route_comparisons=comparisons,
        failed_records=failures, warnings=warnings, output_sha_audits=output_audits, repeated_output_audits=repeat_audits,
        all_output_sha_passed=len(output_audits) == 320 and all(row["matches"] for row in output_audits),
        all_repeats_deterministic=len(repeat_audits) == 80 and all(row["complete"] and row["output_sha_identical"] for row in repeat_audits),
        all_latent_and_scheduler_pairs_match=len(pair_audits) == 240 and all(row["latent_scheduler_match"] for row in pair_audits),
        preregistration_checks=preregistration_checks, all_preregistration_checks_passed=all(preregistration_checks.values()),
        runtime_audits=runtime_audits, all_runtime_audits_passed=len(runtime_audits) == 320 and all(row["passed"] for row in runtime_audits),
        correction_table_preparation=environment.get("tables", {}),
        latent_scheduler_pair_audits=pair_audits, execution_order_matches=order_matches,
        execution_order=dict(expected=expected_execution, observed=observed),
        numeric_failure_candidates=[dict(case_id=row["case_id"], config=row["config"], lpips_to_F=row.get("lpips_mapmean_to_F"),
            extra_background_mae_over_F=row.get("extra_background_mae_over_F"), gate_lpips=row.get("gate_lpips_le_0p02"),
            gate_extra_background=row.get("gate_extra_background_le_1over255")) for row in failed_numeric],
        protocol=dict(configs=CONFIGS, repeats=REPEATS, quality_repeat=0, quality_independent_sources=16,
            frozen_training=True, fitting_on_validation=False, extra_dense_oracles_in_performance=0,
            far_background="complement of frozen R2; not historical manual allowed-box",
            quality_gate=dict(lpips_mapmean_to_F_max=.02, extra_background_mae_over_F_max=1 / 255),
            latency="Each candidate paired with same case/repeat F; geometric over repeats then source-equal geometric over cases",
            bootstrap=dict(unit="source image", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, interval="95% percentile; descriptive"),
            gate_interval="Wilson95%, missing quality counts as non-pass; 16 frozen sources, not repeats",
            semantic="Separate nonblind visual review required; F is computational reference, not semantic ground truth",
            limitations="No Test generation, no seed-generalization claim, no online refit, no RL, no automatic noninferiority or formal Go"),
        provenance=dict(script_sha256=sha256(Path(__file__)), helper_script=str(helper_path), helper_sha256=sha256(helper_path),
            cases_sha256=sha256(cases_path), evaluation_regions_sha256=sha256(root / "masks/evaluation_regions.json"),
            preregistration_sha256=sha256(preregistration_path) if preregistration_path.exists() else None,
            runs_sha256=sha256(runs_path), environment_sha256=sha256(environment_path) if environment_path.exists() else None))
    save_csv(data / "quality_val16.csv", metrics)
    save_csv(data / "timings_val16.csv", timings)
    save_csv(data / "numeric_failure_candidates.csv", summary["numeric_failure_candidates"])
    save_json(data / "summary_val16.json", summary)
    metric_environment = dict(device=args.device, network="alex", spatial=True, normalize=True, rgb_range=[0, 1], packages={}, weights={})
    for name in ["lpips", "torch", "torchvision", "numpy", "scikit-image", "matplotlib", "Pillow"]:
        try:
            metric_environment["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            metric_environment["packages"][name] = None
    weight_paths = list(Path.home().joinpath(".cache/torch/hub/checkpoints").glob("alexnet*.pth"))
    weight_paths += list(Path(lpips.__file__).resolve().parent.glob("weights/v0.1/alex.pth"))
    metric_environment["weights"] = {str(path): sha256(path) for path in weight_paths}
    metric_environment.update(summary["provenance"])
    save_json(data / "metrics_environment_val16.json", metric_environment)
    make_summary_plots(root, metrics, summaries)
    write_md(root, cases, summary)
    print(json.dumps(dict(status=summary["status"], counts=counts, configs=summaries, warnings=warnings), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
