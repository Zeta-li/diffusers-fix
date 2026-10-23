import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity


PRIMARY_CONFIGS = ["F", "R0", "R1", "R2", "R3"]
LPIPS_GATE = 0.02
EXTRA_BACKGROUND_GATE = 1 / 255
EXPECTED_REPEATS = 5


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def config_id(row):
    config = row.get("config", "")
    return config.get("id", "") if isinstance(config, dict) else str(config)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dilate(mask, radius):
    h, w = mask.shape
    padded = np.pad(mask, radius, constant_values=False)
    result = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result |= padded[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
    return result


def upsample_mask(mask, height, width):
    assert height % mask.shape[0] == 0 and width % mask.shape[1] == 0
    return np.repeat(np.repeat(mask, height // mask.shape[0], axis=0), width // mask.shape[1], axis=1)


def evaluation_masks(case_data, root, height, width):
    core_grid = np.load(root / case_data["regions"]["R0"]["mask_npy"], allow_pickle=False).astype(bool)
    assert list(core_grid.shape) == case_data["target_grid_hw"]
    masks = {"core": upsample_mask(core_grid, height, width)}
    for radius in [1, 2, 4]:
        band = dilate(core_grid, radius) & dilate(~core_grid, radius)
        masks[f"band{radius}"] = upsample_mask(band, height, width)
    allowed = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = case_data["old_allowed_box_diagnostic_only"]
    allowed[round(y0 * height):round(y1 * height), round(x0 * width):round(x1 * width)] = True
    masks["far_background"] = ~allowed
    return masks


def mean_in(values, mask):
    return float(values[mask].mean()) if mask.any() else None


def gradient_mae(first, second, mask):
    horizontal_mask = mask[:, 1:] & mask[:, :-1]
    vertical_mask = mask[1:] & mask[:-1]
    horizontal = np.abs(np.diff(first, axis=1) - np.diff(second, axis=1))
    vertical = np.abs(np.diff(first, axis=0) - np.diff(second, axis=0))
    denominator = int(horizontal_mask.sum() + vertical_mask.sum()) * first.shape[2]
    if denominator == 0:
        return None
    numerator = float(horizontal[horizontal_mask].sum()) + float(vertical[vertical_mask].sum())
    return numerator / denominator


def as_tensor(image, device):
    array = np.asarray(image, dtype=np.float32) / 255
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)


def lpips_map(network, first, second):
    with torch.inference_mode():
        distance = network(first, second, normalize=True)
    result = distance[0, 0].float().cpu().numpy()
    assert result.shape == tuple(first.shape[-2:])
    return result


def timing_summary(rows, f_by_repeat, expected_repeats=EXPECTED_REPEATS, baseline_config="F"):
    values = [float(row["e2e_seconds"]) for row in rows]
    ratios = [float(row["e2e_seconds"]) / float(f_by_repeat[int(row["repeat"])]["e2e_seconds"]) for row in rows if int(row["repeat"]) in f_by_repeat]
    stage = [row.get("cuda_event_seconds", {}).get("dit", 0) / row["e2e_seconds"] for row in rows]
    result = {
        "baseline_config": baseline_config,
        "timing_n": len(values),
        "expected_repeats": expected_repeats,
        "timing_complete": expected_repeats is not None and len(values) == expected_repeats and {int(row["repeat"]) for row in rows} == set(range(expected_repeats)),
        "e2e_median_seconds": statistics.median(values) if values else None,
        "e2e_mean_seconds": statistics.mean(values) if values else None,
        "e2e_cv": statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else None,
        "paired_F_n": len(ratios),
        "paired_median_latency_ratio_to_F": statistics.median(ratios) if ratios else None,
        "paired_geometric_latency_ratio_to_F": math.exp(statistics.mean(math.log(value) for value in ratios)) if ratios else None,
        "paired_median_speedup_over_F": statistics.median(1 / value for value in ratios) if ratios else None,
        "peak_allocated_GiB": max((row.get("peak_allocated_bytes", 0) / 2**30 for row in rows), default=None),
        "peak_reserved_GiB": max((row.get("peak_reserved_bytes", 0) / 2**30 for row in rows), default=None),
        "max_cache_GiB": max((row.get("cache_bytes", 0) / 2**30 for row in rows), default=None),
        "mean_DiT_fraction": statistics.mean(stage) if stage else None,
        "distinct_output_hashes": len({row.get("output_sha256") for row in rows}),
    }
    result.update(paired_baseline_n=len(ratios), paired_median_latency_ratio_to_baseline=result["paired_median_latency_ratio_to_F"],
        paired_geometric_latency_ratio_to_baseline=result["paired_geometric_latency_ratio_to_F"], paired_median_speedup_over_baseline=result["paired_median_speedup_over_F"])
    if baseline_config != "F":
        for key in ["paired_F_n", "paired_median_latency_ratio_to_F", "paired_geometric_latency_ratio_to_F", "paired_median_speedup_over_F"]:
            del result[key]
    result["timing_pairing_status"] = "complete" if len(ratios) == len(values) and values else "missing_baseline_or_measurements"
    return result


def timing_detail(row, f_by_repeat, group, baseline_config="F"):
    reference = f_by_repeat.get(int(row["repeat"]))
    stages = row.get("cuda_event_seconds", {})
    result = {
        "analysis_group": group,
        "baseline_config": baseline_config,
        "case_id": row["case_id"],
        "config": config_id(row),
        "repeat": int(row["repeat"]),
        "run_id": row["run_id"],
        "attempt": row.get("attempt"),
        "e2e_seconds": row["e2e_seconds"],
        "paired_F_run_id": reference["run_id"] if reference else None,
        "paired_latency_ratio_to_F": row["e2e_seconds"] / reference["e2e_seconds"] if reference else None,
        "dit_seconds": stages.get("dit"),
        "prompt_encode_seconds": stages.get("prompt_encode"),
        "vae_encode_seconds": stages.get("vae_encode"),
        "vae_decode_seconds": stages.get("vae_decode"),
        "scheduler_seconds": stages.get("scheduler"),
        "dit_calls": row.get("dit_calls"),
        "peak_allocated_GiB": row.get("peak_allocated_bytes", 0) / 2**30,
        "peak_reserved_GiB": row.get("peak_reserved_bytes", 0) / 2**30,
        "cache_GiB": row.get("cache_bytes", 0) / 2**30,
        "initial_latent_sha256": row.get("initial_latent_sha256"),
        "output_sha256": row.get("output_sha256"),
        "output": row.get("output"),
    }
    result.update(paired_baseline_run_id=reference["run_id"] if reference else None,
        paired_latency_ratio_to_baseline=row["e2e_seconds"] / reference["e2e_seconds"] if reference else None,
        timing_pairing_status="complete" if reference else "missing_baseline")
    if baseline_config != "F":
        del result["paired_F_run_id"]
        del result["paired_latency_ratio_to_F"]
    return result


def load_image(root, row, expected_size=None):
    path = root / row["output"]
    if row.get("output_sha256"):
        assert sha256(path) == row["output_sha256"], f"Output hash mismatch: {row['run_id']}"
    image = Image.open(path).convert("RGB")
    if expected_size is not None:
        assert image.size == expected_size, f"Unexpected output geometry: {row['run_id']}"
    return image


def quality_metrics(output, reference, source, masks, network, device):
    array = np.asarray(output, dtype=np.float32) / 255
    ref = np.asarray(reference, dtype=np.float32) / 255
    original = np.asarray(source, dtype=np.float32) / 255
    current_tensor = as_tensor(output, device)
    reference_tensor = as_tensor(reference, device)
    distance = lpips_map(network, current_tensor, reference_tensor)
    absolute_ref = np.abs(array - ref).mean(axis=2)
    absolute_source = np.abs(array - original).mean(axis=2)
    reference_source = np.abs(ref - original).mean(axis=2)
    mse = float(np.mean((array - ref) ** 2))
    far_current = mean_in(absolute_source, masks["far_background"])
    far_reference = mean_in(reference_source, masks["far_background"])
    extra_background = far_current - far_reference if far_current is not None and far_reference is not None else None
    result = {
        "quality_status": "complete",
        "mae_to_F": float(absolute_ref.mean()),
        "psnr_to_F_db": -10 * math.log10(mse) if mse > 0 else None,
        "psnr_infinite_identical_to_F": mse == 0,
        "ssim_to_F": float(structural_similarity(ref, array, channel_axis=-1, data_range=1)),
        "lpips_mapmean_to_F": float(distance.mean()),
        "far_background_mae_to_source": far_current,
        "F_far_background_mae_to_source": far_reference,
        "extra_background_mae_over_F": extra_background,
    }
    for name in ["core", "band1", "band2", "band4"]:
        result[f"{name}_area_fraction"] = float(masks[name].mean())
        result[f"{name}_mae_to_F"] = mean_in(absolute_ref, masks[name])
        result[f"{name}_lpips_mapmean_to_F"] = mean_in(distance, masks[name])
    result["band2_gradient_mae_to_F"] = gradient_mae(array, ref, masks["band2"])
    result["far_background_area_fraction"] = float(masks["far_background"].mean())
    result["gate_lpips_le_0p02"] = result["lpips_mapmean_to_F"] <= LPIPS_GATE
    result["gate_extra_background_le_1over255"] = extra_background <= EXTRA_BACKGROUND_GATE if extra_background is not None else None
    result["predefined_quality_gate_pass"] = bool(result["gate_lpips_le_0p02"] and result["gate_extra_background_le_1over255"]) if extra_background is not None else None
    return result


def quality_missing(status, error=None):
    return {"quality_status": status, "quality_error": error, "predefined_quality_gate_pass": None}


def comparison_panel(images, labels, path, crop=None, tile_width=512):
    size = images[0].size
    height = round(size[1] * tile_width / size[0])
    if crop is not None:
        width, height = crop[2] - crop[0], crop[3] - crop[1]
    else:
        width = tile_width
    header = 34
    canvas = Image.new("RGB", (len(images) * width, height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        draw.text((index * width + 8, 9), label, fill="black")
        if image is None:
            draw.text((index * width + 8, header + 12), "MISSING", fill="red")
            continue
        displayed = image.crop(crop) if crop is not None else image.resize((width, height), Image.Resampling.LANCZOS)
        canvas.paste(displayed, (index * width, header))
    canvas.save(path)


def region_visualization(source, masks, path):
    display = np.asarray(source).astype(np.float32)
    for name, color, strength in [("far_background", [50, 180, 60], 0.20), ("core", [30, 135, 255], 0.25), ("band2", [255, 130, 20], 0.45)]:
        display[masks[name]] = display[masks[name]] * (1 - strength) + np.array(color) * strength
    image = Image.fromarray(np.uint8(np.clip(display, 0, 255)))
    image.thumbnail((1100, 1100))
    panel = Image.new("RGB", (image.width, image.height + 34), "white")
    panel.paste(image, (0, 34))
    ImageDraw.Draw(panel).text((8, 9), "Common evaluation masks: blue core | orange band2 | green far background", fill="black")
    panel.save(path)


def metric_environment(lpips_module, device, root):
    result = {"device": device, "network": "alex", "spatial": True, "normalize": True, "rgb_range": [0, 1], "packages": {}, "weights": {}, "script_sha256": sha256(Path(__file__)), "regions_sha256": sha256(root / "masks/regions.json")}
    for name in ["lpips", "torch", "torchvision", "numpy", "scikit-image", "matplotlib", "Pillow"]:
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    paths = list(Path.home().joinpath(".cache/torch/hub/checkpoints").glob("alexnet*.pth"))
    paths += list(Path(lpips_module.__file__).resolve().parent.glob("weights/v0.1/alex.pth"))
    for path in paths:
        result["weights"][str(path)] = sha256(path)
    return result


def quality_comparison(rows, config2="R2", config3="R3"):
    lookup = {(row["case_id"], row["config"]): row for row in rows}
    case_ids = list(dict.fromkeys(row["case_id"] for row in rows))
    comparison = []
    for cid in case_ids:
        first, second = lookup.get((cid, config2)), lookup.get((cid, config3))
        row = {"case_id": cid, "structure_config": config2, "equal_budget_control": config3}
        if not first or not second or first.get("quality_status") != "complete" or second.get("quality_status") != "complete":
            row["status"] = "incomplete"
            comparison.append(row)
            continue
        a, b = first["band2_mae_to_F"], second["band2_mae_to_F"]
        ga, gb = first["band2_gradient_mae_to_F"], second["band2_gradient_mae_to_F"]
        row.update(status="complete", R2_band2_mae=a, R3_band2_mae=b, band2_mae_improvement_absolute=b - a, band2_mae_improvement_relative=(b - a) / b if b else None,
            R2_band2_gradient_mae=ga, R3_band2_gradient_mae=gb, band2_gradient_improvement_absolute=gb - ga if ga is not None and gb is not None else None,
            R2_predefined_quality_gate_pass=first.get("predefined_quality_gate_pass"), R3_predefined_quality_gate_pass=second.get("predefined_quality_gate_pass"))
        comparison.append(row)
    return comparison


def config_overview(rows, configs, cases, baseline_config="F", expected_repeats=EXPECTED_REPEATS):
    result = {}
    for config in configs:
        selected = [row for row in rows if row["config"] == config]
        ratios = [row["paired_geometric_latency_ratio_to_baseline"] for row in selected if row.get("paired_geometric_latency_ratio_to_baseline") is not None]
        quality = [row for row in selected if row.get("quality_status") == "complete"]
        complete = len(quality) == len(cases)
        ratio = math.exp(statistics.mean(math.log(value) for value in ratios)) if ratios else None
        all_repeats = len(selected) == len(cases) and all(row["timing_complete"] for row in selected)
        result[config] = {
            "baseline_config": baseline_config,
            "expected_repeats": expected_repeats,
            "case_medians_seconds": {row["case_id"]: row["e2e_median_seconds"] for row in selected},
            "case_CVs": {row["case_id"]: row["e2e_cv"] for row in selected},
            "paired_geometric_latency_ratio_to_baseline_across_cases": ratio,
            "all_cases_have_expected_repeats": all_repeats,
            "all_timing_pairs_present": bool(selected) and all(row["timing_pairing_status"] == "complete" for row in selected),
            "quality_cases_complete": len(quality),
            "gate_pass_cases": [row["case_id"] for row in quality if row.get("predefined_quality_gate_pass")],
            "gate_pass_all_cases": all(row.get("predefined_quality_gate_pass") for row in quality) if complete else None,
            "mean_lpips_mapmean_to_F": statistics.mean(row["lpips_mapmean_to_F"] for row in quality) if quality else None,
            "max_extra_background_mae_over_F": max((row["extra_background_mae_over_F"] for row in quality if row.get("extra_background_mae_over_F") is not None), default=None),
            "max_allocated_GiB": max((row["peak_allocated_GiB"] for row in selected if row.get("peak_allocated_GiB") is not None), default=None),
        }
        if baseline_config == "F":
            result[config]["paired_geometric_latency_ratio_to_F_across_cases"] = ratio
            result[config]["all_cases_have_five_repeats"] = all_repeats
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    root = args.root.resolve()
    data, figures = root / "data", root / "figures"
    data.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    manifest = json.loads((root / "masks/regions.json").read_text())
    case_file = root / "cases.json"
    cases = json.loads(case_file.read_text())["cases"] if case_file.exists() else [{"case_id": key, "image": value["source_image"]} for key, value in manifest["cases"].items()]
    records = [json.loads(line) for line in (data / "runs.jsonl").read_text().splitlines() if line.strip()]
    successful = {}
    duplicate_success = []
    for row in records:
        if row.get("status") != "complete":
            continue
        if row["run_id"] in successful:
            duplicate_success.append(row["run_id"])
        successful[row["run_id"]] = row
    measurements = [row for row in successful.values() if row.get("kind") == "measurement"]
    primary = [row for row in measurements if config_id(row) in PRIMARY_CONFIGS]
    fallback = [row for row in measurements if config_id(row).endswith("_safe")]
    primary_lookup = {(row["case_id"], config_id(row), int(row["repeat"])): row for row in primary}
    fallback_lookup = {(row["case_id"], config_id(row), int(row["repeat"])): row for row in fallback}
    fallback_plan_path = data / "fallback_plan.json"
    fallback_plan = json.loads(fallback_plan_path.read_text()) if fallback_plan_path.exists() else None
    fallback_repeats = int(fallback_plan["repeats"]) if fallback_plan is not None else None
    if fallback_repeats is not None:
        assert fallback_repeats > 0
    safe_configs = list(fallback_plan.get("configs", [])) if fallback_plan else []
    safe_configs += sorted({config_id(row) for row in fallback} - set(safe_configs))
    warnings = []
    if fallback and fallback_plan is None:
        warnings.append("Fallback measurements exist but fallback_plan.json is missing; expected repeat count is unknown, not assumed to equal the primary five.")
    primary_rows, fallback_rows, timing_rows, fallback_timing_rows = [], [], [], []
    fallback_reference_checks = []
    import lpips
    network = lpips.LPIPS(net="alex", spatial=True).eval().to(args.device)
    for case in cases:
        cid = case["case_id"]
        case_data = manifest["cases"][cid]
        target_h, target_w = case_data["target_hw"]
        source = Image.open(root / case.get("image", case_data["source_image"])).convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
        masks = evaluation_masks(case_data, root, target_h, target_w)
        f_rows = {repeat: row for (key, config, repeat), row in primary_lookup.items() if key == cid and config == "F"}
        f_zero = f_rows.get(0)
        reference = None
        if f_zero:
            try:
                reference = load_image(root, f_zero, (target_w, target_h))
            except Exception as error:
                warnings.append(f"{cid}: invalid F repeat0 reference: {type(error).__name__}: {error}")
        f_safe_rows = {repeat: row for (key, config, repeat), row in fallback_lookup.items() if key == cid and config == "F_safe"}
        f_safe_zero = f_safe_rows.get(0)
        check = {"case_id": cid, "primary_reference_config": "F", "same_round_reference_config": "F_safe", "repeat": 0}
        if not f_safe_zero or reference is None:
            check.update(status="missing_reference", primary_F_available=reference is not None, F_safe_available=f_safe_zero is not None)
        else:
            try:
                f_safe_image = load_image(root, f_safe_zero, reference.size)
                difference = np.abs(np.asarray(f_safe_image, dtype=np.float32) - np.asarray(reference, dtype=np.float32)) / 255
                check.update(status="complete", primary_F_run_id=f_zero["run_id"], F_safe_run_id=f_safe_zero["run_id"],
                    png_sha256_equal=f_safe_zero.get("output_sha256") == f_zero.get("output_sha256"),
                    pixel_identical=bool(np.array_equal(np.asarray(f_safe_image), np.asarray(reference))), mae=float(difference.mean()), max_channel_error=float(difference.max()))
                if not check["pixel_identical"]:
                    warnings.append(f"{cid}: F_safe repeat0 differs in pixels from primary F; safe quality remains relative to primary F and this baseline discrepancy must be reported.")
            except Exception as error:
                check.update(status="invalid_reference", error=f"{type(error).__name__}: {error}")
        if safe_configs:
            fallback_reference_checks.append(check)
        if any(row["case_id"] == cid for row in fallback) and not f_safe_rows:
            warnings.append(f"{cid}: fallback measurements have no F_safe baseline; safe timing ratios are missing, never substituted with old F.")
        images = {"INPUT": source}
        for group, configs, lookup, output_rows, detail_rows, pairing, baseline_config, expected_repeats in [
            ("primary", PRIMARY_CONFIGS, primary_lookup, primary_rows, timing_rows, f_rows, "F", EXPECTED_REPEATS),
            ("fallback_safe", safe_configs, fallback_lookup, fallback_rows, fallback_timing_rows, f_safe_rows, "F_safe", fallback_repeats),
        ]:
            for config in configs:
                selected = sorted([row for (key, key_config, repeat), row in lookup.items() if key == cid and key_config == config], key=lambda row: int(row["repeat"]))
                if group == "fallback_safe" and not selected:
                    continue
                row = {"analysis_group": group, "case_id": cid, "config": config}
                row.update(timing_summary(selected, pairing, expected_repeats=expected_repeats, baseline_config=baseline_config))
                region_id = config.removesuffix("_safe")
                n_total = case_data["n_target_tokens"]
                count = n_total if region_id == "F" else case_data["regions"][region_id]["n_active_target_tokens"]
                row.update(n_active_partial_target_tokens=count, n_target_tokens=n_total, partial_target_fraction=count / n_total)
                measurement_zero = lookup.get((cid, config, 0))
                if reference is None:
                    row.update(quality_missing("missing_or_invalid_F_repeat0"))
                elif measurement_zero is None:
                    row.update(quality_missing("missing_repeat0"))
                else:
                    try:
                        assert measurement_zero.get("initial_latent_sha256") == f_zero.get("initial_latent_sha256"), "Initial latent mismatch"
                        assert measurement_zero.get("timesteps") == f_zero.get("timesteps"), "Timesteps differ from F"
                        assert measurement_zero.get("sigmas") == f_zero.get("sigmas"), "Sigmas differ from F"
                        output = load_image(root, measurement_zero, reference.size)
                        row.update(quality_metrics(output, reference, source, masks, network, args.device))
                        row["quality_run_id"] = measurement_zero["run_id"]
                        row["quality_output"] = measurement_zero["output"]
                        row["quality_output_sha256"] = measurement_zero.get("output_sha256")
                        images[config] = output
                    except Exception as error:
                        message = f"{type(error).__name__}: {error}"
                        row.update(quality_missing("invalid_repeat0", message))
                        warnings.append(f"{cid}/{config}: {message}")
                output_rows.append(row)
                detail_rows.extend(timing_detail(item, pairing, group, baseline_config=baseline_config) for item in selected)
        labels = ["INPUT"] + PRIMARY_CONFIGS
        panel_images = [images.get(label) for label in labels]
        comparison_panel(panel_images, labels, figures / f"{cid}_comparison.png")
        ys, xs = np.nonzero(masks["core"])
        crop_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        comparison_panel(panel_images, labels, figures / f"{cid}_core_crop.png", crop=crop_box)
        region_visualization(source, masks, figures / f"{cid}_evaluation_masks.png")
        if safe_configs:
            safe_labels = ["INPUT", "F"] + [name for name in safe_configs if name in images]
            comparison_panel([images.get(name) for name in safe_labels], safe_labels, figures / f"{cid}_fallback_comparison.png")
    assert len(primary_rows) == len(cases) * len(PRIMARY_CONFIGS)
    save_csv(data / "quality_metrics.csv", primary_rows)
    save_csv(data / "timings.csv", timing_rows)
    if fallback_rows:
        save_csv(data / "fallback_quality_metrics.csv", fallback_rows)
        save_csv(data / "fallback_timings.csv", fallback_timing_rows)
    failed = [row for row in records if row.get("status") == "failed"]
    order_by_cycle = []
    for case in cases:
        for repeat in range(EXPECTED_REPEATS):
            actual = [config_id(row) for row in primary if row["case_id"] == case["case_id"] and int(row["repeat"]) == repeat]
            expected = PRIMARY_CONFIGS[repeat % len(PRIMARY_CONFIGS):] + PRIMARY_CONFIGS[:repeat % len(PRIMARY_CONFIGS)]
            order_by_cycle.append({"case_id": case["case_id"], "repeat": repeat, "expected_order": expected, "observed_log_order": actual, "order_matches": actual == expected})
    summary = {
        "status": "complete" if len(primary_rows) == 15 and all(row["timing_complete"] and row["quality_status"] == "complete" for row in primary_rows) else "partial",
        "counts": {"raw_records": len(records), "failed_records": len(failed), "successful_unique_runs": len(successful), "primary_measurement_runs": len(primary), "expected_primary_measurements": len(cases) * len(PRIMARY_CONFIGS) * EXPECTED_REPEATS, "fallback_measurement_runs": len(fallback), "nonmeasurement_successful_runs": sum(row.get("kind") != "measurement" for row in successful.values()), "primary_quality_rows": len(primary_rows), "primary_quality_rows_complete": sum(row["quality_status"] == "complete" for row in primary_rows)},
        "failed_runs": failed,
        "duplicate_success_run_ids": duplicate_success,
        "warnings": warnings,
        "measurement_repeat_order": order_by_cycle,
        "timing_protocol": "Five repeats per case/config; pair each local request with F from the same case and repeat cycle. Report median, sample CV, median paired ratio and geometric paired ratio. Cyclic order is audited. Warmup, correctness, reset and shape-audit runs are excluded. No hypothesis test, CI, P95 or independent-seed quality claim.",
        "quality_protocol": "Only measurement repeat0 per case/config. Full-image LPIPS Alex spatial map computed from unmasked RGB in [0,1], then average globally or over common R0/bands. Spatial LPIPS receptive fields cross region boundaries. Source uses LANCZOS resize. F is a computational reference, not editing ground truth.",
        "common_boundary_protocol": "For k=1/2/4 target tokens, band=dilate_Linf(R0,k) AND dilate_Linf(NOT R0,k), clipped to the finite target grid; combines inside and outside around the same frozen core for every config. No artificial outside-image boundary. One token is 16x16 pixels here. Bands are cumulative and overlap.",
        "gradient_protocol": "Band2 RGB finite-difference MAE to F; include horizontal/vertical edges only when both endpoints lie in the common band. No masked-image black borders.",
        "background_protocol": "Far background is the complement of the frozen previous allowed_box, with round(normalized coordinate * pixel dimension), identical for all configs. ExtraBG=MAE(config,source,far)-MAE(F,source,far), possibly negative; it is not MAE(config,F,far). Coarse boxes are diagnostics rather than validated preservation truth.",
        "predefined_gate": {"lpips_mapmean_to_F_max": LPIPS_GATE, "extra_background_mae_over_F_max": EXTRA_BACKGROUND_GATE, "operator": "AND, per case", "frozen_before_analysis": True, "meaning": "Provisional retention relative to F, not semantic correctness, noninferiority or visual acceptance. Missing or failed quality is not a pass."},
        "configs": config_overview(primary_rows, PRIMARY_CONFIGS, cases),
        "R2_vs_R3_common_band2": quality_comparison(primary_rows),
        "fallback": {"separate_from_primary": True, "plan": fallback_plan, "expected_repeats": fallback_repeats, "timing_baseline_config": "F_safe", "quality_reference_config": "primary F", "F_safe_repeat0_vs_primary_F": fallback_reference_checks,
            "configs": config_overview(fallback_rows, safe_configs, cases, baseline_config="F_safe", expected_repeats=fallback_repeats),
            "R2_vs_R3_common_band2": quality_comparison(fallback_rows, "R2_safe", "R3_safe") if fallback_rows else [],
            "limitations": "Fallback is an extra three-repeat diagnostic with its own native F_safe baseline in the same round. Pair timing only to F_safe from the same case and repeat; missing F_safe remains missing. Quality is relative to primary F, with F_safe repeat0 pixel/PNG hash agreement checked separately. Never mix these three repeats with the primary five, and never overwrite primary [1,8] full-refresh results."},
    }
    correctness_path = data / "real_correctness.json"
    if correctness_path.exists():
        correctness = json.loads(correctness_path.read_text())
        summary["correctness_status"] = correctness.get("status")
    write_json(data / "summary.json", summary)
    write_json(data / "metrics_environment.json", metric_environment(lpips, args.device, root))
    fig, ax = plt.subplots(figsize=(9, 4.3))
    width = 0.23
    lookup_rows = {(row["case_id"], row["config"]): row for row in primary_rows}
    for index, case in enumerate(cases):
        values = [lookup_rows[(case["case_id"], config)]["e2e_median_seconds"] for config in PRIMARY_CONFIGS]
        values = [value if value is not None else float("nan") for value in values]
        positions = np.arange(len(PRIMARY_CONFIGS)) + (index - (len(cases) - 1) / 2) * width
        ax.bar(positions, values, width=width, label=case["case_id"])
        for position, config in zip(positions, PRIMARY_CONFIGS):
            samples = [row["e2e_seconds"] for row in primary if row["case_id"] == case["case_id"] and config_id(row) == config]
            ax.scatter([position] * len(samples), samples, color="black", s=9, alpha=0.55, zorder=3)
    ax.set_xticks(np.arange(len(PRIMARY_CONFIGS)), PRIMARY_CONFIGS)
    ax.set_ylabel("End-to-end seconds (median of 5; dots = repeats)")
    ax.set_title("Primary local pilot; fresh request cache; ROI supplied")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "latency.png", dpi=190)
    plt.close(fig)
    print(json.dumps({"status": summary["status"], "counts": summary["counts"], "configs": summary["configs"], "R2_vs_R3_common_band2": summary["R2_vs_R3_common_band2"], "warnings": warnings}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
