import csv
import hashlib
import json
import importlib.metadata
import math
import statistics
from collections import defaultdict
from pathlib import Path

import lpips
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
torch.set_num_threads(8)
assert json.loads((ROOT / "status.json").read_text())["phase"] == "complete"
cases = json.loads((ROOT / "cases.json").read_text())["cases"]
annotations = json.loads((ROOT / "roi_annotations.json").read_text())
runs = [json.loads(line) for line in (DATA / "runs.jsonl").read_text().splitlines()]
assert len(runs) == 26 and all(row["status"] == "complete" for row in runs)
measurements = [row for row in runs if row["kind"] == "measurement"]
assert len(measurements) == 15
config_ids = list(dict.fromkeys(row["config"]["id"] for row in measurements))
lookup = {(row["case_id"], row["config"]["id"]): row for row in measurements}
for case in cases:
    hashes = {lookup[(case["case_id"], config)]["initial_latent_sha256"] for config in config_ids}
    assert len(hashes) == 1


def save_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mask_from_box(box, height, width):
    x0, y0, x1, y1 = box
    mask = np.zeros((height, width), dtype=bool)
    mask[round(y0 * height):round(y1 * height), round(x0 * width):round(x1 * width)] = True
    return mask


def tensor(image):
    array = np.asarray(image).astype(np.float32) / 255
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to("cuda")


def spatial_distance(network, first, second):
    with torch.inference_mode():
        return network(first, second, normalize=True)[0, 0].float().cpu().numpy()


network = lpips.LPIPS(net="alex", spatial=True).eval().to("cuda")
metric_rows = []
fusion_rows = []
for case in cases:
    case_id = case["case_id"]
    reference = Image.open(ROOT / lookup[(case_id, config_ids[0])]["output"]).convert("RGB")
    source = Image.open(ROOT / case["image"]).convert("RGB").resize(reference.size, Image.Resampling.LANCZOS)
    ref_array = np.asarray(reference).astype(np.float32) / 255
    source_array = np.asarray(source).astype(np.float32) / 255
    source_tensor = tensor(source)
    reference_tensor = tensor(reference)
    height, width = source_array.shape[:2]
    core = mask_from_box(annotations["cases"][case_id]["core_box"], height, width)
    allowed = mask_from_box(annotations["cases"][case_id]["allowed_box"], height, width)
    regions = {"core": core, "far_background": ~allowed}
    for radius in [16, 32, 64]:
        pixels = max(1, round(radius * min(height, width) / 1024))
        regions[f"ring{radius}"] = binary_dilation(core, iterations=pixels) & ~core
    panel_images = [source]
    panel_titles = ["INPUT"]
    outputs = {}
    for config_id in config_ids:
        row = lookup[(case_id, config_id)]
        output = Image.open(ROOT / row["output"]).convert("RGB")
        outputs[config_id] = np.asarray(output)
        array = np.asarray(output).astype(np.float32) / 255
        current = tensor(output)
        distance_reference = spatial_distance(network, current, reference_tensor)
        distance_source = spatial_distance(network, current, source_tensor)
        absolute_ref = np.abs(array - ref_array).mean(axis=2)
        absolute_source = np.abs(array - source_array).mean(axis=2)
        mse = float(np.mean((array - ref_array) ** 2))
        record = dict(
            case_id=case_id,
            config=config_id,
            mae_to_reference=float(absolute_ref.mean()),
            psnr_to_reference_db=-10 * math.log10(mse) if mse else None,
            ssim_to_reference=float(structural_similarity(ref_array, array, channel_axis=-1, data_range=1)),
            lpips_spatial_mean_to_reference=float(distance_reference.mean()),
            mae_to_source=float(absolute_source.mean()),
            lpips_spatial_mean_to_source=float(distance_source.mean()),
            changed_pixel_fraction_gt_0p03=float((absolute_source > 0.03).mean()),
        )
        energy = float(absolute_source.sum())
        for name, mask in regions.items():
            record[f"{name}_area_fraction"] = float(mask.mean())
            record[f"{name}_mae_to_reference"] = float(absolute_ref[mask].mean()) if mask.any() else None
            record[f"{name}_mae_to_source"] = float(absolute_source[mask].mean()) if mask.any() else None
            record[f"{name}_lpips_to_reference"] = float(distance_reference[mask].mean()) if mask.any() else None
            record[f"{name}_source_change_energy_fraction"] = float(absolute_source[mask].sum()) / energy if energy else 0
        metric_rows.append(record)
        panel_images.append(output)
        panel_titles.append(config_id.split("_")[0])
    unfused, fused = outputs[config_ids[3]], outputs[config_ids[4]]
    fusion_diff = np.abs(unfused.astype(np.float32) - fused.astype(np.float32)) / 255
    fusion_rows.append(dict(case_id=case_id, mae=float(fusion_diff.mean()), max_channel_error=float(fusion_diff.max()), identical_pixels_fraction=float((fusion_diff.max(axis=2) == 0).mean())))
    panel = Image.new("RGB", (384 * len(panel_images), 440), "white")
    draw = ImageDraw.Draw(panel)
    for i, (image, title) in enumerate(zip(panel_images, panel_titles)):
        image = image.copy()
        image.thumbnail((376, 398))
        panel.paste(image, (i * 384 + (384 - image.width) // 2, 32))
        draw.text((i * 384 + 10, 10), title, fill="black")
    panel.save(FIGURES / f"{case_id}_comparison.png")
    box = annotations["cases"][case_id]["core_box"]
    crop_box = (round(box[0] * width), round(box[1] * height), round(box[2] * width), round(box[3] * height))
    crops = Image.new("RGB", (600 * 4, 490), "white")
    draw = ImageDraw.Draw(crops)
    for i, index in enumerate([0, 1, 4, 5]):
        image = panel_images[index].crop(crop_box)
        image.thumbnail((580, 450))
        crops.paste(image, (i * 600 + 10, 30))
        draw.text((i * 600 + 10, 10), panel_titles[index], fill="black")
    crops.save(FIGURES / f"{case_id}_core_crop.png")

save_csv(DATA / "quality_metrics.csv", metric_rows)
save_csv(DATA / "fusion_comparison.csv", fusion_rows)
timing_rows = []
for row in runs:
    if row["kind"] == "warmup":
        continue
    stages = row["cuda_event_seconds"]
    timing_rows.append(dict(
        run_id=row["run_id"], case_id=row["case_id"], config=row["config"]["id"], kind=row["kind"], repeat=row["repeat"],
        e2e_seconds=row["e2e_seconds"], dit_seconds=stages.get("dit", 0),
        prompt_encode_seconds=stages.get("prompt_encode", 0), vae_encode_seconds=stages.get("vae_encode", 0),
        vae_decode_seconds=stages.get("vae_decode", 0), scheduler_seconds=stages.get("scheduler", 0),
        dit_calls=row["dit_calls"], allocated_GiB=row["peak_allocated_bytes"] / 2**30,
        reserved_GiB=row["peak_reserved_bytes"] / 2**30,
        target_h=row["target_hw"][0], target_w=row["target_hw"][1],
        initial_latent_sha256=row["initial_latent_sha256"], output_sha256=row["output_sha256"],
    ))
save_csv(DATA / "timings.csv", timing_rows)
summary = dict(configs={}, anchor={}, fusion=fusion_rows, total_requests=len(runs), total_dit_calls=sum(row["dit_calls"] for row in runs))
for config_id in config_ids:
    selected = [row for row in measurements if row["config"]["id"] == config_id]
    ratios = [row["e2e_seconds"] / lookup[(row["case_id"], config_ids[0])]["e2e_seconds"] for row in selected]
    summary["configs"][config_id] = dict(
        mean_seconds=statistics.mean(row["e2e_seconds"] for row in selected),
        paired_geometric_ratio_to_A=math.exp(statistics.mean(math.log(value) for value in ratios)),
        mean_lpips_spatial_to_A=statistics.mean(row["lpips_spatial_mean_to_reference"] for row in metric_rows if row["config"] == config_id),
        max_allocated_GiB=max(row["peak_allocated_bytes"] / 2**30 for row in selected),
        mean_dit_fraction=statistics.mean(row["cuda_event_seconds"]["dit"] / row["e2e_seconds"] for row in selected),
    )
    anchor = [row for row in runs if row["config"]["id"] == config_id and row["case_id"] == cases[0]["case_id"] and row["kind"] != "warmup"]
    values = [row["e2e_seconds"] for row in anchor]
    summary["anchor"][config_id] = dict(n=len(values), seconds=values, median=statistics.median(values), cv=statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else None,
        distinct_output_hashes=len({row["output_sha256"] for row in anchor}))
(DATA / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
metadata = dict(packages={name: importlib.metadata.version(name) for name in ["torch", "torchvision", "numpy", "scipy", "scikit-image", "lpips"]}, lpips_version="0.1.4", network="alex", spatial=True, normalize=True, device="cuda", regional_method="average of full-image spatial LPIPS map; receptive fields cross boundaries", ring_method="cumulative L1 dilation outside coarse core; scaled by shorter image edge / 1024; overlapping rings", weights={})
weight_paths = list(Path.home().joinpath(".cache/torch/hub/checkpoints").glob("alexnet*.pth"))
weight_paths.append(Path(lpips.__file__).parent / "weights" / "v0.1" / "alex.pth")
for p in weight_paths:
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    metadata["weights"][p.name] = h.hexdigest()
(DATA / "metrics_environment.json").write_text(json.dumps(metadata, indent=2))
plt.figure(figsize=(9, 4))
for i, case in enumerate(cases):
    values = [lookup[(case["case_id"], config)]["e2e_seconds"] for config in config_ids]
    plt.bar(np.arange(len(config_ids)) + (i - 1) * 0.25, values, width=0.25, label=case["case_id"])
plt.xticks(np.arange(len(config_ids)), [name.split("_")[0] for name in config_ids])
plt.ylabel("End-to-end seconds (single main measurement)")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "latency.png", dpi=180)
plt.close()
print(json.dumps(summary, ensure_ascii=False, indent=2))
