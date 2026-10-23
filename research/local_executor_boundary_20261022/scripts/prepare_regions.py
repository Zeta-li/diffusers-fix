import argparse
import hashlib
import json
import math
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw


POLYGONS = {
    "blocks_text": [
        [[0.285, 0.320], [0.320, 0.300], [0.500, 0.290], [0.750, 0.310], [0.765, 0.365], [0.745, 0.470], [0.590, 0.490], [0.550, 0.475], [0.310, 0.490], [0.280, 0.430]],
        [[0.325, 0.500], [0.690, 0.490], [0.945, 0.490], [0.970, 0.530], [0.960, 0.650], [0.940, 0.690], [0.730, 0.690], [0.550, 0.680], [0.480, 0.730], [0.460, 0.670], [0.330, 0.660]],
    ],
    "shirt_color": [
        [[0.260, 0.520], [0.310, 0.480], [0.390, 0.450], [0.470, 0.500], [0.520, 0.540], [0.590, 0.490], [0.650, 0.500], [0.720, 0.540], [0.770, 0.610], [0.800, 0.720], [0.830, 0.930], [0.850, 1.000], [0.180, 1.000], [0.190, 0.830], [0.210, 0.720], [0.240, 0.630]],
    ],
    "remove_hair": [
        [[0.420, 0.240], [0.520, 0.240], [0.520, 0.300], [0.490, 0.350], [0.540, 0.400], [0.640, 0.420], [0.720, 0.400], [0.780, 0.450], [0.750, 0.500], [0.640, 0.490], [0.530, 0.440], [0.440, 0.400], [0.400, 0.340], [0.400, 0.280]],
    ],
}
PURPOSES = {
    "blocks_text": "Manual context around wooden block rows, cast-shadow margins and the paper contact/seam.",
    "shirt_color": "Manual shoulder, hair-to-shirt contact and sleeve context; the frozen core already includes hand occlusions.",
    "remove_hair": "Manual plate-rim and hair-to-fork contact context around the frozen hair-edit core.",
}
REGION_NAMES = {
    "R0": "Frozen core box, conservatively covered by target tokens",
    "R1": "Fixed Chebyshev dilation of R0 by two target tokens",
    "R2": "R0 union manual source-image structure/contact polygons",
    "R3": "Ordinary distance expansion with exactly the same target-token count as R2",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def box_mask(box, grid_h, grid_w):
    x0, y0, x1, y1 = box
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    mask[max(0, math.floor(y0 * grid_h)):min(grid_h, math.ceil(y1 * grid_h)), max(0, math.floor(x0 * grid_w)):min(grid_w, math.ceil(x1 * grid_w))] = True
    return mask


def dilate(mask, radius):
    h, w = mask.shape
    padded = np.pad(mask, radius)
    result = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result |= padded[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
    return result


def polygon_mask(polygons, target_h, target_w, grid_h, grid_w):
    image = Image.new("1", (target_w, target_h), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        points = [(min(target_w - 1, max(0, round(x * (target_w - 1)))), min(target_h - 1, max(0, round(y * (target_h - 1))))) for x, y in polygon]
        draw.polygon(points, fill=1)
    array = np.asarray(image, dtype=bool)
    return array.reshape(grid_h, target_h // grid_h, grid_w, target_w // grid_w).any(axis=(1, 3))


def equal_budget_expansion(core, budget, case_id, seed):
    ys, xs = np.nonzero(core)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    candidates = []
    for y, x in np.argwhere(~core):
        dy, dx = max(y0 - int(y), int(y) - y1, 0), max(x0 - int(x), int(x) - x1, 0)
        digest = hashlib.sha256(f"{seed}:{case_id}:{int(y)}:{int(x)}".encode()).hexdigest()
        candidates.append((dy * dy + dx * dx, digest, int(y), int(x)))
    candidates.sort()
    result = core.copy()
    for _, _, y, x in candidates[:budget - int(core.sum())]:
        result[y, x] = True
    return result


def components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    count = 0
    while remaining:
        count += 1
        queue = deque([remaining.pop()])
        while queue:
            y, x = queue.popleft()
            for cell in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if cell in remaining:
                    remaining.remove(cell)
                    queue.append(cell)
    return count


def perimeter(mask):
    padded = np.pad(mask.astype(np.int8), 1)
    return int(np.abs(np.diff(padded, axis=0)).sum() + np.abs(np.diff(padded, axis=1)).sum())


def overlay(source, mask, core, title, count):
    image = source.copy().convert("RGB")
    h, w = image.height, image.width
    full = np.asarray(Image.fromarray(mask).resize((w, h), Image.Resampling.NEAREST))
    core_full = np.asarray(Image.fromarray(core).resize((w, h), Image.Resampling.NEAREST))
    array = np.asarray(image).astype(np.float32)
    array[full] = array[full] * 0.73 + np.array([240, 115, 30]) * 0.27
    array[core_full] = array[core_full] * 0.75 + np.array([30, 155, 245]) * 0.25
    image = Image.fromarray(np.uint8(np.clip(array, 0, 255)))
    edge = full & ~np.pad(full, 1)[0:h, 1:w + 1]
    edge |= full & ~np.pad(full, 1)[2:h + 2, 1:w + 1]
    edge |= full & ~np.pad(full, 1)[1:h + 1, 0:w]
    edge |= full & ~np.pad(full, 1)[1:h + 1, 2:w + 2]
    painted = np.asarray(image).copy()
    painted[edge] = [255, 55, 20]
    image = Image.fromarray(painted)
    image.thumbnail((440, 400))
    panel = Image.new("RGB", (456, 456), "white")
    panel.paste(image, ((456 - image.width) // 2, 42))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 9), f"{title} | {count}/{mask.size} tokens ({count / mask.size:.1%})", fill="black")
    draw.text((8, 433), "Blue: frozen core | orange/red: computation region", fill="black")
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-root", type=Path, default=Path(__file__).resolve().parents[2] / "Lightning8前置实验_20261022")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--tie-seed", type=int, default=2026102202)
    args = parser.parse_args()
    previous, root = args.previous_root.resolve(), args.output_root.resolve()
    annotations_path = previous / "roi_annotations.json"
    annotations = json.loads(annotations_path.read_text())
    runs = [json.loads(line) for line in (previous / "data" / "runs.jsonl").read_text().splitlines()]
    geometries = {row["case_id"]: row for row in runs if row.get("status") == "complete" and row.get("kind") == "measurement" and row["config"]["id"] == "A_base50_cfg4"}
    for name in ["inputs", "masks", "figures"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "frozen_before_new_local_execution_outputs",
        "annotation_provenance": "Manual source-image-only geometry diagnostic. Prior dense/Lightning baseline outputs were already available to the researchers; this is not a blinded annotation or an automatic method.",
        "model_output_pixels_read_by_script": False,
        "old_roi_sha256": sha256(annotations_path),
        "geometry_source": "Prior run metadata only: target_hw and first_cond_shapes. No generated image is opened.",
        "packing": "Row-major target index = row * target_grid_width + column; target pixels per token = 16 x 16 in these cases.",
        "full_context_policy": {"condition_image_tokens": "always computed", "text_tokens": "always computed", "region_masks_apply_to": "target image tokens only"},
        "r0_rule": "Include every target token intersecting the normalized frozen core box; floor lower bounds and ceil upper bounds.",
        "r1_rule": "Chebyshev dilation by exactly 2 target-grid tokens, clipped at grid edges.",
        "r2_rule": "Union R0 and manual polygons rasterized at target resolution with rounded normalized vertices, then include any token with at least one polygon pixel.",
        "r3_rule": "Keep all R0; add nearest target-grid tokens by squared Euclidean distance to the R0 rectangle. Resolve equal-distance ties by SHA256(seed:case:row:column). Match R2 count exactly.",
        "tie_seed": args.tie_seed,
        "fairness": "R2 and R3 share the same core, target-token count, complete condition/text context and intended refresh schedule. Connectivity/perimeter are reported rather than assumed equal. No generated output is used to select masks.",
        "cases": {},
    }
    all_panels = []
    for case_id in ["blocks_text", "shirt_color", "remove_hair"]:
        row = geometries[case_id]
        target_h, target_w = row["target_hw"]
        shape = row["first_cond_shapes"]["img_shapes"][0]
        grid_h, grid_w = shape[0][1:]
        cond_h, cond_w = shape[1][1:]
        text_tokens = row["first_cond_shapes"]["encoder_hidden_states"][1]
        assert target_h == grid_h * 16 and target_w == grid_w * 16
        assert [grid_h, grid_w] == ([64, 64] if case_id == "remove_hair" else [52, 78])
        source_path = previous / "inputs" / f"{case_id}.png"
        dest = root / "inputs" / source_path.name
        if dest.exists():
            assert sha256(dest) == sha256(source_path)
        else:
            shutil.copy2(source_path, dest)
        source = Image.open(dest).convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
        core_box = annotations["cases"][case_id]["core_box"]
        r0 = box_mask(core_box, grid_h, grid_w)
        r1 = dilate(r0, 2)
        r2 = r0 | polygon_mask(POLYGONS[case_id], target_h, target_w, grid_h, grid_w)
        r3 = equal_budget_expansion(r0, int(r2.sum()), case_id, args.tie_seed)
        masks = dict(R0=r0, R1=r1, R2=r2, R3=r3)
        assert int(r2.sum()) == int(r3.sum())
        assert np.any(r2 != r3)
        case_dir = root / "masks" / case_id
        case_dir.mkdir(exist_ok=True)
        case_record = {
            "source_image": str(dest.relative_to(root)),
            "source_sha256": sha256(dest),
            "source_wh": list(Image.open(dest).size),
            "target_hw": [target_h, target_w],
            "target_grid_hw": [grid_h, grid_w],
            "condition_grid_hw": [cond_h, cond_w],
            "n_target_tokens": grid_h * grid_w,
            "n_condition_tokens_always_computed": cond_h * cond_w,
            "n_positive_text_tokens_always_computed_in_old_run": text_tokens,
            "core_box_normalized_xyxy": core_box,
            "old_allowed_box_diagnostic_only": annotations["cases"][case_id]["allowed_box"],
            "manual_polygons_normalized_xy": POLYGONS[case_id],
            "manual_polygon_purpose": PURPOSES[case_id],
            "regions": {},
        }
        case_panels = []
        for name, mask in masks.items():
            assert np.all(mask[r0]) and 0 < int(mask.sum()) < mask.size
            path = case_dir / f"{name}.npy"
            np.save(path, mask, allow_pickle=False)
            Image.fromarray(np.uint8(mask) * 255).save(case_dir / f"{name}.tokens.png")
            Image.fromarray(np.uint8(mask) * 255).resize((target_w, target_h), Image.Resampling.NEAREST).save(case_dir / f"{name}.pixels.png")
            indices = np.flatnonzero(mask.reshape(-1)).tolist()
            count = len(indices)
            case_record["regions"][name] = {
                "description": REGION_NAMES[name],
                "mask_npy": str(path.relative_to(root)),
                "mask_sha256": sha256(path),
                "indices": indices,
                "n_active_target_tokens": count,
                "target_active_fraction": count / mask.size,
                "additional_tokens_beyond_r0": count - int(r0.sum()),
                "active_queries_including_complete_condition_and_text": count + cond_h * cond_w + text_tokens,
                "all_query_active_fraction": (count + cond_h * cond_w + text_tokens) / (mask.size + cond_h * cond_w + text_tokens),
                "connected_components_4": components(mask),
                "perimeter_grid_edges": perimeter(mask),
                "contains_r0": True,
            }
            case_panels.append(overlay(source, mask, r0, name, count))
        case_record["r2_r3_equal_target_count"] = True
        case_record["r2_r3_jaccard"] = float((r2 & r3).sum() / (r2 | r3).sum())
        case_record["r2_r3_symmetric_difference_tokens"] = int((r2 ^ r3).sum())
        manifest["cases"][case_id] = case_record
        panel = Image.new("RGB", (456 * 4, 480), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), f"{case_id}: R2 is manual structure prior; R2/R3 exact target-token budget match", fill="black")
        for column, image in enumerate(case_panels):
            panel.paste(image, (column * 456, 24))
        panel.save(root / "figures" / f"{case_id}_regions.png")
        all_panels.append(panel)
    combined = Image.new("RGB", (456 * 4, 480 * len(all_panels)), "white")
    for index, panel in enumerate(all_panels):
        combined.paste(panel, (0, index * 480))
    combined.save(root / "figures" / "regions_all_cases.png")
    (root / "masks" / "regions.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    summary = {case: {name: {"n": region["n_active_target_tokens"], "fraction": region["target_active_fraction"], "all_query_fraction": region["all_query_active_fraction"]} for name, region in data["regions"].items()} for case, data in manifest["cases"].items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
