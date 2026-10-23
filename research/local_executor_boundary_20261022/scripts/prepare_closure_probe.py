import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw

from prepare_regions import components, dilate, equal_budget_expansion, overlay, perimeter, polygon_mask, sha256


PLATE_POLYGON = [
    [1.000, 0.012], [0.875, 0.030], [0.735, 0.060], [0.650, 0.090],
    [0.585, 0.120], [0.535, 0.150], [0.497, 0.180], [0.445, 0.240],
    [0.408, 0.300], [0.392, 0.360], [0.391, 0.420], [0.407, 0.480],
    [0.436, 0.540], [0.480, 0.600], [0.546, 0.660], [0.632, 0.720],
    [0.690, 0.750], [0.767, 0.780], [0.860, 0.812], [0.980, 0.842],
    [1.000, 0.844],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    root = args.root.resolve()
    main_manifest = root / "masks/regions.json"
    main_hash = sha256(main_manifest)
    original = json.loads(main_manifest.read_text())
    source_data = original["cases"]["remove_hair"]
    source_path = root / source_data["source_image"]
    assert sha256(source_path) == source_data["source_sha256"]
    target_h, target_w = source_data["target_hw"]
    grid_h, grid_w = source_data["target_grid_hw"]
    assert [target_h, target_w, grid_h, grid_w] == [1024, 1024, 64, 64]
    core = np.load(root / source_data["regions"]["R0"]["mask_npy"], allow_pickle=False).astype(bool)
    raw_object = polygon_mask([PLATE_POLYGON], target_h, target_w, grid_h, grid_w)
    object_with_guard = dilate(raw_object, 1)
    r4 = core | object_with_guard
    seed = original["tie_seed"]
    r5 = equal_budget_expansion(core, int(r4.sum()), "remove_hair", seed)
    assert np.all(r4[core]) and np.all(r5[core])
    assert int(r4.sum()) == int(r5.sum()) and np.any(r4 != r5)
    assert 0 < int(r4.sum()) < grid_h * grid_w
    out_dir = root / "masks/closure_probe/remove_hair"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    case = {
        "source_image": source_data["source_image"],
        "source_sha256": sha256(source_path),
        "target_hw": [target_h, target_w],
        "target_grid_hw": [grid_h, grid_w],
        "n_target_tokens": grid_h * grid_w,
        "n_condition_tokens_always_computed": source_data["n_condition_tokens_always_computed"],
        "n_positive_text_tokens_always_computed_in_old_run": source_data["n_positive_text_tokens_always_computed_in_old_run"],
        "core_box_normalized_xyxy": source_data["core_box_normalized_xyxy"],
        "R0_mask_sha256": source_data["regions"]["R0"]["mask_sha256"],
        "whole_visible_plate_polygon_normalized_xy": PLATE_POLYGON,
        "raw_object_polygon_target_count": int(raw_object.sum()),
        "object_outward_guard_tokens_Linf": 1,
        "r4_rule": "R0 union the whole visible plate support polygon conservatively rasterized to target tokens, with one outward target-token guard. All plate-interior food and fork are inside this closed support; plate continuation outside the canvas is clipped at the image boundary.",
        "r5_rule": "Keep R0 and expand by the original squared-Euclidean distance to its rectangle; SHA256(seed:case:row:column) resolves equal-distance ties. Stop at exactly the R4 target-token count.",
        "regions": {},
    }
    source = Image.open(source_path).convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
    panels = [overlay(source, core, core, "R0 frozen core", int(core.sum()))]
    for name, mask in [("R4", r4), ("R5", r5)]:
        path = out_dir / f"{name}.npy"
        np.save(path, mask, allow_pickle=False)
        Image.fromarray(np.uint8(mask) * 255).save(out_dir / f"{name}.tokens.png")
        Image.fromarray(np.uint8(mask) * 255).resize((target_w, target_h), Image.Resampling.NEAREST).save(out_dir / f"{name}.pixels.png")
        indices = np.flatnonzero(mask.reshape(-1)).tolist()
        complete_context = case["n_condition_tokens_always_computed"] + case["n_positive_text_tokens_always_computed_in_old_run"]
        case["regions"][name] = {
            "mask_npy": str(path.relative_to(root)),
            "mask_sha256": sha256(path),
            "indices_sha256": hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest(),
            "indices": indices,
            "n_active_target_tokens": len(indices),
            "target_active_fraction": len(indices) / mask.size,
            "additional_tokens_beyond_R0": len(indices) - int(core.sum()),
            "all_query_active_fraction": (len(indices) + complete_context) / (mask.size + complete_context),
            "connected_components_4": components(mask),
            "perimeter_grid_edges": perimeter(mask),
            "contains_R0": True,
            "contains_whole_plate_polygon": bool(np.all(mask[raw_object])),
        }
        panels.append(overlay(source, mask, core, name, len(indices)))
    case["R4_R5_exact_target_token_count_match"] = True
    case["R4_R5_jaccard"] = float((r4 & r5).sum() / (r4 | r5).sum())
    case["R4_R5_symmetric_difference_tokens"] = int((r4 ^ r5).sum())
    manifest = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "posthoc_exploratory_after_primary_artifact_observed",
        "trigger": "A visible plate brightness block/seam aligned with the primary local computation region motivated testing a complete support-object boundary.",
        "annotation_provenance": "Manual polygon from the source image only, after a primary local artifact had already been observed. Prior baseline outputs were also known. This probe is not blinded, not preregistered, not an automatic region selector, and not evidence of innovation or general boundary superiority.",
        "model_output_pixels_read_by_script": False,
        "primary_regions_json_sha256": main_hash,
        "primary_regions_unchanged": True,
        "tie_seed": seed,
        "context_policy": "Region indices apply only to target image tokens. Condition-image and text tokens are always computed.",
        "intended_experiment": {"configs": ["F_closure", "R4", "R5"], "case_ids": ["remove_hair"], "full_steps_1based_local": [1, 8], "native_baseline": "F_closure", "warmups_per_config": 1, "repeats": 3, "planned_requests": 12},
        "limitations": "One posthoc case; R4/R5 have equal target-token counts but differ in placement, perimeter and topology. R4 includes a one-token outward guard around the entire support object, not a one-token ring around the hair. No claim of learned or automatic object closure. Inspect quality and end-to-end cost together; no replacement of primary results.",
        "cases": {"remove_hair": case},
    }
    combined = Image.new("RGB", (456 * 3, 486), "white")
    draw = ImageDraw.Draw(combined)
    draw.text((8, 7), "POSTHOC EXPLORATORY: source-only whole visible plate support; R4/R5 exact target-token budget", fill="black")
    for index, panel in enumerate(panels):
        combined.paste(panel, (index * 456, 30))
    combined.save(figures / "closure_probe_regions.png")
    boundary = source.copy()
    draw = ImageDraw.Draw(boundary)
    points = [(round(x * (target_w - 1)), round(y * (target_h - 1))) for x, y in PLATE_POLYGON]
    draw.line(points + [points[0]], fill=(255, 65, 20), width=4)
    draw.text((16, 16), "POSTHOC manual whole-plate polygon; mask adds a one-token outward guard", fill=(0, 0, 0), stroke_width=1, stroke_fill=(255, 255, 255))
    boundary.save(figures / "closure_probe_source_boundary.png")
    (root / "masks/closure_probe.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    assert sha256(main_manifest) == main_hash
    print(json.dumps({"regions": {name: {key: value for key, value in record.items() if key in ["n_active_target_tokens", "target_active_fraction", "all_query_active_fraction", "perimeter_grid_edges", "contains_whole_plate_polygon"]} for name, record in case["regions"].items()}, "R4_R5_jaccard": case["R4_R5_jaccard"], "primary_regions_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
