import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw


CONFIGS = ["F", "R2", "FIX0", "FIX1", "FIX2", "ROT"]
PARTIAL_STEPS = [2, 3, 4, 5, 6, 7]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def background_order(core, seed, case_id):
    h, w = core.shape
    blocks = []
    for ty in range((h + 1) // 2):
        for tx in range((w + 1) // 2):
            indices = [y * w + x for y in range(2 * ty, min(h, 2 * ty + 2)) for x in range(2 * tx, min(w, 2 * tx + 2)) if not core[y, x]]
            if indices:
                digest = hashlib.sha256(f"{seed}:{case_id}:{ty}:{tx}".encode()).hexdigest()
                blocks.append((digest, ty, tx, indices))
    blocks.sort(key=lambda row: (row[0], row[1], row[2]))
    return blocks, [index for _, _, _, indices in blocks for index in indices]


def histogram(values):
    return {str(key): int(count) for key, count in sorted(Counter(int(value) for value in values).items())}


def schedule_audit(per_step, target_count, sigmas, labels):
    last_real = np.full(target_count, -1, dtype=np.int64)
    audits = []
    for step in range(8):
        selected = np.array(per_step[str(step)], dtype=np.int64)
        valid = last_real >= 0
        before = np.where(valid, step - last_real, -1)
        before_sigma = np.abs(sigmas[step] - sigmas[np.maximum(last_real, 0)])
        last_real[selected] = step
        after = step - last_real
        assert np.all(last_real >= 0)
        after_sigma = np.abs(sigmas[step] - sigmas[last_real])
        active = np.zeros(target_count, dtype=bool)
        active[selected] = True
        audits.append({
            "step_1based": step + 1,
            "step_0based": step,
            "group": labels[step],
            "full_step": selected.size == target_count,
            "selected_target_count": int(selected.size),
            "inactive_target_count": int(target_count - selected.size),
            "max_age_before_update": int(before[valid].max()) if valid.any() else None,
            "uninitialized_before_update": int((~valid).sum()),
            "age_before_update_histogram": histogram(before),
            "max_age_at_attention_read": int(after.max()),
            "age_at_attention_read_histogram": histogram(after),
            "max_inactive_age_at_attention_read": int(after[~active].max()) if (~active).any() else None,
            "max_sigma_age_before_update": float(before_sigma[valid].max()) if valid.any() else None,
            "max_sigma_age_at_attention_read": float(after_sigma.max()),
            "max_inactive_sigma_age_at_attention_read": float(after_sigma[~active].max()) if (~active).any() else None,
        })
    return audits


def preview_panel(source, core, extra, title, subtitle, partition=None):
    display_w = 400
    display_h = round(source.height * display_w / source.width)
    image = source.resize((display_w, display_h), Image.Resampling.LANCZOS)
    array = np.asarray(image).astype(np.float32)
    if partition is not None:
        colors = {0: [250, 80, 75], 1: [65, 210, 100], 2: [235, 190, 40], 3: [30, 210, 230], 4: [70, 130, 245]}
        for label, color in colors.items():
            mask = np.asarray(Image.fromarray(partition == label).resize(image.size, Image.Resampling.NEAREST))
            array[mask] = .65 * array[mask] + .35 * np.array(color)
    else:
        for mask, color, alpha in [(extra, [255, 115, 30], .35), (core, [50, 130, 245], .30)]:
            expanded = np.asarray(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST))
            array[expanded] = (1 - alpha) * array[expanded] + alpha * np.array(color)
    image = Image.fromarray(np.uint8(np.clip(array, 0, 255)))
    result = Image.new("RGB", (410, display_h + 64), "white")
    draw = ImageDraw.Draw(result)
    draw.text((5, 7), title, fill="black")
    draw.text((5, 26), subtitle, fill="black")
    result.paste(image, (5, 52))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-root", type=Path, default=Path(__file__).resolve().parents[2] / "局部执行与边界验证_20261022")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--seed", type=int, default=2026102203)
    args = parser.parse_args()
    previous, root = args.previous_root.resolve(), args.output_root.resolve()
    source_manifest_path = previous / "masks/regions.json"
    source_hash = sha256(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    old_runs = [json.loads(line) for line in (previous / "data/runs.jsonl").read_text().splitlines() if line.strip()]
    old_f = {row["case_id"]: row for row in old_runs if row.get("status") == "complete" and row.get("kind") == "measurement" and row.get("config") == "F" and row.get("repeat") == 0}
    masks_dir = root / "masks"
    previews_dir = masks_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "frozen_before_rotation_outputs_after_prior_local_and_attribution_results",
        "previous_root": str(previous),
        "source_regions_path": str(source_manifest_path),
        "source_regions_sha256": source_hash,
        "generation_script_sha256": sha256(Path(__file__)),
        "seed": args.seed,
        "configs": CONFIGS,
        "full_steps_1based": [1, 8],
        "partial_steps_1based": PARTIAL_STEPS,
        "warmups_per_case_config": 1,
        "measurement_repeats": 3,
        "planned_requests": 72,
        "planned_dit_forwards": 576,
        "group_order": "2x2 target-token tiles sorted by SHA256(seed:case_id:tile_row:tile_column), then tile row/column as collision tie break; within each tile target row then column, excluding R2 core.",
        "remainder_rule": "Take first r=N_background%3 background tokens in the frozen tile order as E; E is refreshed in all FIX and ROT partial steps. Split the remaining ordered sequence into three equal-length consecutive groups. Splits inside a tile are allowed and listed.",
        "rotation_rule": "Partial steps 2/5 use G0, 3/6 use G1, 4/7 use G2; R2 core and E always refreshed. FIXj uses Gj on all six partial steps. Steps 1/8 are full.",
        "indexing": "Each per_step_indices dict uses string keys 0..7, zero-based scheduler indices. Values are sorted row-major target-image indices only. Condition-image and text tokens are always computed separately.",
        "age_protocol": "Before update, age=step0-last_real_step0, with -1 for uninitialized; selected K/V are updated to the current state before attention. At attention read, age=step0-last_real_step0_after_update. ROT max read age<=2; max before-update age may be3. All indices are real recomputations from current latent and original position.",
        "sigma_age_formula": "sigma_age_at_read(token,s)=abs(sigma[s]-sigma[last_real_step_after_update[token]]). Values below use the old F scheduler sigmas as frozen planning metadata; runtime must verify the same actual schedule.",
        "causal_limit": "KV and velocity are jointly refreshed. This is a posthoc three-demo mechanism screen, not a blinded/generalization test or a pure KV causal intervention. FIX/ROT are matched by target-token count, not by all geometry or cache history.",
        "preview": "masks/rotation_groups_preview.png",
        "cases": {},
    }
    combined_previews = []
    for case_id in ["blocks_text", "shirt_color", "remove_hair"]:
        old = source_manifest["cases"][case_id]
        h, w = old["target_grid_hw"]
        n = h * w
        core_indices = sorted(old["regions"]["R2"]["indices"])
        core = np.zeros((h, w), dtype=bool)
        core.reshape(-1)[core_indices] = True
        assert core.sum() == old["regions"]["R2"]["n_active_target_tokens"]
        blocks, ordered = background_order(core, args.seed, case_id)
        r = len(ordered) % 3
        remainder = ordered[:r]
        rest = ordered[r:]
        q = len(rest) // 3
        groups = {f"G{i}": sorted(rest[i * q:(i + 1) * q]) for i in range(3)}
        assert len(set(ordered)) == len(ordered) == n - int(core.sum())
        sets = [set(core_indices), set(remainder)] + [set(groups[f"G{i}"]) for i in range(3)]
        assert sum(len(value) for value in sets) == n and len(set.union(*sets)) == n
        assert all(not (sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))
        assert all(len(value) == q for value in groups.values())
        assignments = {index: name for name, values in [("E", remainder)] + list(groups.items()) for index in values}
        split_tiles = []
        for digest, ty, tx, indices in blocks:
            grouped = {}
            for index in indices:
                grouped.setdefault(assignments[index], []).append(index)
            if len(grouped) > 1:
                split_tiles.append({"tile_row": ty, "tile_column": tx, "assignment_indices": grouped})
        sigma_values = np.asarray(old_f[case_id]["sigmas"], dtype=np.float64)
        assert sigma_values.size == 9
        full = list(range(n))
        case = {
            "source_image_previous_relative": old["source_image"],
            "source_image_sha256": old["source_sha256"],
            "source_R2_mask_sha256": old["regions"]["R2"]["mask_sha256"],
            "target_hw": old["target_hw"],
            "target_grid_hw": [h, w],
            "n_target_tokens": n,
            "core_indices": core_indices,
            "n_core_tokens": len(core_indices),
            "n_background_tokens": len(ordered),
            "always_fresh_remainder_indices": sorted(remainder),
            "n_remainder_tokens": r,
            "n_tokens_per_group": q,
            "n_fixed_or_rotating_partial_tokens": len(core_indices) + r + q,
            "groups": {name: {"indices": values, "n_tokens": len(values)} for name, values in groups.items()},
            "background_order_indices": ordered,
            "background_order_sha256": hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest(),
            "tile_boundary_splits": split_tiles,
            "tile_boundary_split_count": len(split_tiles),
            "planning_sigmas": sigma_values.tolist(),
            "planning_timesteps": old_f[case_id]["timesteps"],
            "configs": {},
            "preview": f"masks/previews/{case_id}_rotation_groups.png",
        }
        for config in CONFIGS:
            per_step, labels = {}, []
            for step in range(8):
                if config == "F" or step in [0, 7]:
                    selected, label = full, "FULL"
                elif config == "R2":
                    selected, label = core_indices, "CORE_ONLY"
                else:
                    group = f"G{(step - 1) % 3}" if config == "ROT" else f"G{config[-1]}"
                    selected = sorted(core_indices + remainder + groups[group])
                    label = group
                assert len(selected) == len(set(selected)) and set(core_indices).issubset(selected)
                per_step[str(step)] = selected
                labels.append(label)
            default_indices = per_step["1"]
            audits = schedule_audit(per_step, n, sigma_values, labels)
            if config == "ROT":
                assert all(row["max_age_at_attention_read"] <= 2 for row in audits)
                assert Counter(row["group"] for row in audits if not row["full_step"]) == Counter({"G0": 2, "G1": 2, "G2": 2})
            case["configs"][config] = {
                "native": config == "F",
                "indices": default_indices,
                "n_partial_target_tokens": len(default_indices),
                "partial_target_fraction": len(default_indices) / n,
                "full_steps_1based": list(range(1, 9)) if config == "F" else [1, 8],
                "per_step_indices": per_step,
                "per_step_audit": audits,
                "target_token_step_budget": sum(len(value) for value in per_step.values()),
            }
        budgets = {case["configs"][name]["target_token_step_budget"] for name in ["FIX0", "FIX1", "FIX2", "ROT"]}
        assert len(budgets) == 1
        for step in range(8):
            assert len({len(case["configs"][name]["per_step_indices"][str(step)]) for name in ["FIX0", "FIX1", "FIX2", "ROT"]}) == 1
        source_path = previous / old["source_image"]
        assert sha256(source_path) == old["source_sha256"]
        source = Image.open(source_path).convert("RGB")
        partition = np.full((h, w), -1, dtype=np.int8)
        partition.reshape(-1)[core_indices] = 4
        partition.reshape(-1)[remainder] = 3
        for j in range(3):
            partition.reshape(-1)[groups[f"G{j}"]] = j
        panels = [preview_panel(source, core, core, f"{case_id}: frozen partition", "core blue | E cyan | G0 red G1 green G2 yellow", partition=partition)]
        for j in range(3):
            extra = np.zeros_like(core)
            extra.reshape(-1)[remainder + groups[f"G{j}"]] = True
            panels.append(preview_panel(source, core, extra, f"FIX{j} / ROT steps {j + 2},{j + 5}", f"{len(core_indices) + r + q}/{n} target tokens; core always fresh"))
        row_preview = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "white")
        left = 0
        for panel in panels:
            row_preview.paste(panel, (left, 0))
            left += panel.width
        row_preview.save(previews_dir / f"{case_id}_rotation_groups.png")
        combined_previews.append(row_preview)
        manifest["cases"][case_id] = case
    combined = Image.new("RGB", (max(image.width for image in combined_previews), sum(image.height for image in combined_previews)), "white")
    top = 0
    for image in combined_previews:
        combined.paste(image, (0, top))
        top += image.height
    combined.save(masks_dir / "rotation_groups_preview.png")
    assert sha256(source_manifest_path) == source_hash
    write_json(masks_dir / "rotation_plan.json", manifest)
    print(json.dumps({"planned_requests": 72, "planned_dit_forwards": 576, "seed": args.seed, "cases": {cid: {"core": c["n_core_tokens"], "E": c["n_remainder_tokens"], "group_size": c["n_tokens_per_group"], "partial_target": c["n_fixed_or_rotating_partial_tokens"], "fraction": c["n_fixed_or_rotating_partial_tokens"] / c["n_target_tokens"], "split_tiles": c["tile_boundary_split_count"], "ROT_max_read_age": max(row["max_age_at_attention_read"] for row in c["configs"]["ROT"]["per_step_audit"])} for cid, c in manifest["cases"].items()}}, indent=2))


if __name__ == "__main__":
    main()
