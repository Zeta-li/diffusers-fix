"""Offline CPU analysis of run_s1.py; never makes a semantic quality decision.

Example: python scripts/analyze_s1.py --attempt runs/s1_v1 [--partial]
Pillow and NumPy are required. PyTorch is optional and only reads local tensor
snapshots on CPU with weights_only=True. No model, LPIPS, or network is used.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("NATIVE_FULL", "MASK_FULL")
DIAGNOSTICS = ("MASK_ALL_WHITE", "MASK_ALL_BLACK")
TARGET_WH = (1024, 1024)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path, errors, *, optional=False):
    if not Path(path).exists():
        if not optional:
            errors.append(f"missing JSON: {path}")
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        errors.append(f"unreadable JSON {path}: {type(exc).__name__}: {exc}")
        return None


def read_jsonl(path, errors):
    if not Path(path).exists():
        return []
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            # In --partial a concurrent append can leave an incomplete last line.
            # Preserve the issue and prohibit a complete result, never silently skip.
            errors.append(f"invalid JSONL line {number} in {path}: {exc}")
    return rows


def font(size):
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def mask_regions(case):
    original = Image.open(ROOT / case["allow_mask"])
    if original.mode != "L" or set(np.unique(np.asarray(original))).difference((0, 255)):
        raise ValueError("allow mask is not binary L 0/255")
    # This is the fixed evaluation canvas, not a claim about packed runtime mask.
    mask = original.resize(TARGET_WH, Image.Resampling.NEAREST)
    allow = np.asarray(mask) > 0
    # Exact65x65 binary dilation/erosion via summed-area table, edge padding.
    padded = np.pad(allow.astype(np.int32), ((32, 32), (32, 32)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    counts = integral[65:, 65:] - integral[:-65, 65:] - integral[65:, :-65] + integral[:-65, :-65]
    outer, inner = counts > 0, counts == 65 * 65
    return mask, {"all": np.ones_like(allow), "allow": allow,
                  "outside": ~allow, "boundary32": outer & ~inner}


def region_mae(first, second, regions):
    error = np.abs(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)) / 255.0
    return {name: float(error[selected].mean()) if selected.any() else None
            for name, selected in regions.items()}


def safe_torch_load(torch, path):
    return torch.load(path, map_location="cpu", weights_only=True)


def tensor_digest(torch, value):
    data = value.detach().contiguous().cpu()
    head = f"{data.dtype}:{tuple(data.shape)}:".encode()
    return hashlib.sha256(head + data.view(torch.uint8).numpy().tobytes()).hexdigest()


def control_summary(torch, path):
    control = safe_torch_load(torch, path)
    packed, latent = control["packed_mask"], control["latent_mask"]
    active = control["active_ids"]
    exact = torch.nonzero(packed.any(-1)[0], as_tuple=False).flatten()
    return {
        "path": str(path), "file_sha256": sha(path),
        "packed_mask_shape": list(packed.shape), "packed_mask_dtype": str(packed.dtype),
        "packed_mask_tensor_sha256": tensor_digest(torch, packed),
        "latent_mask_shape": list(latent.shape),
        "latent_mask_white_subpositions": int(latent.sum().item()),
        "actual_active_target_count": int(exact.numel()),
        "actual_target_count": int(packed.shape[1]),
        "actual_active_target_fraction": float(exact.numel() / packed.shape[1]),
        "active_ids_equal_mask_support": bool(torch.equal(active, exact)),
        "actual_active_ids": exact.tolist(),
        "estimate_is_not_ground_truth": True,
    }, control


def paired_latent(torch, native_path, mask_path, control):
    native = safe_torch_load(torch, native_path)["final"].float()
    masked = safe_torch_load(torch, mask_path)["final"].float()
    if native.shape != masked.shape:
        raise ValueError("paired final latent shapes differ")
    delta2 = (masked - native).square()
    packed = control["packed_mask"].bool()
    if packed.shape != delta2.shape:
        raise ValueError("paired latent and packed mask shapes differ")
    def rms(selected):
        return float(delta2[selected].mean().sqrt().item()) if selected.any().item() else None
    return {"shape": list(native.shape), "rms_all": float(delta2.mean().sqrt().item()),
            "rms_actual_allow_subpositions": rms(packed),
            "rms_actual_outside_subpositions": rms(~packed),
            "meaning": "Final packed latent MASK_FULL minus NATIVE_FULL, FP32 arithmetic on saved tensors; not an edit-quality metric."}


def ledger_summary(calls, runs, status, expected_pairs, errors):
    by = collections.defaultdict(lambda: {"invoked": 0, "completed": 0, "steps_invoked": [], "steps_completed": []})
    seen = collections.Counter()
    for row in calls:
        event = row.get("event")
        if event not in ("invoked", "completed"):
            errors.append(f"unknown calls ledger event: {event!r}")
            continue
        key = (row.get("case_id"), row.get("arm"))
        by[key][event] += 1
        by[key]["steps_" + event].append(row.get("step"))
        seen[(key, event, row.get("step"))] += 1
    duplicate_events = [str(key) for key, count in seen.items() if count != 1]
    if duplicate_events:
        errors.append("duplicate calls ledger event keys: " + ", ".join(duplicate_events))
    successful = collections.Counter((row.get("case_id"), row.get("arm")) for row in runs)
    for key, number in successful.items():
        if number != 1:
            errors.append(f"duplicate successful request rows: {key}: {number}")
        if by[key]["invoked"] != 8 or by[key]["completed"] != 8:
            errors.append(f"successful request lacks exactly 8+8 ledger events: {key}")
    actual_groups = set(by) | set(successful)
    active_key = None
    if status:
        cid = status.get("active_case", status.get("case_id"))
        arm = status.get("active_arm", status.get("arm"))
        if cid is not None and arm is not None:
            active_key = (cid, arm)
            actual_groups.add(active_key)
    unexpected = actual_groups - set(expected_pairs)
    if unexpected:
        errors.append("unexpected request groups retained in ledger: " + str(sorted(unexpected)))
    groups = []
    for key in sorted(actual_groups):
        item = dict(case_id=key[0], arm=key[1], **by[key],
                    successful_run_rows=successful[key], diagnostic=key[1] in DIAGNOSTICS)
        item["runtime_failed"] = bool(status and status.get("phase") == "failed" and active_key == key)
        item["unfinished_or_failed"] = successful[key] == 0
        if successful[key] and (sorted(item["steps_invoked"]) != list(range(8)) or
                                sorted(item["steps_completed"]) != list(range(8))):
            errors.append(f"successful request step indices are not 0..7: {key}")
        groups.append(item)
    invoked = sum(x["invoked"] for x in groups)
    completed = sum(x["completed"] for x in groups)
    for name, actual in (("invoked_dit", invoked), ("completed_dit", completed),
                         ("successful_requests", len(runs))):
        if status and status.get(name) != actual:
            errors.append(f"status/ledger count mismatch for {name}: {status.get(name)} vs {actual}")
    return {"source_of_truth": "Every calls.jsonl event, including diagnostics and incomplete/failed requests; runs.jsonl counts successful requests.",
            "actual_invoked_dit": invoked, "actual_completed_dit": completed,
            "uncompleted_invocations": invoked - completed,
            "successful_requests": len(runs), "attempted_request_groups": len(actual_groups),
            "main_successful_requests": sum(x["successful_run_rows"] for x in groups if not x["diagnostic"]),
            "diagnostic_successful_requests": sum(x["successful_run_rows"] for x in groups if x["diagnostic"]),
            "main_invoked_dit": sum(x["invoked"] for x in groups if not x["diagnostic"]),
            "diagnostic_invoked_dit": sum(x["invoked"] for x in groups if x["diagnostic"]),
            "failed_request_groups": sum(x["runtime_failed"] for x in groups),
            "unfinished_or_failed_groups": sum(x["unfinished_or_failed"] for x in groups),
            "groups": groups, "duplicate_event_keys": duplicate_events}


def placeholder(title, width=1024, height=1024):
    result = Image.new("RGB", (width, height), "#eeeeee")
    ImageDraw.Draw(result).text((30, 50), title, font=font(35), fill="#aa2222")
    return result


def panel(case, source, mask, images, destination):
    side, top, bottom = 512, 50, 104
    canvas = Image.new("RGB", (side * 4, side + top + bottom), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, value) in enumerate((("SOURCE / Lanczos 1024", source),
                                            ("ALLOW MASK / nearest", mask.convert("RGB")),
                                            ("NATIVE_FULL", images.get("NATIVE_FULL", placeholder("MISSING"))),
                                            ("MASK_FULL", images.get("MASK_FULL", placeholder("MISSING"))))):
        draw.text((column * side + 12, 10), label, fill="black", font=font(23))
        canvas.paste(value.resize((side, side), Image.Resampling.LANCZOS), (column * side, top))
    draw.text((12, side + top + 6), f"{case['order']:02d}  {case['case_id']}  [{case['task_type']}]  SOURCE-LEVEL TRAINING PILOT", fill="black", font=font(21))
    lines = textwrap.wrap(case["prompt"], width=160)
    for index, line in enumerate(lines[:2]):
        draw.text((12, side + top + 37 + index * 26), line, fill="black", font=font(20))
    canvas.save(destination)
    return canvas


def manual_template(cases, attempt):
    blank = {"review_status": "PENDING", "editing_success": None,
             "structure_intact": None, "seams_absent": None,
             "other_objects_preserved": None, "uncertain": None,
             "overall_semantic_pass": None, "notes": ""}
    return {"status": "PENDING_MANUAL_REVIEW", "attempt": str(attempt),
            "reviewer": None, "review_time_local": None,
            "instructions": "Review source, mask, complete images, and boundaries. Numeric similarity cannot determine semantic correctness. Fill booleans explicitly and explain uncertainty. Preserve every source in denominator. F is not ground truth.",
            "frozen_gate": {"mask_semantic_min_of_8": 7, "mask_semantic_min_per_stratum_of_4": 3,
                            "severe_mask_boundary_failures_max": 0,
                            "note": "Requires actual reviewer observations; this script never sets quality PASS."},
            "cases": [{"case_id": c["case_id"], "task_type": c["task_type"],
                       "NATIVE_FULL": blank.copy(), "MASK_FULL": blank.copy(),
                       "mask_introduced_semantic_failure": None,
                       "severe_mask_boundary_failure": None,
                       "paired_notes": ""} for c in cases],
            "final_decision": "PENDING", "quality_pass": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, help="runs/s1_v1, s1_v1, or an absolute attempt directory")
    parser.add_argument("--partial", action="store_true", help="Render completed items; incomplete never receives a complete/PASS status")
    args = parser.parse_args()
    candidate = Path(args.attempt)
    attempt = candidate if candidate.is_absolute() else ROOT / (candidate if candidate.parts[0] == "runs" else Path("runs") / candidate)
    if not attempt.is_dir():
        parser.error(f"attempt directory does not exist: {attempt}")
    output = attempt / "analysis"
    output.mkdir(exist_ok=True)
    errors, warnings = [], []
    cases_path = ROOT / "data/cases_s1.json"
    spec = json.loads(cases_path.read_text())
    cases = spec["cases"]
    if len(cases) != 8 or collections.Counter(c["task_type"] for c in cases) != {"replace": 4, "attribute": 4}:
        raise ValueError("expected frozen 8-case 4+4 training pilot")
    status = read_json(attempt / "status.json", errors)
    protocol = read_json(attempt / "preregistration.json", errors)
    if protocol and protocol.get("cases_sha256") != sha(cases_path):
        errors.append("cases manifest differs from run preregistration hash")
    calls = read_jsonl(attempt / "calls.jsonl", errors)
    runs = read_jsonl(attempt / "runs.jsonl", errors)
    diag_case = next(c for c in cases if c["img_id"] == "15953")
    expected_pairs = [(c["case_id"], arm) for c in cases for arm in ARMS]
    expected_pairs += [(diag_case["case_id"], arm) for arm in DIAGNOSTICS]
    ledger = ledger_summary(calls, runs, status, expected_pairs, errors)
    torch = None
    try:
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(4)
    except ImportError:
        warnings.append("PyTorch unavailable: saved control tensors and paired latent RMS skipped; all image/ledger analysis remains available.")
    metrics, paired, tensor_details, completion, panels = [], [], [], [], []
    reconstruction_metrics, reconstruction_tiles, missing_recon = [], [], []
    rows_by_key = {(r.get("case_id"), r.get("arm")): r for r in runs}
    for case in cases:
        cid = case["case_id"]
        try:
            for key, hk in (("image", "source_sha256"), ("user_mask", "user_mask_sha256"), ("allow_mask", "allow_mask_sha256")):
                if sha(ROOT / case[key]) != case[hk]:
                    raise ValueError(f"asset hash mismatch: {key}")
            source = Image.open(ROOT / case["image"]).convert("RGB").resize(TARGET_WH, Image.Resampling.LANCZOS)
            mask, regions = mask_regions(case)
        except (OSError, ValueError) as exc:
            errors.append(f"{cid} source/mask: {exc}")
            continue
        images, arm_records = {}, {}
        source_recon = None
        recon_path = attempt / cid / "SOURCE_RECON.png"
        if recon_path.exists():
            try:
                source_recon = Image.open(recon_path).convert("RGB")
                if source_recon.size != TARGET_WH:
                    raise ValueError(f"expected1024x1024 SOURCE_RECON, got {source_recon.size}")
                recon_mae = region_mae(source_recon, source, regions)
                reconstruction_metrics.append({
                    "case_id": cid, "img_id": case["img_id"], "task_type": case["task_type"],
                    "path": str(recon_path), "sha256": sha(recon_path),
                    **{f"mae_recon_vs_source_{k}": v for k, v in recon_mae.items()}})
                # Separate diagnostic contact: main four-column panels stay unchanged.
                difference = np.abs(np.asarray(source_recon, dtype=np.int16) -
                                    np.asarray(source, dtype=np.int16))
                amplified = Image.fromarray(np.clip(difference * 8, 0, 255).astype(np.uint8))
                recon_tile = Image.new("RGB", (1536, 570), "white")
                recon_draw = ImageDraw.Draw(recon_tile)
                for column, (label, value) in enumerate((("SOURCE / Lanczos1024", source),
                                                         ("SOURCE_RECON / VAE", source_recon),
                                                         ("ABS RGB DIFFERENCE x8", amplified))):
                    recon_draw.text((column * 512 + 9, 8), label, fill="black", font=font(22))
                    recon_tile.paste(value.resize((512, 512), Image.Resampling.LANCZOS), (column * 512, 43))
                recon_draw.text((8, 550), f"{case['order']:02d} {cid} / zero DiT reconstruction", fill="black", font=font(16))
                reconstruction_tiles.append(recon_tile)
            except (OSError, ValueError) as exc:
                errors.append(f"{cid}/SOURCE_RECON: {exc}")
                source_recon = None
        else:
            missing_recon.append(cid)
        for arm in ARMS:
            base = attempt / cid
            png, record_path = base / f"{arm}.png", base / f"{arm}.json"
            exists = {"png": png.exists(), "record": record_path.exists(),
                      "runs_ledger_row": (cid, arm) in rows_by_key}
            record = read_json(record_path, errors, optional=True)
            ready = all(exists.values()) and record is not None
            completion.append({"case_id": cid, "arm": arm, "ready": ready, **exists})
            if not ready:
                continue
            if record.get("case_id") != cid or record.get("arm") != arm or record.get("calls") != 8:
                errors.append(f"{cid}/{arm}: request result record binding/call count differs")
            arm_records[arm] = record
            try:
                value = Image.open(png).convert("RGB")
                if value.size != TARGET_WH:
                    raise ValueError(f"expected1024x1024 result, got {value.size}")
                images[arm] = value
            except (OSError, ValueError) as exc:
                errors.append(f"{cid}/{arm}: {exc}")
                continue
            maes = region_mae(value, source, regions)
            row = {"case_id": cid, "img_id": case["img_id"], "task_type": case["task_type"],
                   "arm": arm, "image_sha256": sha(png),
                   "record_sha256": sha(record_path), "eval_allow_pixels": int(regions["allow"].sum()),
                   "eval_outside_pixels": int(regions["outside"].sum()),
                   "eval_boundary32_pixels": int(regions["boundary32"].sum())}
            row.update({f"mae_vs_source_{name}": score for name, score in maes.items()})
            if source_recon is not None:
                recon_diff = region_mae(value, source_recon, regions)
                row.update({f"mae_vs_source_recon_{name}": score for name, score in recon_diff.items()})
                row["source_recon_sha256"] = sha(recon_path)
            metrics.append(row)
        if all(arm in images for arm in ARMS):
            diff = region_mae(images["MASK_FULL"], images["NATIVE_FULL"], regions)
            paired.append({"case_id": cid, "task_type": case["task_type"],
                           **{f"mae_mask_vs_native_{k}": v for k, v in diff.items()}})
            next(r for r in metrics if r["case_id"] == cid and r["arm"] == "MASK_FULL").update(
                {f"mae_mask_vs_native_{k}": v for k, v in diff.items()})
        control_path = attempt / cid / "MASK_FULL_control.pt"
        if torch is not None and control_path.exists():
            try:
                detail, control = control_summary(torch, control_path)
                detail.update(case_id=cid, geometric_estimate=case.get("mask_processing_reference", {}))
                if not detail["active_ids_equal_mask_support"]:
                    errors.append(f"{cid}: saved active IDs differ from actual packed mask support")
                npth, mpth = attempt / cid / "NATIVE_FULL.pt", attempt / cid / "MASK_FULL.pt"
                if npth.exists() and mpth.exists():
                    detail["paired_latent"] = paired_latent(torch, npth, mpth, control)
                tensor_details.append(detail)
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                errors.append(f"{cid}: saved CPU tensor analysis failed: {type(exc).__name__}: {exc}")
        elif "MASK_FULL" in arm_records:
            manifest = arm_records["MASK_FULL"].get("mask_record", {})
            tensor_details.append({"case_id": cid, "status": "JSON_MANIFEST_ONLY_TENSOR_NOT_READ",
                                   "active_target_count_from_runtime_json": manifest.get("active_target_count"),
                                   "active_target_ids_from_runtime_json": manifest.get("active_target_ids"),
                                   "tensor_sha256_from_runtime_json": manifest.get("tensor_sha256"),
                                   "reason": "PyTorch unavailable" if torch is None else "control.pt missing",
                                   "estimate_is_not_ground_truth": True})
        if images:
            path = output / f"{case['order']:02d}__{cid}__panel.png"
            panels.append({"case_id": cid, "path": str(path), "image": panel(case, source, mask, images, path)})
    diagnostic_results = []
    for arm in DIAGNOSTICS:
        key = (diag_case["case_id"], arm)
        record_path = attempt / key[0] / f"{arm}.json"
        record = read_json(record_path, errors, optional=True)
        checks = record.get("diagnostic_checks", {}) if record else {}
        expected_keys = ("native_stepwise_bitwise_equal", "native_condition_bitwise_equal") if arm == "MASK_ALL_WHITE" else ("final_bitwise_source",)
        ready = record is not None and key in rows_by_key and all(checks.get(k) is True for k in expected_keys)
        diagnostic_results.append({"case_id": key[0], "arm": arm, "completed_and_checks_true": ready,
                                   "saved_diagnostic_checks": checks, "record_path": str(record_path)})
        if record and not all(checks.get(k) is True for k in expected_keys):
            errors.append(f"{arm}: diagnostic equivalence checks not all true")
    overview_files = []
    for begin in (0, 4):
        group = panels[begin:begin + 4]
        if not group:
            continue
        width = 1600
        thumb_h = round(group[0]["image"].height * width / group[0]["image"].width)
        overview = Image.new("RGB", (width, thumb_h * len(group)), "white")
        for index, item in enumerate(group):
            overview.paste(item["image"].resize((width, thumb_h), Image.Resampling.LANCZOS), (0, index * thumb_h))
        path = output / f"overview_{begin+1:02d}_{begin+len(group):02d}.png"
        overview.save(path)
        overview_files.append({"path": str(path), "sha256": sha(path)})
    reconstruction_overviews = []
    for begin in (0, 4):
        group = reconstruction_tiles[begin:begin + 4]
        if not group:
            continue
        overview = Image.new("RGB", (1536, 570 * len(group)), "white")
        for index, tile in enumerate(group):
            overview.paste(tile, (0, 570 * index))
        path = output / f"source_reconstruction_overview_{begin+1:02d}_{begin+len(group):02d}.png"
        overview.save(path)
        reconstruction_overviews.append({"path": str(path), "sha256": sha(path)})
    if missing_recon:
        warnings.append("SOURCE_RECON is unavailable for these sources; reconstruction-relative metrics skipped without changing the DiT ledger: " + ", ".join(missing_recon))
    main_ready = len(completion) == 16 and all(x["ready"] for x in completion) and len(metrics) == 16
    diagnostic_ready = all(x["completed_and_checks_true"] for x in diagnostic_results)
    run_complete = bool(status and status.get("phase") == "generation_complete_quality_pending" and
                        ledger["actual_invoked_dit"] == 144 and ledger["actual_completed_dit"] == 144 and
                        ledger["successful_requests"] == 18 and main_ready and diagnostic_ready and not errors)
    if not run_complete:
        warnings.append("Attempt incomplete or inconsistent: partial images and every observed call are retained; no complete/PASS result is issued.")
    aggregates = []
    for stratum in ("all", "replace", "attribute"):
        for arm in ARMS:
            selected = [r for r in metrics if r["arm"] == arm and (stratum == "all" or r["task_type"] == stratum)]
            aggregates.append({"stratum": stratum, "arm": arm, "n_completed": len(selected),
                               "mean_mae_vs_source": {name: float(np.mean([r[f"mae_vs_source_{name}"] for r in selected if r[f"mae_vs_source_{name}"] is not None]))
                                                      if any(r[f"mae_vs_source_{name}"] is not None for r in selected) else None
                                                      for name in ("all", "allow", "outside", "boundary32")}})
    reconstruction_aggregate = {}
    for name in ("all", "allow", "outside", "boundary32"):
        values = [r[f"mae_recon_vs_source_{name}"] for r in reconstruction_metrics
                  if r[f"mae_recon_vs_source_{name}"] is not None]
        reconstruction_aggregate[name] = float(np.mean(values)) if values else None
    reconstruction_relative_aggregates = []
    for arm in ARMS:
        selected = [r for r in metrics if r["arm"] == arm and "mae_vs_source_recon_all" in r]
        means = {}
        for name in ("all", "allow", "outside", "boundary32"):
            values = [r[f"mae_vs_source_recon_{name}"] for r in selected
                      if r[f"mae_vs_source_recon_{name}"] is not None]
            means[name] = float(np.mean(values)) if values else None
        reconstruction_relative_aggregates.append({"arm": arm, "n_sources_with_recon": len(selected),
                                                   "mean_mae_vs_source_recon": means})
    analysis = {
        "status": "NUMERIC_ANALYSIS_COMPLETE_MANUAL_REVIEW_PENDING" if run_complete else "INCOMPLETE_OR_INCONSISTENT_NO_PASS",
        "automatic_quality_pass": False, "manual_quality_status": "PENDING_REVIEW_SEPARATE_FROM_NUMERIC_ANALYSIS",
        "attempt": str(attempt), "script_sha256": sha(__file__), "cases_sha256": sha(cases_path),
        "partial_mode": args.partial, "run_complete_and_counts_consistent": run_complete,
        "calls_ledger_sha256": sha(attempt / "calls.jsonl") if (attempt / "calls.jsonl").exists() else None,
        "runs_ledger_sha256": sha(attempt / "runs.jsonl") if (attempt / "runs.jsonl").exists() else None,
        "runtime_status": status, "ledger": ledger, "completion": completion,
        "metric_contract": {"rgb_scale": "mean absolute channel difference divided by255, float64 CPU",
                            "source_reference": "same original source resized once to1024x1024 with PIL LANCZOS; not a VAE reconstruction baseline",
                            "evaluation_mask": "M_allow500x500 resized PIL nearest to1024x1024; fixed across both methods",
                            "boundary32": "L-infinity band dilation(radius32) minus erosion(radius32), binary65x65 summed-area filter with edge padding; inside+outside",
                            "runtime_mask_difference": "Evaluation mask is a pixel-level measurement region. Saved actual packed latent mask is reported separately and may differ by nearest sampling/packing. No estimate equality assertion.",
                            "no_rgb_copy": True, "semantic_inference_from_metrics": False},
        "metrics": metrics, "paired_image_metrics": paired, "aggregates": aggregates,
        "source_reconstruction": {
            "purpose": "SOURCE_RECON is an additional VAE decode of the native reference-condition latent, not a DiT request or a semantic teacher.",
            "source_reference": "same original source resized with PIL LANCZOS1024 as the main evaluation; differences include VAE reconstruction and preprocessing conventions",
            "n_saved_reconstruction_images": len(reconstruction_metrics),
            "additional_dit": 0, "counted_as_new_requests": False,
            "missing_source_ids": missing_recon,
            "metrics": reconstruction_metrics,
            "mean_mae_recon_vs_source": reconstruction_aggregate,
            "arm_relative_aggregates": reconstruction_relative_aggregates,
            "overviews": reconstruction_overviews,
            "difference_visualization": "absolute per-channel RGB difference multiplied8, clipped to0..255; black is zero. Main four-column overviews unchanged.",
            "interpretation": "Reconstruction is a background/preprocessing reference, not semantic ground truth; no quality conclusion follows automatically."},
        "saved_control_and_latent_details": tensor_details, "torch_cpu_analysis_available": torch is not None,
        "diagnostics": diagnostic_results,
        "panels": [{"case_id": p["case_id"], "path": p["path"], "sha256": sha(p["path"])} for p in panels],
        "overviews": overview_files, "errors": errors, "warnings": warnings,
        "quality_gate": {"MASK_FULL_semantic_min_of8": 7, "min_per_stratum_of4": 3,
                         "severe_mask_boundary_failures_max": 0, "decision_requires_actual_manual_review": True},
        "new_model_calls_by_this_analysis": 0, "lpips_calls": 0, "network_calls": 0,
        "limitations": ["Training pilot only, not independent validation.",
                        "MAE cannot determine instruction success, object completeness, or natural boundaries.",
                        "These diagnostic executions include CPU snapshots and assertions; no net-latency claim."]}
    write_json(output / "analysis.json", analysis)
    fieldnames = ["case_id", "img_id", "task_type", "arm", "image_sha256", "record_sha256",
                  "eval_allow_pixels", "eval_outside_pixels", "eval_boundary32_pixels"]
    fieldnames += [f"mae_vs_source_{name}" for name in ("all", "allow", "outside", "boundary32")]
    fieldnames += ["source_recon_sha256"]
    fieldnames += [f"mae_vs_source_recon_{name}" for name in ("all", "allow", "outside", "boundary32")]
    fieldnames += [f"mae_mask_vs_native_{name}" for name in ("all", "allow", "outside", "boundary32")]
    with (output / "summary_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    recon_fields = ["case_id", "img_id", "task_type", "path", "sha256"]
    recon_fields += [f"mae_recon_vs_source_{name}" for name in ("all", "allow", "outside", "boundary32")]
    with (output / "source_reconstruction_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=recon_fields)
        writer.writeheader()
        writer.writerows(reconstruction_metrics)
    review_path = output / "manual_review_template.json"
    if not review_path.exists():
        write_json(review_path, manual_template(cases, attempt))
    print(json.dumps({"status": analysis["status"], "analysis": str(output / "analysis.json"),
                      "main_completed": len(metrics), "actual_invoked_dit": ledger["actual_invoked_dit"],
                      "actual_completed_dit": ledger["actual_completed_dit"],
                      "successful_requests": ledger["successful_requests"], "errors": errors,
                      "manual_review_preserved_or_created": str(review_path)}, ensure_ascii=False))
    if not run_complete and not args.partial:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
