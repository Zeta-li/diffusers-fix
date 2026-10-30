import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch
from PIL import Image, ImageDraw

from controlled_mask import tensor_digest
from analyze_latency_temporal import metric

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/budget_preflight_v1"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    status = json.loads((RUN / "status.json").read_text())
    if status["phase"] != "generation_complete_analysis_pending":
        raise RuntimeError("Generation incomplete")
    spec = json.loads((RUN / "preregistration.json").read_text())
    rows = [json.loads(line) for line in (RUN / "runs.jsonl").read_text().splitlines()]
    if len(rows) != 50 or status["completed_dit"] != 400:
        raise RuntimeError("Count mismatch")
    out = RUN / "analysis"
    out.mkdir(exist_ok=True)
    (out / "panels").mkdir(exist_ok=True)
    audited_euler, cache_checks, repeats = 0, 0, []
    warm_prefix = None
    for row in rows:
        prefix = ROOT / row["output_prefix"]
        raw = torch.load(prefix.with_suffix(".pt"), map_location="cpu", weights_only=True)
        image = Image.open(prefix.with_suffix(".png")).convert("RGB")
        assert tensor_digest(raw["final"]) == row["final_tensor_sha256"]
        assert hashlib.sha256(image.tobytes()).hexdigest() == row["raw_RGB_sha256"]
        if raw["states"]:
            previous = raw["input0"]
            for k, state in enumerate(raw["states"]):
                expected = (previous.float() + (raw["sigmas"][k + 1] - raw["sigmas"][k]) * raw["velocities"][k]).to(state.dtype)
                assert torch.equal(expected, state)
                previous = state
                audited_euler += 1
            if row["phase"] == "warmup":
                if row["arm"] == "F":
                    warm_prefix = raw["states"][:2]
                else:
                    assert all(torch.equal(x, y) for x, y in zip(warm_prefix, raw["states"][:2]))
        detail = json.loads(prefix.with_suffix(".json").read_text())
        bridge = detail["bridge_record"]
        if row["arm"] in ("SOURCE", "ALL", "FULL"):
            for trace in bridge["trace"]:
                k = trace["step"] - 1
                plan = detail["plan"]
                anchor = k in plan["full_steps_zero_based"]
                m, active = trace["text_tokens"], plan["target_tokens_per_step"][k]
                assert trace["selected_target"] == active
                assert trace["fresh_condition"] == (4096 if anchor else 0)
                assert trace["fresh_text"] == (m if anchor or row["arm"] == "SOURCE" else 0)
                assert trace["joint_query_tokens"] == active + trace["fresh_condition"] + trace["fresh_text"]
                assert trace["joint_kv_tokens"] == 8192 + m
                cache_checks += 1
        if row["phase"] == "repeat":
            repeats.append(dict(case_id=row["case_id"], arm=row["arm"],
                                total_seconds=row["total_wall_seconds"],
                                deterministic=row["checks"]["repeat_final_bitwise"]))
    assert audited_euler == 112
    for name, digest in spec["code_sha256"].items():
        assert hashlib.sha256((RUN / "code_snapshot" / name).read_bytes()).hexdigest() == digest
    records = []
    for case in spec["cases"]:
        arms = {r["arm"]: r for r in rows if r["phase"] == "main" and r["case_id"] == case["case_id"]}
        ref = ROOT / (arms["F"]["output_prefix"] + ".png")
        record = dict(case_id=case["case_id"], source_group_id=case["source_group_id"],
                      prompt=case["prompt"], budget_B_per_step=case["plans"]["B"]["target_tokens_per_step"],
                      budget_upshift=case["plans"]["B"]["fallback_reason"], arms={})
        canvas = Image.new("RGB", (2400, 530), "white")
        draw = ImageDraw.Draw(canvas)
        for k, arm in enumerate(("INPUT", "F", "FIX32", "SOURCE", "ALL")):
            path = ROOT / case["image"] if arm == "INPUT" else ROOT / (arms[arm]["output_prefix"] + ".png")
            canvas.paste(Image.open(path).convert("RGB").resize((480, 480)), (k * 480, 28))
            draw.text((k * 480 + 8, 8), arm, fill="black")
            if arm == "INPUT":
                continue
            row = arms[arm]
            quality = metric(path, ref, case["evaluation_bbox"], case["source_wh"])
            quality.pop("numeric_pass")
            detailed = json.loads((ROOT / (row["output_prefix"] + ".json")).read_text())
            br = detailed["bridge_record"]
            trace = [] if br is None else [r for r in br["trace"] if r["layer"] == 0]
            query_fraction = 1.0 if not trace else statistics.mean(r["joint_query_tokens"] / r["joint_kv_tokens"] for r in trace)
            record["arms"][arm] = dict(seconds=row["total_wall_seconds"],
                speedup_vs_F=arms["F"]["total_wall_seconds"] / row["total_wall_seconds"],
                components={k: sum(v) / 1000 for k, v in row["component_cuda_ms"].items()},
                peak_GiB=row["peak_allocated_bytes"] / 2**30, query_row_fraction=query_fraction,
                cache_GiB=0 if br is None else br["raw_cache_bytes_before_clear"] / 2**30,
                quality=quality)
        draw.text((8, 510), case["case_id"], fill="black")
        canvas.save(out / "panels" / (case["case_id"] + ".png"))
        records.append(record)
    aggregate = {}
    fmean = statistics.mean(r["arms"]["F"]["seconds"] for r in records)
    for arm in ("F", "FIX32", "SOURCE", "ALL"):
        values = [r["arms"][arm] for r in records]
        seconds = statistics.mean(v["seconds"] for v in values)
        aggregate[arm] = dict(mean_seconds=seconds, speedup_vs_F=fmean / seconds,
            latency_reduction_vs_F=1 - seconds / fmean,
            mean_DiT_seconds=statistics.mean(v["components"]["dit"] for v in values),
            mean_MAE_vs_F=statistics.mean(v["quality"]["mae_vs_F"] for v in values),
            mean_outside_MAE_vs_F=statistics.mean(v["quality"]["outside_mae_vs_F"] for v in values),
            mean_query_row_fraction=statistics.mean(v["query_row_fraction"] for v in values),
            max_peak_GiB=max(v["peak_GiB"] for v in values),
            per_source_speedup_range=[min(v["speedup_vs_F"] for v in values), max(v["speedup_vs_F"] for v in values)],
            mean_components={k: statistics.mean(v["components"][k] for v in values)
                             for k in ("dit", "prompt_encode", "prepare_latents_vae_encode", "vae_decode")})
    result = dict(status="NUMERICAL_ANALYSIS_COMPLETE_VISUAL_REVIEW_PENDING", n_sources=8,
        scope="seen-source mechanism/development screen; main one sample/arm, extra2 sources two candidate repetitions",
        aggregate=aggregate, records=records, repeats=repeats,
        audit=dict(outputs=50, euler_steps=audited_euler, cache_trace_checks=cache_checks,
                   full_native_contract_sources=2, status="PASS"),
        engine=json.loads((RUN / "engine_choice.json").read_text()),
        actual_requests=50, actual_dit=400, new_training_updates=0, new_RL_updates=0)
    write(out / "results.json", result)
    print(json.dumps(dict(aggregate=aggregate, audit=result["audit"], engine=result["engine"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
