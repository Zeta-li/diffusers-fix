import collections
import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/latency_temporal_v1"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def metric(path, reference, box, source_wh):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    b = np.asarray(Image.open(reference).convert("RGB"), dtype=np.int32)
    error = np.abs(a - b)
    h, w = a.shape[:2]
    sw, sh = source_wh
    x0, y0, x1, y1 = box
    left, top = math.floor(x0 * w / sw), math.floor(y0 * h / sh)
    right, bottom = math.ceil(x1 * w / sw), math.ceil(y1 * h / sh)
    mask = np.ones((h, w), dtype=bool)
    mask[top:bottom, left:right] = False
    whole = int(error.sum()) / (error.size * 255)
    outside = int(error[mask].sum()) / (int(mask.sum()) * 3 * 255)
    return dict(mae_vs_F=whole, outside_mae_vs_F=outside,
                whole_integer_abs_sum=int(error.sum()), outside_integer_abs_sum=int(error[mask].sum()),
                outside_pixel_count=int(mask.sum()), evaluation_box_output=[left, top, right, bottom],
                numeric_pass=whole <= .015 and outside <= .01)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-only", action="store_true")
    args = parser.parse_args()
    status = json.loads((RUN / "status.json").read_text())
    if status["phase"] != "generation_complete_analysis_pending" and not args.latency_only:
        raise RuntimeError("Model batch incomplete")
    spec = json.loads((RUN / "preregistration.json").read_text())
    rows = [json.loads(line) for line in (RUN / "runs.jsonl").read_text().splitlines()]
    assert len(rows) == 82 or args.latency_only
    out = RUN / "analysis"
    out.mkdir(exist_ok=True)
    latency_rows = [r for r in rows if r["phase"] == "latency"]
    paired = []
    for case in spec["cases"]:
        record = dict(case_id=case["case_id"], source_group_id=case["source_group_id"],
                      historical_F_semantic_pass=case["historical_F_semantic_pass"])
        for arm in ("F", "FIX32"):
            values = sorted([r for r in latency_rows if r["case_id"] == case["case_id"] and r["arm"] == arm], key=lambda r: r["repeat"])
            assert len(values) == 3
            times = [r["total_wall_seconds"] for r in values]
            components = {key: [sum(r["component_cuda_ms"][key]) / 1000 for r in values]
                          for key in ("dit", "prompt_encode", "prepare_latents_vae_encode", "vae_decode")}
            record[arm] = dict(seconds=times, median_seconds=statistics.median(times),
                               mean_seconds=statistics.mean(times), min_seconds=min(times), max_seconds=max(times),
                               range_over_median=(max(times)-min(times))/statistics.median(times),
                               components_seconds=components,
                               component_medians_seconds={k: statistics.median(v) for k,v in components.items()},
                               dit_per_step_ms_median=[statistics.median(r["component_cuda_ms"]["dit"][i] for r in values) for i in range(8)],
                               plan_seconds=[r["plan_seconds"] for r in values],
                               peak_allocated_GiB=[r["peak_allocated_bytes"] / 2**30 for r in values])
        record["speedup_medians"] = record["F"]["median_seconds"] / record["FIX32"]["median_seconds"]
        record["latency_reduction"] = 1 - 1/record["speedup_medians"]
        record["paired_repeat_speedups"] = [x/y for x,y in zip(record["F"]["seconds"],record["FIX32"]["seconds"])]
        record["target_update_fraction"] = case["plans"]["EARLY"]["total_target_token_updates"]/(8*4096)
        record["all_image_token_update_fraction"] = (record["target_update_fraction"] + 1)/2
        record["quality"] = metric(RUN / "latency" / case["case_id"] / "FIX32_r0.png",
                                    RUN / "latency" / case["case_id"] / "F_r0.png", case["evaluation_bbox"], case["source_wh"])
        paired.append(record)
    aggregate = dict(n_sources=8, n_repeats_each_arm=3, diagnostic_only=True, deployable_quality_claim=False)
    for arm in ("F", "FIX32"):
        aggregate[arm] = dict(mean_source_median_seconds=statistics.mean(r[arm]["median_seconds"] for r in paired),
            median_source_median_seconds=statistics.median(r[arm]["median_seconds"] for r in paired),
            component_mean_source_medians_seconds={key: statistics.mean(r[arm]["component_medians_seconds"][key] for r in paired)
                for key in ("dit","prompt_encode","prepare_latents_vae_encode","vae_decode")})
    aggregate["ratio_mean_source_medians"] = aggregate["F"]["mean_source_median_seconds"]/aggregate["FIX32"]["mean_source_median_seconds"]
    aggregate["latency_reduction_mean_source_medians"] = 1 - 1/aggregate["ratio_mean_source_medians"]
    aggregate["median_paired_source_speedup"] = statistics.median(r["speedup_medians"] for r in paired)
    aggregate["range_paired_source_speedup"] = [min(r["speedup_medians"] for r in paired),max(r["speedup_medians"] for r in paired)]
    aggregate["mean_target_update_fraction"] = statistics.mean(r["target_update_fraction"] for r in paired)
    aggregate["mean_all_image_update_fraction"] = statistics.mean(r["all_image_token_update_fraction"] for r in paired)
    rng = np.random.default_rng(2026102801)
    boots = []
    for _ in range(10000):
        chosen = rng.integers(0,8,8)
        boots.append(sum(paired[i]["F"]["median_seconds"] for i in chosen) / sum(paired[i]["FIX32"]["median_seconds"] for i in chosen))
    aggregate["source_bootstrap_95pct_speed_ratio"] = np.quantile(boots,[.025,.975]).tolist()
    aggregate["bootstrap_limit"] = "8 seen sources; not independent deployment generalization; thermal/repetition effects not captured by this interval"
    write(out / "latency.json", dict(aggregate=aggregate,paired=paired,raw_rows=latency_rows))
    if args.latency_only:
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))
        return
    temporal = []
    panel_dir = out / "panels"
    panel_dir.mkdir(exist_ok=True)
    for case in spec["cases"]:
        directory = RUN / "temporal" / case["case_id"]
        for arm in ("EARLY", "MIDDLE", "LATE"):
            row = dict(case_id=case["case_id"], source_group_id=case["source_group_id"], arm=arm,
                       historical_F_semantic_pass=case["historical_F_semantic_pass"],
                       target_updates=case["plans"][arm]["total_target_token_updates"],
                       per_step_counts=case["plans"][arm]["target_tokens_per_step"])
            row.update(metric(directory/(arm+"_r0.png"),directory/"F_r0.png",case["evaluation_bbox"],case["source_wh"]))
            temporal.append(row)
        canvas = Image.new("RGB", (2000, 458), "white")
        draw = ImageDraw.Draw(canvas)
        names = ["SOURCE", "F", "EARLY", "MIDDLE", "LATE"]
        for i,name in enumerate(names):
            path = ROOT/case["image"] if name == "SOURCE" else directory/(name+"_r0.png")
            picture = Image.open(path).convert("RGB").resize((400,400))
            canvas.paste(picture,(i*400,35))
            draw.text((i*400+8,8),name,fill="black")
        draw.text((8,439),case["case_id"],fill="black")
        canvas.save(panel_dir/(case["case_id"]+".png"))
    summaries = {}
    for group in ("all8", "historical_F_pass"):
        selected = [c for c in spec["cases"] if group == "all8" or c["historical_F_semantic_pass"]]
        ids = [c["case_id"] for c in selected]
        sub = [r for r in temporal if r["case_id"] in ids]
        means = {arm: statistics.mean(r["mae_vs_F"] for r in sub if r["arm"] == arm) for arm in ("EARLY","MIDDLE","LATE")}
        outside_means = {arm: statistics.mean(r["outside_mae_vs_F"] for r in sub if r["arm"] == arm) for arm in means}
        best_constant = min(means,key=means.get)
        winners = [min([r for r in sub if r["case_id"] == case_id],key=lambda r:r["mae_vs_F"]) for case_id in ids]
        oracle_mean = statistics.mean(r["mae_vs_F"] for r in winners)
        count = dict(collections.Counter(r["arm"] for r in winners))
        summaries[group] = dict(n_sources=len(ids),mean_MAE=means,mean_outside_MAE=outside_means,
            numeric_passes={arm:sum(r["numeric_pass"] for r in sub if r["arm"] == arm) for arm in means},
            best_constant=best_constant,oracle_mean_MAE=oracle_mean,
            oracle_relative_gain_over_best_constant=1-oracle_mean/means[best_constant],
            oracle_winner_counts=count,dominant_winner_fraction=max(count.values())/len(ids),
            oracle_winners=[dict(case_id=r["case_id"],arm=r["arm"]) for r in winners])
    write(out / "temporal.json", dict(rows=temporal,summary=summaries,visual_review_pending=True,
        oracle_is_retrospective_not_trained_policy=True,new_training_updates=0))
    print(json.dumps(dict(latency=aggregate,temporal=summaries),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
