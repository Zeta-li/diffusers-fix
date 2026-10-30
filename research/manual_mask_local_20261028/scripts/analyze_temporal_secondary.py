import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
ANALYSIS=ROOT/"runs/latency_temporal_v1/analysis"


def mean_error(a,b):
    x=np.asarray(Image.open(a).convert("RGB"),dtype=np.int32)
    y=np.asarray(Image.open(b).convert("RGB"),dtype=np.int32)
    return int(np.abs(x-y).sum())/(x.size*255)


def main():
    temporal=json.loads((ANALYSIS/"temporal.json").read_text())
    spec=json.loads((ROOT/"data/latency_temporal_v1_preregistration.json").read_text())
    secondary=json.loads((ROOT/"data/latency_temporal_secondary_analysis_v1.json").read_text())
    old=json.loads((ROOT/"runs/free_head_closed_v1/analysis_fixed32/metrics.json").read_text())["rows"]
    matched=[]
    for c in spec["cases"]:
        if c["source_group_id"] not in secondary["matched_source_ids"]:continue
        directory=ROOT/"runs/latency_temporal_v1/temporal"/c["case_id"]
        budget=c["plans"]["EARLY"]["total_target_token_updates"]/32768
        fixed=[]
        for arm in ("FIX16","FIX32","FIX48"):
            path=(ROOT/c["old_fix32"]).with_suffix(".png") if arm=="FIX32" else (ROOT/c["old_reference"]).parent/(arm+".png")
            assert path.is_file()
            row=next(r for r in old if r["case_id"]==c["case_id"] and r["arm"]==arm)
            fixed.append(dict(arm=arm,MAE=mean_error(path,directory/"F_r0.png"),
                              fraction=row["target_token_update_fraction"],eligible=row["target_token_update_fraction"]<=budget))
        tr=[r for r in temporal["rows"] if r["case_id"]==c["case_id"]]
        matched.append(dict(case_id=c["case_id"],source_group_id=c["source_group_id"],fixed=fixed,
            baseline_MAE=next(r["mae_vs_F"] for r in tr if r["arm"]=="EARLY"),
            spatial_oracle_MAE=min(r["MAE"] for r in fixed if r["eligible"]),
            temporal_oracle_MAE=min(r["mae_vs_F"] for r in tr),
            temporal_MAE={r["arm"]:r["mae_vs_F"] for r in tr}))
    base=statistics.mean(r["baseline_MAE"] for r in matched)
    spatial=statistics.mean(r["spatial_oracle_MAE"] for r in matched)
    schedule=statistics.mean(r["temporal_oracle_MAE"] for r in matched)
    rng=np.random.default_rng(2026102802)
    bootstrap={}
    for label,allowed in [("all8",[c["case_id"] for c in spec["cases"]]),
                          ("historical_F_pass6",[c["case_id"] for c in spec["cases"] if c["historical_F_semantic_pass"]]),
                          ("current_strict_F_pass5_posthoc",[c["case_id"] for c in spec["cases"] if c["historical_F_semantic_pass"] and c["source_group_id"]!="429111"])]:
        matrix=np.asarray([[next(r["mae_vs_F"] for r in temporal["rows"] if r["case_id"]==cid and r["arm"]==a) for a in ("EARLY","MIDDLE","LATE")] for cid in allowed])
        samples=[]
        for _ in range(10000):
            sub=matrix[rng.integers(0,len(allowed),len(allowed))]
            samples.append(float(1-sub.min(axis=1).mean()/sub.mean(axis=0).min()))
        bootstrap[label]=dict(n=len(allowed),mean_MAE=dict(zip(["EARLY","MIDDLE","LATE"],matrix.mean(axis=0).tolist())),
            oracle_gain_over_best_constant=float(1-matrix.min(axis=1).mean()/matrix.mean(axis=0).min()),
            source_bootstrap_percentile95=np.quantile(samples,[.025,.975]).tolist(),
            caveat="retrospective oracle and best constant reselected within each resample; not a learned-policy confidence interval or independent generalization")
    result=dict(status="EXPLORATORY_SECONDARY_NOT_PRIMARY_GATE",matched_source_count=len(matched),matched_rows=matched,
                common_FIX32_mean_MAE=base,spatial_oracle_mean_MAE=spatial,temporal_oracle_mean_MAE=schedule,
                spatial_oracle_gain_vs_FIX32=1-spatial/base,temporal_oracle_gain_vs_FIX32=1-schedule/base,
                limitations=secondary["limitations"],sensitivity=bootstrap)
    (ANALYSIS/"secondary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!="matched_rows"},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
