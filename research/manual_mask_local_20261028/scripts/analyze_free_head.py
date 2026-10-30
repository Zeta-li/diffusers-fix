import csv
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--fixed32",action="store_true");args=parser.parse_args()
    out=ROOT/"runs/free_head_closed_v1"/("analysis_fixed32" if args.fixed32 else "analysis")
    arms=["FIX16","FIX32","FIX48","HEAD","MATCH"] if args.fixed32 else ["FIX16","FIX48","HEAD","MATCH"]
    out.mkdir(exist_ok=True)
    spec=json.loads((ROOT/"data/free_head_closed_v1_preregistration.json").read_text())
    review={c["case_id"]:c for c in json.loads((ROOT/"data/free_full24_visual_review_v1.json").read_text())["cases"]}
    rows=[];panels=[]
    for c in spec["cases"]:
        id=c["case_id"];refdir=ROOT/"runs"/c["reference_run"]/id
        fimage=Image.open(refdir/"F.png").convert("RGB")
        f=np.asarray(fimage,dtype=np.float32)/255
        source=Image.open(ROOT/c["image"]).convert("RGB")
        raw=np.asarray(source.resize(fimage.size,Image.Resampling.LANCZOS),dtype=np.float32)/255
        box=review[id]["influence_bbox"];w,h=fimage.size
        xx0,yy0,xx1,yy1=[int(v/500*d) for v,d in zip(box,[w,h,w,h])]
        region=np.zeros((h,w),bool);region[yy0:yy1,xx0:xx1]=True
        images=[source,fimage];names=["SOURCE","F"]
        plan_counts={}
        for name in arms:
            run="free_contract_v2" if c["role"]=="fit" and name.startswith("FIX") else "free_head_closed_v1"
            if name=="FIX32":run="free_fixed32_v1"
            folder=ROOT/"runs"/run/id
            image=Image.open(folder/(name+".png")).convert("RGB")
            a=np.asarray(image,dtype=np.float32)/255
            info=json.loads((folder/(name+".json")).read_text());plan=info["record"]["plan"]
            error=np.abs(a-f);outside=float(error[~region].mean()) if (~region).any() else 0.
            target=review[id]["target_bbox"]
            tx0,ty0,tx1,ty1=[int(np.floor(v/500*64)) if i<2 else int(np.ceil(v/500*64)) for i,v in enumerate(target)]
            target_ids={y*64+x for y in range(max(0,ty0),min(64,ty1)) for x in range(max(0,tx0),min(64,tx1))}
            coverage=len(target_ids&set(plan["core_indices"]))/max(1,len(target_ids))
            mae=float(error.mean());numeric=mae<=.015 and outside<=.01
            plan_counts[name]=plan["target_tokens_per_step"]
            rows.append(dict(case_id=id,source_group_id=c["source_group_id"],role=c["role"],arm=name,
                full_reference_task_pass=review[id]["full_task_pass"],mae_vs_F=mae,
                outside_common_mae_vs_F=outside,inside_common_mae_vs_F=float(error[region].mean()) if region.any() else 0.,
                outside_common_mae_vs_source=float(np.abs(a-raw)[~region].mean()) if (~region).any() else 0.,
                p95_abs_error=float(np.quantile(error,.95)),numeric_gate_pass=bool(numeric),
                target_token_update_fraction=plan["total_target_token_updates"]/(8*4096),
                image_token_update_fraction=(plan["total_target_token_updates"]+8*4096)/(8*8192),
                core_fraction=len(plan["core_indices"])/4096,
                coarse_target_bbox_coverage=coverage,full_fallback=plan["kind"]=="all_selected",
                first_two_states_bitwise_native=info["checks"].get("first_two_states_bitwise_native"),
                matched_each_step_budget=info["checks"].get("matched_each_step_token_budget")))
            images.append(image);names.append(name)
        if plan_counts["HEAD"]!=plan_counts["MATCH"]:
            raise RuntimeError("unequal matched budget: "+id)
        panel=Image.new("RGB",((len(arms)+2)*260,295),"white");draw=ImageDraw.Draw(panel)
        for col,(im,name) in enumerate(zip(images,names)):
            panel.paste(im.resize((260,260)),(col*260,0));draw.text((col*260+4,263),c["source_group_id"]+" "+name,fill="black")
        panel.save(out/(id+"_panel.jpg"),quality=94);panels.append(panel)
    for page in range(3):
        subset=panels[page*4:page*4+4]
        canvas=Image.new("RGB",((len(arms)+2)*260,295*len(subset)),"white")
        for k,panel in enumerate(subset):canvas.paste(panel,(0,k*295))
        canvas.save(out/("overview_"+str(page)+".jpg"),quality=93)
    summary=[]
    for role in ["fit","calibration","internal_validation","all"]:
        for arm in arms:
            rs=[v for v in rows if v["arm"]==arm and (role=="all" or v["role"]==role)]
            summary.append(dict(role=role,arm=arm,n=len(rs),mean_mae_vs_F=float(np.mean([v["mae_vs_F"] for v in rs])),
                max_mae_vs_F=max(v["mae_vs_F"] for v in rs),
                mean_outside_common_mae_vs_F=float(np.mean([v["outside_common_mae_vs_F"] for v in rs])),
                mean_target_token_fraction=float(np.mean([v["target_token_update_fraction"] for v in rs])),
                mean_image_token_fraction=float(np.mean([v["image_token_update_fraction"] for v in rs])),
                numeric_passes=sum(v["numeric_gate_pass"] for v in rs),
                mean_coarse_coverage=float(np.mean([v["coarse_target_bbox_coverage"] for v in rs])),
                full_fallbacks=sum(v["full_fallback"] for v in rs)))
    with (out/"metrics.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(rows=rows,summary=summary,semantic_review_pending=True,net_latency_measured=False,
                common_evaluation_region="fixed full-reference visual-review influence bbox, shared by all arms",
                matched_each_step_budget_all_pass=True)
    (out/"metrics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
