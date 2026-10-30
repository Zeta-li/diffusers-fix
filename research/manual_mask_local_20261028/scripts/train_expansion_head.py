import argparse
import json
from pathlib import Path
import shutil
import time

import torch
from expansion_head import SideExpansionHead, side_features, margins_to_box
import run_s1 as common

ROOT=common.ROOT


def reference_path(case):
    run="free_contract_v2" if case["source_group_id"] in {"15953","286770","26498","486018"} else "free_full20_v1"
    return ROOT/"runs"/run/case["case_id"]/"F.pt"


def target_margins(case,label):
    b=case["bbox"];t=label["influence_bbox"]
    return [max(0,b[0]-t[0])/500,max(0,b[1]-t[1])/500,
            max(0,t[2]-b[2])/500,max(0,t[3]-b[3])/500]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--attempt",default="expansion_fit_v1");args=ap.parse_args()
    out=ROOT/"runs"/args.attempt
    if out.exists():raise RuntimeError("new attempt required")
    spec=json.loads((ROOT/"data/free_head24_master_v2.json").read_text())
    labels_path=ROOT/"data/free_head24_labels_v1.json"
    labels=json.loads(labels_path.read_text())["labels"]
    out.mkdir(parents=True)
    for name in ("expansion_head.py","train_expansion_head.py"):
        shutil.copy2(ROOT/"scripts"/name,out/name)
    shutil.copy2(labels_path,out/labels_path.name)
    shutil.copy2(ROOT/"data/free_head24_master_v2.json",out/"master.json")
    torch.manual_seed(20261028);torch.cuda.manual_seed_all(20261028);torch.set_num_threads(8)
    rows={};provenance={}
    for c in spec["cases"]:
        # Validation data and its labels are never opened by the fitter.
        if c["role"]=="internal_validation":continue
        label=labels[c["case_id"]]
        if not label["usable_for_training"]:continue
        path=reference_path(c)
        data=torch.load(path,weights_only=True,map_location="cuda")
        x,a=side_features(data["warmup_clean"],data["condition"],data["text_pool"],c["bbox"],c["source_wh"])
        rows[c["case_id"]]=(c,x,a,torch.tensor(target_margins(c,label),device="cuda"))
        provenance[c["case_id"]]=dict(role=c["role"],reference_sha256=common.sha(path))
    fit=[r for r in rows.values() if r[0]["role"]=="fit"]
    cal=[r for r in rows.values() if r[0]["role"]=="calibration"]
    if len(fit)<6 or len(cal)<2:raise RuntimeError("insufficient reliable sources: require fit6 calibration2")
    x=torch.cat([r[1] for r in fit]);available=torch.cat([r[2] for r in fit]);y=torch.cat([r[3] for r in fit])
    mean=x.mean(0);std=x.std(0).clamp_min(.1);z=(x-mean)/std
    model=SideExpansionHead(x.shape[-1]).cuda();optim=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
    history=[];started=time.perf_counter()
    for step in range(1200):
        pred=model(z,available)
        mse=(pred-y).square().mean();miss=torch.relu(y-pred).square().mean();area=pred.mean()
        loss=mse+4*miss+.002*area
        optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step()
        if step%50==0 or step==1199:
            row=dict(step=step+1,loss=float(loss.detach()),mse=float(mse.detach()),miss=float(miss.detach()),area=float(area.detach()))
            history.append(row);common.append(out/"training.jsonl",row)
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    model.eval()
    with torch.no_grad():
        residuals=[torch.relu(r[3]-model((r[1]-mean)/std,r[2])) for r in cal]
        pad=torch.stack(residuals).amax(0)
        bundle=dict(state_dict=model.state_dict(),feature_dim=x.shape[-1],mean=mean,std=std,safety_pad=pad,
            seed=20261028,fit_source_ids=[r[0]["source_group_id"] for r in fit],
            calibration_source_ids=[r[0]["source_group_id"] for r in cal],training_steps=1200)
        torch.save(bundle,out/"head.pt")
        predictions={}
        for id,(c,xx,aa,yy) in rows.items():
            raw=model((xx-mean)/std,aa);pred=torch.minimum(raw+pad,aa)
            predictions[id]=dict(role=c["role"],target=yy.cpu().tolist(),raw=raw.cpu().tolist(),
                calibrated=pred.cpu().tolist(),envelope=margins_to_box(pred,c["bbox"],c["source_wh"]))
    result=dict(status="TRAINING_COMPLETE_CLOSED_LOOP_PENDING",training_updates=1200,
        parameter_count=sum(p.numel() for p in model.parameters()),fit_sources=len(fit),calibration_sources=len(cal),
        validation_sources_read=0,teacher_dit=0,gpu_dit=0,rl_updates=0,training_seconds=seconds,
        safety_pad=pad.cpu().tolist(),last_loss=history[-1],checkpoint_sha256=common.sha(out/"head.pt"),
        labels_sha256=common.sha(labels_path),provenance=provenance,predictions=predictions,
        calibration_meaning="max positive residual over valid calibration sources; no distribution-free guarantee")
    common.write(out/"result.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ["provenance","predictions"]}))


if __name__=="__main__":main()
