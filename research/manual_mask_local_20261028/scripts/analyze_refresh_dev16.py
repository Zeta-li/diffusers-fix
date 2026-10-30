import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/refresh_dev16_v1'


def main():
    torch.set_num_threads(4)
    spec=json.loads((RUN/'preregistration.json').read_text());status=json.loads((RUN/'status.json').read_text())
    assert status['phase']=='complete_analysis_pending' and status['completed_dit']==status['invoked_dit']==512 and status['successful_requests']==64
    rows=[];prefix=0;selected_matches=[]
    for c in spec['cases']:
        cid=c['case_id'];folder=RUN/'evaluation'/cid
        raw={a:torch.load(folder/(a+'.pt'),weights_only=True,map_location='cpu') for a in spec['evaluation_arms']}
        for arm in ('R25','R6','SUP'):
            for step in range(2):assert torch.equal(raw[arm]['states'][step],raw['F']['states'][step]);prefix+=1
        for step in range(4):assert torch.equal(raw['R6']['states'][step],raw['R25']['states'][step]);prefix+=1
        action=json.loads((folder/'SUP.json').read_text())['runtime_metadata']['resolved_action']
        if action in ('R25','R6'):
            assert all(torch.equal(a,b) for a,b in zip(raw['SUP']['states'],raw[action]['states']))
            selected_matches.append(dict(case_id=cid,action=action,steps=8))
        ref=np.asarray(Image.open(folder/'F.png').convert('RGB'),np.float32)/255
        h,w=ref.shape[:2];sw,sh=c['source_wh'];x0,y0,x1,y1=c['bbox']
        inside=np.zeros((h,w),bool)
        inside[max(0,int((y0-32)*h/sh)):min(h,int(np.ceil((y1+32)*h/sh))),max(0,int((x0-32)*w/sw)):min(w,int(np.ceil((x1+32)*w/sw)))]=True
        for arm in spec['evaluation_arms']:
            im=Image.open(folder/(arm+'.png')).convert('RGB');meta=json.loads((folder/(arm+'.json')).read_text())
            assert hashlib.sha256(im.tobytes()).hexdigest()==meta['RGB_sha256']
            assert len(raw[arm]['states'])==8 and torch.equal(raw[arm]['states'][-1],raw[arm]['final'])
            assert all(torch.isfinite(z).all() for z in raw[arm]['states'])
            error=np.abs(np.asarray(im,np.float32)/255-ref);outer=float(error[~inside].mean()) if (~inside).any() else 0.
            rows.append(dict(case_id=cid,arm=arm,RGB_MAE=float(error.mean()),inside_MAE=float(error[inside].mean()),outside_MAE=outer,
                surrogate_loss=float(error.mean())+0.25*outer,latent_MAE=float((raw[arm]['final'].float()-raw['F']['final'].float()).abs().mean()),
                target_update_fraction=1 if arm=='F' else c['plans'][arm]['total_target_token_updates']/32768,
                selected_action=meta['runtime_metadata']['resolved_action']))
    summary={};by={a:[r for r in rows if r['arm']==a] for a in spec['evaluation_arms']};rng=np.random.default_rng(2026103029)
    for a,rs in by.items():
        summary[a]=dict(n=len(rs),**{k:float(np.mean([r[k] for r in rs])) for k in ('RGB_MAE','inside_MAE','outside_MAE','surrogate_loss','latent_MAE','target_update_fraction')})
    comparisons={}
    for b,a in [('R25','R6'),('R25','SUP'),('R6','SUP')]:
        x=np.array([r['RGB_MAE'] for r in by[b]]);y=np.array([r['RGB_MAE'] for r in by[a]]);ix=rng.integers(0,16,(10000,16))
        comparisons[b+'_to_'+a]=dict(relative_RGB=float(y.mean()/x.mean()-1),wins=int((y<x-1e-10).sum()),ties=int((np.abs(y-x)<1e-10).sum()),
            bootstrap95_relative_RGB=np.percentile(y[ix].mean(1)/x[ix].mean(1)-1,[2.5,97.5]).tolist())
    gates={a:summary[a]['RGB_MAE']<=summary['R25']['RGB_MAE'] and summary[a]['surrogate_loss']<=summary['R25']['surrogate_loss'] for a in ('R6','SUP')}
    out=RUN/'analysis';out.mkdir(exist_ok=True)
    result=dict(status='NUMERIC_COMPLETE_VISUAL_PENDING',counts=status,records=rows,summary=summary,comparisons=comparisons,numeric_gates=gates,
        shared_prefix_checks=prefix,SUP_selected_fixed_matches=selected_matches,training_updates=0,RL_updates=0,new_net_latency_claim=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'source_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','SUP_selected_fixed_matches')},indent=2))


if __name__=='__main__':main()
