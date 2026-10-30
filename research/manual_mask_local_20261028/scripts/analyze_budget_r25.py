import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/budget_r25_v1'


def main():
    torch.set_num_threads(4)
    spec=json.loads((RUN/'preregistration.json').read_text())
    status=json.loads((RUN/'status.json').read_text())
    assert status['invoked_dit']==status['completed_dit']==384 and status['successful_requests']==48
    out=RUN/'analysis';out.mkdir(exist_ok=True)
    rows=[];checks=[];same=[];rng=np.random.default_rng(2026103036)
    for case in spec['cases']:
        cid=case['case_id'];folder=RUN/'evaluation'/cid
        ref=np.asarray(Image.open(folder/'F.png').convert('RGB'),dtype=np.float32)/255
        h,w=ref.shape[:2];sw,sh=case['source_wh'];x0,y0,x1,y1=case['bbox']
        box=[max(0,int((x0-32)*w/sw)),max(0,int((y0-32)*h/sh)),min(w,int(np.ceil((x1+32)*w/sw))),min(h,int(np.ceil((y1+32)*h/sh)))]
        inside=np.zeros((h,w),dtype=bool);inside[box[1]:box[3],box[0]:box[2]]=True
        raw={a:torch.load(folder/(a+'.pt'),map_location='cpu',weights_only=True) for a in spec['evaluation_arms']}
        for step in range(4):
            assert torch.equal(raw['R25']['states'][step],raw['B']['states'][step])
            checks.append(dict(case_id=cid,kind='R25_B_shared_prefix',step=step))
        for step in range(2):
            assert torch.equal(raw['R25']['states'][step],raw['F']['states'][step])
        if case['plans']['R25']['selected_indices_per_step']==case['plans']['C']['selected_indices_per_step']:
            for step in range(8):assert torch.equal(raw['R25']['states'][step],raw['C']['states'][step])
            same.append(cid)
        for arm in spec['evaluation_arms']:
            im=Image.open(folder/(arm+'.png')).convert('RGB');meta=json.loads((folder/(arm+'.json')).read_text())
            assert hashlib.sha256(im.tobytes()).hexdigest()==meta['RGB_sha256']
            assert len(raw[arm]['states'])==8 and torch.equal(raw[arm]['final'],raw[arm]['states'][-1])
            assert torch.isfinite(raw[arm]['final']).all()
            error=np.abs(np.asarray(im,dtype=np.float32)/255-ref)
            plan=None if arm=='F' else case['plans'][arm]
            rows.append(dict(case_id=cid,role=case['role'],arm=arm,RGB_MAE=float(error.mean()),
                inside_MAE=float(error[inside].mean()),outside_MAE=float(error[~inside].mean()) if (~inside).any() else None,
                latent_MAE=float((raw[arm]['final'].float()-raw['F']['final'].float()).abs().mean()),
                target_update_fraction=1 if plan is None else plan['total_target_token_updates']/32768,
                effective_local_fraction=1 if plan is None else plan['effective_local_fraction']))
    assert len(same)==15 and len(checks)==192
    summary={}
    for role in ('train','calibration','all'):
        subset=rows if role=='all' else [r for r in rows if r['role']==role]
        by={arm:{r['case_id']:r for r in subset if r['arm']==arm} for arm in spec['evaluation_arms']}
        result={}
        for arm in by:
            vals=list(by[arm].values())
            result[arm]=dict(n=len(vals),**{k:float(np.mean([r[k] for r in vals if r[k] is not None]))
                for k in ('RGB_MAE','inside_MAE','outside_MAE','latent_MAE','target_update_fraction')})
        for src,dst in (('B','R25'),('C','R25'),('B50','C'),('R25','C')):
            ids=list(by[src]);a=np.array([by[src][cid]['RGB_MAE'] for cid in ids]);b=np.array([by[dst][cid]['RGB_MAE'] for cid in ids])
            indices=rng.integers(0,len(ids),(10000,len(ids)))
            result[src+'_to_'+dst]=dict(relative_change=float(b.mean()/a.mean()-1),improved=int((b<a-1e-10).sum()),
                tied=int((np.abs(b-a)<1e-10).sum()),bootstrap95_relative_change=np.percentile(b[indices].mean(1)/a[indices].mean(1)-1,[2.5,97.5]).tolist())
        changed=[cid for cid in by['R25'] if cid not in same]
        result['R25_C_distinct_plan_subset']=dict(n=len(changed),R25_MAE=float(np.mean([by['R25'][c]['RGB_MAE'] for c in changed])),C_MAE=float(np.mean([by['C'][c]['RGB_MAE'] for c in changed])),
            R25_wins=sum(by['R25'][c]['RGB_MAE']<by['C'][c]['RGB_MAE'] for c in changed))
        winners=collections.Counter();vals=[]
        for cid in by['B']:
            arm=min(('B','R25','C'),key=lambda a:by[a][cid]['RGB_MAE']);winners[arm]+=1;vals.append(by[arm][cid]['RGB_MAE'])
        best=min(('B','R25','C'),key=lambda a:result[a]['RGB_MAE'])
        result['oracle']=dict(best_fixed=best,winners=dict(winners),MAE=float(np.mean(vals)),relative_change_vs_best_fixed=float(np.mean(vals)/result[best]['RGB_MAE']-1),deployable=False)
        result['factorial_MAE_absolute']=dict(refresh25=result['R25']['RGB_MAE']-result['B']['RGB_MAE'],refresh50=result['C']['RGB_MAE']-result['B50']['RGB_MAE'],
            space3anchors=result['B50']['RGB_MAE']-result['B']['RGB_MAE'],space4anchors=result['C']['RGB_MAE']-result['R25']['RGB_MAE'])
        summary[role]=result
    output=dict(status='NUMERIC_COMPLETE_VISUAL_PENDING',actual_counts=status,reused_requests=192,records=rows,summary=summary,
        R25_B_first4_checks=192,R25_F_first2_checks=96,R25_C_exact_all8_case_ids=same,
        numeric_candidate_gate=summary['calibration']['R25']['RGB_MAE']<=1.05*summary['calibration']['C']['RGB_MAE'],
        training_updates=0,RL_updates=0,independent_validation=False)
    (out/'results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    with (out/'source_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(summary=summary,numeric_candidate_gate=output['numeric_candidate_gate']),indent=2))


if __name__=='__main__':main()
