import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/budget_sweep_v1'


def main():
    torch.set_num_threads(4)
    spec = json.loads((RUN / 'preregistration.json').read_text())
    status = json.loads((RUN / 'status.json').read_text())
    assert status['completed_dit'] == status['invoked_dit'] == 1280
    assert status['successful_requests'] == 160
    out = RUN / 'analysis'; out.mkdir(exist_ok=True)
    records = []; anchor_checks = []; duplicate_checks = []; shared_prefix_checks = []; rng = np.random.default_rng(2026103021)
    for case in spec['cases']:
        cid = case['case_id']; folder = RUN / 'evaluation' / cid
        reference = np.asarray(Image.open(folder / 'F.png').convert('RGB'), dtype=np.float32)/255
        h,w = reference.shape[:2]; x0,y0,x1,y1 = case['bbox']; sw,sh = case['source_wh']
        box = [max(0,int((x0-32)*w/sw)), max(0,int((y0-32)*h/sh)),
               min(w,int(np.ceil((x1+32)*w/sw))), min(h,int(np.ceil((y1+32)*h/sh)))]
        inside = np.zeros((h,w),dtype=bool); inside[box[1]:box[3], box[0]:box[2]] = True
        ref = torch.load(folder / 'F.pt', weights_only=True, map_location='cpu')
        raws = {a:torch.load(folder/(a+'.pt'),weights_only=True,map_location='cpu') for a in ('B','B50','C')}
        for step in range(4):
            assert torch.equal(raws['B50']['states'][step],raws['C']['states'][step])
            shared_prefix_checks.append((cid,step))
        if case['plans']['B']['selected_indices_per_step']==case['plans']['B50']['selected_indices_per_step']:
            for step in range(8):assert torch.equal(raws['B']['states'][step],raws['B50']['states'][step])
            duplicate_checks.append(cid)
        for arm in spec['evaluation_arms']:
            im = Image.open(folder / (arm+'.png')).convert('RGB')
            meta = json.loads((folder / (arm+'.json')).read_text())
            assert hashlib.sha256(im.tobytes()).hexdigest() == meta['RGB_sha256']
            raw = torch.load(folder / (arm+'.pt'), weights_only=True, map_location='cpu')
            assert len(raw['states']) == 8 and torch.equal(raw['final'], raw['states'][-1])
            assert torch.isfinite(raw['final']).all()
            if arm != 'F':
                for step in (0,1):
                    assert torch.equal(ref['states'][step], raw['states'][step])
                    anchor_checks.append(dict(case_id=cid,arm=arm,step=step,bitwise_exact=True))
            error = np.abs(np.asarray(im,dtype=np.float32)/255-reference)
            plan = None if arm == 'F' else case['plans'][arm]
            records.append(dict(case_id=cid,role=case['role'],arm=arm,
                RGB_MAE=float(error.mean()), inside_MAE=float(error[inside].mean()),
                outside_MAE=float(error[~inside].mean()) if (~inside).any() else None,
                latent_MAE=float((raw['final'].float()-ref['final'].float()).abs().mean()),
                effective_local_fraction=1 if plan is None else plan['effective_local_fraction'],
                target_update_fraction=1 if plan is None else plan['total_target_token_updates']/(8*4096),
                target_updates=8*4096 if plan is None else plan['total_target_token_updates'],
                spatial_budget_increased=case['plans']['B']['effective_local_fraction']<case['plans']['B50']['effective_local_fraction']))
    summary = {}
    for role in ('train','calibration','all'):
        rows = records if role == 'all' else [r for r in records if r['role']==role]
        summary[role] = {}
        by = {arm: {r['case_id']:r for r in rows if r['arm']==arm} for arm in spec['evaluation_arms']}
        for arm in spec['evaluation_arms']:
            vals = list(by[arm].values())
            summary[role][arm] = {k: float(np.mean([r[k] for r in vals if r[k] is not None]))
                for k in ('RGB_MAE','inside_MAE','outside_MAE','latent_MAE','target_update_fraction')}
            summary[role][arm]['n'] = len(vals)
        for src,dst in (('B','B50'),('B50','C'),('B','C')):
            ids=list(by[src]); a=np.array([by[src][c]['RGB_MAE'] for c in ids]); b=np.array([by[dst][c]['RGB_MAE'] for c in ids])
            indices=rng.integers(0,len(ids),(10000,len(ids)))
            summary[role][src+'_to_'+dst]=dict(relative_change=float(b.mean()/a.mean()-1),
                improved=int((b<a-1e-10).sum()),tied=int((np.abs(b-a)<1e-10).sum()),
                bootstrap95_relative_change=np.percentile(b[indices].mean(1)/a[indices].mean(1)-1,[2.5,97.5]).tolist())
        winners=collections.Counter(); oracle=[]
        for cid in by['B']:
            best=min(('B','B50','C'),key=lambda arm:by[arm][cid]['RGB_MAE'])
            winners[best]+=1; oracle.append(by[best][cid]['RGB_MAE'])
        best_fixed=min(('B','B50','C'),key=lambda arm:summary[role][arm]['RGB_MAE'])
        subset=[c for c in by['B'] if by['B'][c]['spatial_budget_increased']]
        a=np.array([by['B'][c]['RGB_MAE'] for c in subset]);b=np.array([by['B50'][c]['RGB_MAE'] for c in subset])
        summary[role]['spatial_increased_subset']=dict(n=len(subset),B_MAE=float(a.mean()),B50_MAE=float(b.mean()),relative_change=float(b.mean()/a.mean()-1),improved=int((b<a).sum()))
        summary[role]['reference_MAE_oracle']=dict(winners=dict(winners),MAE=float(np.mean(oracle)),
            best_fixed=best_fixed,relative_change_vs_best_fixed=float(np.mean(oracle)/summary[role][best_fixed]['RGB_MAE']-1),
            deployable=False,semantic_reward=False)
    result=dict(status='NUMERICAL_COMPLETE_VISUAL_PENDING',summary=summary,records=records,
        actual_counts=status,reused_requests=32,first_two_anchor_bitwise_checks=len(anchor_checks),B50_C_first4_bitwise_checks=len(shared_prefix_checks),
        B_B50_duplicate_plan_exact_8step_cases=duplicate_checks,
        training_updates=0,RL_updates=0,net_latency_claim=False,independent_evaluation=False)
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'anchor_checks.json').write_text(json.dumps(anchor_checks,indent=2)+'\n')
    with (out/'source_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader();writer.writerows(records)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
