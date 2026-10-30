import collections
import csv
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]


def main():
    run=ROOT/'runs/refresh_bandit_v1'
    status=json.loads((run/'status.json').read_text())
    assert status['completed_dit']==status['invoked_dit']==1280 and status['RL_updates']==16
    cases=[];records=[];folders={}
    for name in ('refresh_timing_pilot8_v1','refresh_timing_remaining40_v1'):
        old=ROOT/'runs'/name
        cases.extend(json.loads((old/'preregistration.json').read_text())['cases'])
        records.extend(json.loads((old/'analysis/results.json').read_text())['records'])
        folders.update({p.name:p for p in (old/'evaluation').iterdir()})
    val=[json.loads(l) for l in (run/'calibration.jsonl').read_text().splitlines()]
    eps=[json.loads(l) for l in (run/'episodes.jsonl').read_text().splitlines()]
    assert len(val)==32 and len(eps)==128 and all(r['matched_frozen_action'] and r['all8states_match'] for r in val+eps)
    out=run/'analysis';out.mkdir(exist_ok=True)
    summary={};rng=np.random.default_rng(2026103021)
    for role in ('train','calibration'):
        ids=[c['case_id'] for c in cases if c['role']==role]
        by={a:[next(r for r in records if r['case_id']==cid and r['arm']==a) for cid in ids] for a in ('B','R4','R25','R6')}
        if role=='calibration':
            for a in ('SUP','RL'):
                by[a]=[next(r for r in val if r['case_id']==cid and r['policy']==a) for cid in ids]
        result={a:dict(n=len(ids),RGB_MAE=float(np.mean([r['RGB_MAE'] for r in rs])),
            outside_MAE=float(np.mean([r['outside_MAE'] or 0 for r in rs])),
            surrogate_loss=float(np.mean([r['RGB_MAE']+0.25*(r['outside_MAE'] or 0) for r in rs]))) for a,rs in by.items()}
        fixed=min(('R4','R25','R6'),key=lambda a:result[a]['surrogate_loss'])
        mat=np.array([[r['RGB_MAE']+0.25*(r['outside_MAE'] or 0) for r in by[a]] for a in ('R4','R25','R6')]).T
        result['oracle']=dict(winners=dict(collections.Counter([('R4','R25','R6')[i] for i in mat.argmin(1)])),
            loss=float(mat.min(1).mean()),relative_vs_best_fixed=float(mat.min(1).mean()/result[fixed]['surrogate_loss']-1),best_fixed=fixed,deployable=False)
        if role=='calibration':
            for a in ('SUP','RL'):
                b=np.array([r['RGB_MAE'] for r in by['R25']]);v=np.array([r['RGB_MAE'] for r in by[a]])
                ix=rng.integers(0,len(ids),(10000,len(ids)))
                result[a].update(relative_RGB_vs_R25=float(v.mean()/b.mean()-1),wins_vs_R25=int((v<b-1e-10).sum()),
                    ties_vs_R25=int((np.abs(v-b)<1e-10).sum()),bootstrap95_relative_RGB_vs_R25=np.percentile(v[ix].mean(1)/b[ix].mean(1)-1,[2.5,97.5]).tolist(),
                    actions=dict(collections.Counter(r['action'] for r in by[a])))
        summary[role]=result
    result=dict(status=status,summary=summary,all160_rollouts_match_selected_action=True,
        trajectory_checks=1280,RL_episodes=128,RL_updates=16,training_action_counts=dict(collections.Counter(r['action'] for r in eps)),
        train_reward_by_round={str(i):float(np.mean([r['reward'] for r in eps if r['round']==i])) for i in range(4)},
        semantic_review='pending',new_net_latency_requests=0,independent_validation=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'calibration_metrics.csv').open('w') as f:
        keys=['case_id','policy','action','RGB_MAE','outside_MAE','surrogate_loss','reward']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in val)
    review=out/'review';review.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    subset=[c for c in cases if c['role']=='calibration']
    for page in range(4):
        canvas=Image.new('RGB',(1280,1160),'white');d=ImageDraw.Draw(canvas)
        for row,c in enumerate(subset[page*4:page*4+4]):
            y=row*290;cid=c['case_id'];folder=folders[cid]
            for col,arm in enumerate(('SOURCE','F','R25','R4','R6')):
                im=Image.open(ROOT/c['image'] if arm=='SOURCE' else folder/(arm+'.png')).convert('RGB')
                im.thumbnail((248,228));canvas.paste(im,(col*256+(256-im.width)//2,y+22))
                d.text((col*256+4,y+2),arm,fill='black',font=font)
            prompt=c.get('original_instruction',c['prompt'])
            d.text((5,y+252),cid+' | '+prompt[:145],fill='black',font=font)
        canvas.save(review/f'calibration_{page+1}.jpg',quality=94)
    print(json.dumps({k:v for k,v in result.items() if k!='status'},indent=2))


if __name__=='__main__':main()
