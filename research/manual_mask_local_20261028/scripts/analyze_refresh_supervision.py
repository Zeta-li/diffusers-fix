import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import run_s1 as common
from refresh_supervision import distance_map

ROOT = common.ROOT


def main():
    run = ROOT/'runs/refresh_supervision_v1'
    out = run/'analysis'
    out.mkdir(exist_ok=True)
    spec = json.loads((run/'preregistration.json').read_text())
    state_rows, action_rows = [], []
    for case in spec['cases']:
        dst = run/case['case_id']
        plan = json.loads((dst/'FIX32_plan.json').read_text())
        panels = []
        for step in spec['teacher_steps']:
            d = torch.load(dst/f'step{step}.pt', map_location='cpu', weights_only=True)
            e = d['error_map'].numpy()
            outside = np.ones(4096, bool)
            outside[d['selected']] = False
            near = distance_map(plan['core_indices'], [64, 64]).numpy() <= 4
            top = np.argsort(-e)[:410]
            warmup = (d['warmup_clean'].float()-d['condition'].float()).square().mean(-1)[0].sqrt().numpy()
            inactive = np.flatnonzero(outside)
            k = max(1, int(np.ceil(len(inactive)*.1)))
            ranked = inactive[np.argsort(-warmup[inactive])[:k]]
            oracle = inactive[np.argsort(-e[inactive])[:k]]
            denom = float(e[inactive].sum())
            row = dict(case_id=case['case_id'], step=step, velocity_mse=d['velocity_mse'],
                update_mse=d['update_mse'], inactive_count=int(outside.sum()),
                inactive_error_share=float(e[outside].sum()/e.sum()),
                core_and_4token_ring_error_share=float(e[near].sum()/e.sum()),
                top10percent_error_share=float(e[top].sum()/e.sum()),
                inactive_warmup_top10_error_capture=float(e[ranked].sum()/denom),
                inactive_oracle_top10_error_capture=float(e[oracle].sum()/denom))
            state_rows.append(row)
            for a in d['actions']:
                extra = a['extra']
                own_gain = float(a['update_gain_map'][extra].sum()/4096)
                total_gain = a['update_gain']
                action_rows.append(dict(case_id=case['case_id'], step=step, action=a['action'],
                    extra_tokens=a['additional_tokens'], baseline_update_mse=d['update_mse'],
                    update_mse=a['update_mse'], update_gain=total_gain,
                    update_gain_fraction=total_gain/d['update_mse'], velocity_gain=a['velocity_gain'],
                    refreshed_region_gain=own_gain, spillover_gain=total_gain-own_gain))
            v=np.log1p(e/max(float(np.median(e)), 1e-12))
            v=np.clip(v/max(float(np.quantile(v,.99)), 1e-12),0,1).reshape(64,64)
            heat=np.stack([v, .12*v, 1-v],-1)
            heat=(heat*255).astype('uint8')
            im=Image.fromarray(heat).resize((256,256),Image.Resampling.NEAREST)
            canvas=Image.new('RGB',(256,280),'white');canvas.paste(im,(0,24))
            draw=ImageDraw.Draw(canvas);draw.text((4,5),case['source_group_id']+' step '+str(step),fill='black')
            for a in d['actions']:
                if a['action']=='ERROR':
                    for idx in a['extra']:
                        y,x=divmod(idx,64);draw.rectangle([4*x,24+4*y,4*x+3,24+4*y+3],outline='white')
            panels.append(canvas)
        sheet=Image.new('RGB',(5*256,280),'white')
        for j,p in enumerate(panels):sheet.paste(p,(j*256,0))
        sheet.save(out/(case['case_id']+'_errors.png'))
    summary=[]
    for name in spec['candidates']:
        rows=[r for r in action_rows if r['action']==name]
        summary.append(dict(action=name,n=len(rows),positive_update_gain=sum(r['update_gain']>0 for r in rows),
            mean_update_gain_fraction=float(np.mean([r['update_gain_fraction'] for r in rows])),
            min_update_gain_fraction=min(r['update_gain_fraction'] for r in rows),
            max_update_gain_fraction=max(r['update_gain_fraction'] for r in rows)))
    pair=[]
    for case in spec['cases']:
        for step in spec['probe_steps']:
            d={r['action']:r for r in action_rows if r['case_id']==case['case_id'] and r['step']==step}
            pair.append(dict(case_id=case['case_id'],step=step,
                error_beats_random=d['ERROR']['update_gain']>d['RANDOM']['update_gain'],
                error_beats_near=d['ERROR']['update_gain']>d['NEAR']['update_gain'],
                winner=max(d,key=lambda k:d[k]['update_gain'])))
    result=dict(state_rows=state_rows,action_rows=action_rows,summary=summary,paired=pair,
        error_beats_random=sum(p['error_beats_random'] for p in pair),
        error_beats_near=sum(p['error_beats_near'] for p in pair),
        mean_inactive_error_share=float(np.mean([r['inactive_error_share'] for r in state_rows])),
        mean_core_ring_error_share=float(np.mean([r['core_and_4token_ring_error_share'] for r in state_rows])),
        mean_top10_error_share=float(np.mean([r['top10percent_error_share'] for r in state_rows])),
        mean_warmup_capture=float(np.mean([r['inactive_warmup_top10_error_capture'] for r in state_rows])),
        mean_oracle_capture=float(np.mean([r['inactive_oracle_top10_error_capture'] for r in state_rows])),
        final_image_gain_measured=False,net_latency_measured=False,
        caveat='Eight correlated states from four seen fit sources; teacher oracle costs are not inference savings')
    result['diagnostic_signal_gate_pass']=result['error_beats_random']>=6 and next(s for s in summary if s['action']=='ERROR')['positive_update_gain']>=6
    common.write(out/'metrics.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('state_rows','action_rows','paired')},indent=2))


if __name__ == '__main__':
    main()
