import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'runs/context_closed_v1';out=folder/'analysis';out.mkdir(exist_ok=True)
    spec=json.loads((folder/'preregistration.json').read_text());rows=[];panels=[]
    for case in spec['cases']:
        cid=case['case_id'];fd=ROOT/'runs'/case['reference_F_run']/cid;bd=ROOT/'runs'/case['reference_FIX32_run']/cid
        f=Image.open(fd/'F.png').convert('RGB');fa=np.asarray(f,dtype=np.float32)/255
        source=Image.open(ROOT/case['image']).convert('RGB')
        box=case['evaluation_box'];x0,y0,x1,y1=[int(v/500*1024) for v in box]
        region=np.zeros((1024,1024),bool);region[y0:y1,x0:x1]=True
        images=[source,f];names=['SOURCE','F'];budgets={}
        for arm in ['FIX32','FRESH64','CONTEXT64']:
            directory=bd if arm=='FIX32' else folder/cid
            im=Image.open(directory/(arm+'.png')).convert('RGB');a=np.asarray(im,dtype=np.float32)/255;e=np.abs(a-fa)
            metadata=json.loads((directory/(arm+'.json')).read_text())
            plan=metadata['record']['plan'] if arm=='FIX32' else metadata['execution']['plan']
            budgets[arm]=plan['target_tokens_per_step']
            rows.append(dict(case_id=cid,source_group_id=case['source_group_id'],arm=arm,
                baseline_semantic_pass=case['baseline_semantic_pass'],mae_vs_F=float(e.mean()),
                outside_mae_vs_F=float(e[~region].mean()),inside_mae_vs_F=float(e[region].mean()),
                numeric_pass=bool(float(e.mean())<=.015 and float(e[~region].mean())<=.01),
                target_token_fraction=sum(plan['target_tokens_per_step'])/(8*4096)))
            images.append(im);names.append(arm)
        assert budgets['FRESH64']==budgets['CONTEXT64']
        assert sum(budgets['CONTEXT64'])-sum(budgets['FIX32'])==128
        panel=Image.new('RGB',(5*340,365),'white');draw=ImageDraw.Draw(panel)
        for j,(im,name) in enumerate(zip(images,names)):
            panel.paste(im.resize((340,340)),(j*340,25));draw.text((j*340+3,6),case['source_group_id']+' '+name,fill='black')
        panel.save(out/(cid+'_panel.jpg'),quality=95);panels.append(panel)
    summary=[];pair=[]
    for case in spec['cases']:
        r={v['arm']:v for v in rows if v['case_id']==case['case_id']}
        pair.append(dict(case_id=case['case_id'],source_group_id=case['source_group_id'],
            base_MAE=r['FIX32']['mae_vs_F'],fresh_MAE=r['FRESH64']['mae_vs_F'],context_MAE=r['CONTEXT64']['mae_vs_F'],
            context_relative_gain=1-r['CONTEXT64']['mae_vs_F']/r['FIX32']['mae_vs_F'],
            fresh_relative_gain=1-r['FRESH64']['mae_vs_F']/r['FIX32']['mae_vs_F'],
            context_beats_fresh=r['CONTEXT64']['mae_vs_F']<r['FRESH64']['mae_vs_F']))
    for scope in ['all12','baseline_valid11']:
        for arm in ['FIX32','FRESH64','CONTEXT64']:
            rs=[v for v in rows if v['arm']==arm and (scope=='all12' or v['baseline_semantic_pass'])]
            summary.append(dict(scope=scope,arm=arm,n=len(rs),mean_MAE=float(np.mean([v['mae_vs_F'] for v in rs])),
                mean_outside_MAE=float(np.mean([v['outside_mae_vs_F'] for v in rs])),numeric_passes=sum(v['numeric_pass'] for v in rs),
                mean_target_fraction=float(np.mean([v['target_token_fraction'] for v in rs]))))
    base=next(s['mean_MAE'] for s in summary if s['scope']=='all12' and s['arm']=='FIX32')
    context=next(s['mean_MAE'] for s in summary if s['scope']=='all12' and s['arm']=='CONTEXT64')
    gain=1-context/base;wins=sum(v['context_relative_gain']>0 for v in pair);worst=min(v['context_relative_gain'] for v in pair)
    result=dict(rows=rows,paired=pair,summary=summary,context_mean_MAE_reduction=gain,context_improved_sources=wins,
        context_worst_relative_gain=worst,context_beats_fresh=sum(v['context_beats_fresh'] for v in pair),
        numeric_relative_gate_pass=bool(gain>=.02 and wins>=8 and worst>=-.10),semantic_review_pending=True,
        independent_validation=False,net_latency_measured=False,extra_tokens_per_image=128)
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for i in range(3):
        sheet=Image.new('RGB',(1700,4*365),'white')
        for j,p in enumerate(panels[i*4:i*4+4]):sheet.paste(p,(0,j*365))
        sheet.save(out/f'overview_{i}.jpg',quality=95)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','paired')},indent=2))


if __name__=='__main__':main()
