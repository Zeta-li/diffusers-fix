import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'runs/localdev16_v1'
    spec = json.loads((folder / 'preregistration.json').read_text())
    status = json.loads((folder / 'status.json').read_text())
    if status['phase'] != 'generation_complete_quality_pending' or status['successful_requests'] != 64:
        raise RuntimeError('Generation incomplete')
    out = folder / 'analysis'
    out.mkdir(exist_ok=True)
    rows, paired, panels = [], [], []
    for case in spec['cases']:
        cid = case['case_id']
        d = folder / cid
        source = Image.open(ROOT / case['image']).convert('RGB')
        ref = Image.open(d / 'F.png').convert('RGB')
        fa = np.asarray(ref, dtype=np.float64) / 255
        x0,y0,x1,y1 = [int(v / 500 * 1024) for v in case['evaluation_box']]
        inside = np.zeros((1024,1024), dtype=bool)
        inside[y0:y1,x0:x1] = True
        current = {}
        images = [source, ref]
        for arm in ('FIX32','FRESH64','CONTEXT64'):
            image = Image.open(d / (arm + '.png')).convert('RGB')
            e = np.abs(np.asarray(image, dtype=np.float64) / 255 - fa)
            meta = json.loads((d / (arm + '.json')).read_text())
            row = dict(case_id=cid, source_group_id=case['source_group_id'], task_type=case['task_type'],
                arm=arm, mae_vs_F=float(e.mean()), outside_mae_vs_F=float(e[~inside].mean()),
                inside_mae_vs_F=float(e[inside].mean()),
                numeric_pass=bool(e.mean() <= .015 and e[~inside].mean() <= .01),
                target_token_fraction=sum(meta['target_tokens_per_step']) / (8*4096))
            rows.append(row)
            current[arm] = row
            images.append(image)
        b,c,f = [current[a]['mae_vs_F'] for a in ('FIX32','CONTEXT64','FRESH64')]
        paired.append(dict(case_id=cid, source_group_id=case['source_group_id'], task_type=case['task_type'],
            fix_MAE=b, context_MAE=c, fresh_MAE=f, context_relative_gain=1-c/b,
            fresh_relative_gain=1-f/b, context_beats_fresh=c<f))
        panel = Image.new('RGB', (5*384, 414), 'white')
        draw = ImageDraw.Draw(panel)
        for j,(im,name) in enumerate(zip(images,('SOURCE','F','FIX32','FRESH64','CONTEXT64'))):
            panel.paste(im.resize((384,384)),(j*384,30))
            draw.text((j*384+5,8),case['source_group_id']+' '+name, fill='black')
        panel.save(out / (cid + '_panel.jpg'), quality=96)
        panels.append(panel)
    summary = []
    for scope in ('all16','attribute','replace','expansion'):
        for arm in ('FIX32','FRESH64','CONTEXT64'):
            rs = [r for r in rows if r['arm']==arm and (scope=='all16' or r['task_type']==scope)]
            summary.append(dict(scope=scope,arm=arm,n=len(rs),
                mean_MAE=float(np.mean([r['mae_vs_F'] for r in rs])),
                mean_outside_MAE=float(np.mean([r['outside_mae_vs_F'] for r in rs])),
                numeric_passes=sum(r['numeric_pass'] for r in rs),
                mean_target_fraction=float(np.mean([r['target_token_fraction'] for r in rs]))))
    total = {s['arm']:s for s in summary if s['scope']=='all16'}
    gain = 1-total['CONTEXT64']['mean_MAE']/total['FIX32']['mean_MAE']
    wins = sum(p['context_relative_gain']>0 for p in paired)
    worst = min(p['context_relative_gain'] for p in paired)
    numeric_gate = gain>=.02 and wins>=11 and worst>=-.10 and total['CONTEXT64']['numeric_passes']>=15
    result = dict(rows=rows,paired=paired,summary=summary,context_mean_MAE_reduction=gain,
        context_improved_sources=wins,context_worst_relative_gain=worst,
        context_beats_fresh=sum(p['context_beats_fresh'] for p in paired),
        numeric_gate_pass=bool(numeric_gate),semantic_review_pending=True,
        independent_frozen_source_validation=True,net_latency_measured=False)
    (out / 'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out / 'metrics.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for k in range(4):
        sheet=Image.new('RGB',(1920,4*414),'white')
        for j,panel in enumerate(panels[k*4:k*4+4]):sheet.paste(panel,(0,j*414))
        sheet.save(out / f'overview_{k}.jpg',quality=95)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','paired')},indent=2))


if __name__ == '__main__':
    main()
