import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/'runs/object_selectivity_analysis_v1'
    out.mkdir(exist_ok=True)
    spec = json.loads((ROOT/'data/object_selectivity_fixed32_v1_preregistration.json').read_text())
    rows, panels = [], []
    for case in spec['cases']:
        cid = case['case_id']
        images = [Image.open(ROOT/case['image']).convert('RGB'),
                  Image.open(ROOT/'runs/free_full20_v1'/case['original_case_id']/'F.png').convert('RGB')]
        labels = ['SOURCE', 'OLD F']
        for seed in (1, 2):
            fdir = ROOT/'runs'/('object_selectivity_v1' if seed == 1 else 'object_selectivity_seed2_v1')/cid
            ldir = ROOT/'runs'/('object_selectivity_fixed32_v1' if seed == 1 else 'object_selectivity_seed2_v1')/cid
            f = Image.open(fdir/'F.png').convert('RGB')
            local = Image.open(ldir/'FIX32.png').convert('RGB')
            fa, la = np.asarray(f, dtype=np.float32)/255, np.asarray(local, dtype=np.float32)/255
            box = case['bbox']
            box = [max(0, box[0]-32), max(0, box[1]-32), min(500, box[2]+32), min(500, box[3]+32)]
            x0,y0,x1,y1 = [int(v/500*1024) for v in box]
            region = np.zeros((1024,1024), bool);region[y0:y1,x0:x1] = True
            e = np.abs(fa-la)
            info = json.loads((ldir/'FIX32.json').read_text())
            plan = info['record']['plan']
            mae, outside = float(e.mean()), float(e[~region].mean())
            rows.append(dict(case_id=cid,seed_index=seed,mae_vs_F=mae,outside_fixed32_box_mae_vs_F=outside,
                numeric_pass=mae<=.015 and outside<=.01, evaluation_box_source_xyxy=box,
                first_two_states_bitwise_F=info['checks']['first_two_states_bitwise_native'],
                target_token_fraction=plan['total_target_token_updates']/(8*4096)))
            images.extend([f,local]);labels.extend([f'SEED{seed} F',f'SEED{seed} FIX32'])
        panel=Image.new('RGB',(6*300,325),'white');draw=ImageDraw.Draw(panel)
        for i,(im,label) in enumerate(zip(images,labels)):
            panel.paste(im.resize((300,300)),(i*300,25));draw.text((i*300+3,5),case['source_group_id']+' '+label,fill='black')
        panel.save(out/(cid+'_panel.jpg'),quality=95);panels.append(panel)
    canvas=Image.new('RGB',(1800,3*325),'white')
    for i,p in enumerate(panels):canvas.paste(p,(0,i*325))
    canvas.save(out/'overview.jpg',quality=95)
    result=dict(rows=rows,numeric_passes=sum(r['numeric_pass'] for r in rows),n=len(rows),
                mean_mae_vs_F=float(np.mean([r['mae_vs_F'] for r in rows])),
                semantic_review_pending=True, independent_new_sources=False, net_latency_measured=False)
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
