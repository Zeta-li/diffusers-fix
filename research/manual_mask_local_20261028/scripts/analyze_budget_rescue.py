import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    run = ROOT / 'runs/budget_rescue_v1'; spec = json.loads((run / 'preregistration.json').read_text())
    status = json.loads((run / 'status.json').read_text()); assert status['completed_dit'] == 512
    ref_run = ROOT / spec['reference_run']; out = run / 'analysis'; out.mkdir(exist_ok=True)
    old = json.loads((ref_run / 'analysis/results.json').read_text())
    records = [r for r in old['records'] if r['role'] == 'calibration']
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    (out / 'panels').mkdir(exist_ok=True)
    for case in spec['cases']:
        cid = case['case_id']; folder = run / 'evaluation' / cid
        ref = np.asarray(Image.open(ref_run / 'evaluation' / cid / 'F.png').convert('RGB'), dtype=np.float32)/255
        h,w = ref.shape[:2]; sw,sh = case['source_wh']; x0,y0,x1,y1 = case['bbox']
        box = [max(0,int((x0-32)*w/sw)), max(0,int((y0-32)*h/sh)), min(w,int(np.ceil((x1+32)*w/sw))), min(h,int(np.ceil((y1+32)*h/sh)))]
        inside = np.zeros((h,w),bool); inside[box[1]:box[3],box[0]:box[2]] = True
        for arm in spec['arms']:
            name = arm['name']; meta = json.loads((folder / (name+'.json')).read_text())
            im = Image.open(folder / (name+'.png')).convert('RGB')
            assert hashlib.sha256(im.tobytes()).hexdigest() == meta['RGB_sha256']
            raw = torch.load(folder / (name+'.pt'), map_location='cpu', weights_only=True)
            assert len(raw['states']) == 8 and torch.equal(raw['final'],raw['states'][-1])
            error = abs(np.asarray(im,dtype=np.float32)/255-ref)
            records.append(dict(case_id=cid, arm=name, role='calibration', RGB_MAE=float(error.mean()),
                                inside_MAE=float(error[inside].mean()), outside_MAE=float(error[~inside].mean()), latent_MAE=meta['latent_mae_vs_F']))
        panel = Image.new('RGB',(1800,360),'white'); draw=ImageDraw.Draw(panel)
        draw.text((5,3),cid+'  '+case['original_instruction'],fill='black',font=font)
        for i,name in enumerate(['SOURCE','F','PILOT','EXPAND','EARLY32','LOCAL128']):
            path = ROOT / case['image'] if name=='SOURCE' else (ref_run if name in ['F','PILOT','EXPAND'] else run)/'evaluation'/cid/(name+'.png')
            panel.paste(Image.open(path).convert('RGB').resize((300,300)),(300*i,60)); draw.text((300*i+5,37),name,fill='black',font=font)
        panel.save(out/'panels'/(cid+'.jpg'),quality=96)
    baseline={r['case_id']:r['RGB_MAE'] for r in records if r['arm']=='PILOT'}
    rng=np.random.default_rng(2026103007); summary={}
    for arm in ['ALL','PILOT','EXPAND','EARLY32','LOCAL128']:
        rows=[r for r in records if r['arm']==arm]; a=np.array([baseline[r['case_id']] for r in rows]); b=np.array([r['RGB_MAE'] for r in rows]); idx=rng.integers(0,32,(10000,32))
        summary[arm]={k:float(np.mean([r[k] for r in rows])) for k in ['RGB_MAE','inside_MAE','outside_MAE','latent_MAE']}
        summary[arm].update(wins_vs_PILOT=int((b<a).sum()),relative_RGB_change_vs_PILOT=float(b.mean()/a.mean()-1),
                            source_bootstrap95_vs_PILOT=np.percentile(b[idx].mean(1)/a[idx].mean(1)-1,[2.5,97.5]).tolist())
    result=dict(status='NUMERIC_COMPLETE_VISUAL_PENDING',scope='exploratory Cal32 only',summary=summary,records=records,actual_counts=status,Dev32_opened=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(summary))


if __name__ == '__main__':
    main()
