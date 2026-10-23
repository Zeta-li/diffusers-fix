import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--runs', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cases = json.loads((root / 'cases_val16.json').read_text())['cases']
    rows = [json.loads(line) for line in args.runs.read_text().splitlines()]
    configs = ['F', 'R2', 'PRIOR', 'A', 'B']
    lookup = {(r['case_id'],r['config']):r for r in rows if r.get('status')=='complete'
        and r.get('kind')=='measurement' and r.get('repeat')==0}
    folder = root / 'figures/review_during_run'
    folder.mkdir(parents=True,exist_ok=True)
    receipts = []
    for case in cases:
        cid = case['case_id']
        if not all((cid,c) in lookup for c in configs):
            continue
        height,width = case['target_hw']
        images = [Image.open(root/case['image']).convert('RGB').resize((width,height),Image.Resampling.LANCZOS)]
        evidence = []
        for config in configs:
            row = lookup[(cid,config)]
            path = root/row['output']
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != row['output_sha256']:
                raise ValueError(f'Output digest differs: {row["run_id"]}')
            images.append(Image.open(path).convert('RGB'))
            evidence.append(dict(run_id=row['run_id'],output=row['output'],sha256=actual))
        x0,y0,x1,y1 = case['target_mask_bbox_xyxy_exclusive']
        crop = (max(0,x0-64),max(0,y0-64),min(width,x1+64),min(height,y1+64))
        for kind in ['full','crop']:
            shown = [im.resize((512,round(height*512/width)),Image.Resampling.LANCZOS) for im in images] if kind=='full' else [im.crop(crop) for im in images]
            tilew,tileh = shown[0].size
            panel = Image.new('RGB',(3*tilew,2*(tileh+32)), 'white')
            draw = ImageDraw.Draw(panel)
            for j,(image,label) in enumerate(zip(shown,['INPUT']+configs)):
                xx,yy = (j%3)*tilew,(j//3)*(tileh+32)
                draw.text((xx+8,yy+8),label,fill='black')
                panel.paste(image,(xx,yy+32))
            panel.save(folder/f'{cid}_{kind}.png')
        receipts.append(dict(case_id=cid,prompt=case['prompt'],crop_xyxy=crop,quality_repeat=0,outputs=evidence))
    (folder/'receipts.json').write_text(json.dumps(dict(source_runs=str(args.runs),source_runs_sha256=hashlib.sha256(args.runs.read_bytes()).hexdigest(),cases=receipts),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(complete_quality_cases=len(receipts),source='measurement repeat0 only; not warmup',reviewer='assistant vision, nonblind; not human subject evaluation')))


if __name__=='__main__':
    main()
