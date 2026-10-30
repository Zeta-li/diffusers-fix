import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / 'runs/budget_merge_v1'
    spec = json.loads((run / 'preregistration.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    records = [json.loads(s) for s in (run / 'records.jsonl').read_text().splitlines()]
    assert status['phase'] == 'complete_analysis_pending'
    assert len(records) == spec['planned_requests'] == 58
    assert status['completed_dit'] == status['invoked_dit'] == 464
    output = run / 'analysis'; output.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    quality = []
    for case in spec['cases']:
        cid = case['case_id']; old = ROOT / 'runs/budget_student_pilot_v1/evaluation' / cid
        merged = next(r for r in records if r['case_id'] == cid and r['arm'] == 'MERGED')
        path = run / cid / f"MERGED_{merged['phase']}_r{merged['repeat']}.png"
        images = {'SOURCE': Image.open(ROOT / case['image']).convert('RGB'),
                  **{a: Image.open(old / (a+'.png')).convert('RGB') for a in ('F','ALL','LORA_ONLY')},
                  'MERGED': Image.open(path).convert('RGB')}
        assert hashlib.sha256(images['MERGED'].tobytes()).hexdigest() == merged['RGB_sha256']
        arrays = {a:np.asarray(im,dtype=np.float32)/255 for a,im in images.items() if a!='SOURCE'}
        row = dict(case_id=cid, **{a:float(np.abs(v-arrays['F']).mean()) for a,v in arrays.items() if a!='F'},
                   merged_vs_unmerged=float(np.abs(arrays['MERGED']-arrays['LORA_ONLY']).mean()))
        quality.append(row)
        panel = Image.new('RGB', (5*300, 410), 'white'); draw = ImageDraw.Draw(panel)
        draw.text((5,5),cid,fill='black',font=font)
        for i,line in enumerate(textwrap.wrap(case['prompt'],width=150)[:3]):
            draw.text((5,27+19*i),line,fill='black',font=font)
        for i,(arm,im) in enumerate(images.items()):
            draw.text((i*300+5,87),arm,fill='black',font=font)
            panel.paste(im.resize((300,300)),(i*300,110))
        panel.save(output/(cid+'.jpg'),quality=95)
    latency = {}
    for arm in ('F','UNMERGED','MERGED'):
        vals = [r['seconds'] for r in records if r['phase']=='paired' and r['arm']==arm]
        latency[arm] = dict(n=len(vals),mean=float(np.mean(vals)),p50=float(np.percentile(vals,50)),p90=float(np.percentile(vals,90)))
    quality_mean = {a:float(np.mean([r[a] for r in quality])) for a in ('ALL','LORA_ONLY','MERGED','merged_vs_unmerged')}
    diagnostics = json.loads((run/'merge_diagnostics.json').read_text())
    result = dict(actual_counts=status,quality=quality,quality_mean=quality_mean,latency=latency,
        merged_speedup_vs_F=latency['F']['mean']/latency['MERGED']['mean'],
        merged_timesaved_vs_unmerged=1-latency['MERGED']['mean']/latency['UNMERGED']['mean'],
        merged_relative_MAE_vs_unmerged=quality_mean['MERGED']/quality_mean['LORA_ONLY']-1,
        rounding_error_relative_L2=(sum(r['rounding_error_squared'] for r in diagnostics['layers'])/sum(r['delta_squared'] for r in diagnostics['layers']))**.5,
        numeric_merge_gate=quality_mean['MERGED']<=1.01*quality_mean['LORA_ONLY'],
        visual_gate='pending',timing_scope=spec['timing'])
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('quality','actual_counts')},indent=2))


if __name__ == '__main__':
    main()
