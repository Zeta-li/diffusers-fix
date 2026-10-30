import argparse
import json
from pathlib import Path
import random
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', default='budget_expansion_train_v1')
    args = parser.parse_args()
    run = ROOT / 'runs' / args.attempt
    spec = json.loads((run / 'preregistration.json').read_text())
    if json.loads((run / 'status.json').read_text())['phase'] != 'complete_analysis_pending':
        raise RuntimeError('Evaluation incomplete')
    out = run / 'blind_review'
    if out.exists():
        raise RuntimeError('Preserve blind review')
    out.mkdir()
    cases = {c['case_id']:c for c in spec['cases']}
    ids = spec.get('calibration_sources', spec['evaluation_sources'])
    rng = random.Random(2026103005 if args.attempt == "budget_expansion_train_v1" else 2026103015)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    key = {}; form = []
    for start in range(0,len(ids),4):
        sheet = Image.new('RGB',(1500,1640),'white');draw=ImageDraw.Draw(sheet)
        for j,cid in enumerate(ids[start:start+4]):
            case = cases[cid]; top=j*410
            order = spec['evaluation_arms'][:];rng.shuffle(order)
            key[cid] = {label:arm for label,arm in zip('ABCD',order)}
            draw.text((5,top+3),cid,fill='black',font=font)
            for k,line in enumerate(textwrap.wrap(case['prompt'],width=156)[:3]):
                draw.text((5,top+23+18*k),line,fill='black',font=font)
            source=Image.open(ROOT/case['image']).convert('RGB')
            sheet.paste(source.resize((300,300)),(0,top+108));draw.text((5,top+87),'SOURCE',fill='black',font=font)
            for k,(label,arm) in enumerate(key[cid].items()):
                im=Image.open(run/'evaluation'/cid/(arm+'.png')).convert('RGB')
                sheet.paste(im.resize((300,300)),((k+1)*300,top+108))
                draw.text(((k+1)*300+5,top+87),label,fill='black',font=font)
            form.append(dict(case_id=cid,source_instruction=case['prompt'],scores={a:None for a in 'ABCD'}))
        sheet.save(out/('page'+str(start//4).zfill(2)+'.jpg'),quality=96)
    (out/'unblinding_key.json').write_text(json.dumps(key,indent=2)+'\n')
    (out/'blank_form.json').write_text(json.dumps(form,indent=2)+'\n')


if __name__ == '__main__':
    main()
