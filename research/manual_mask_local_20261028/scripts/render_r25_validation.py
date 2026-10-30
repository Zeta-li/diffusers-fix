import argparse
import json
from pathlib import Path
import random
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    run = ROOT / 'runs' / args.attempt
    spec = json.loads((run / 'preregistration.json').read_text())
    assert json.loads((run / 'status.json').read_text())['phase'] == 'complete_analysis_pending'
    out = run / 'blind_review'
    assert not out.exists()
    out.mkdir()
    rng = random.Random(2026103105 if args.attempt == 'r25_seed2_v1' else 2026103106)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    key = {}
    form = []
    for start in range(0, len(spec['cases']), 4):
        sheet = Image.new('RGB', (1200, 1640), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, case in enumerate(spec['cases'][start:start + 4]):
            cid = case['case_id']
            top = j * 410
            arms = list(spec['visual_arms'])
            rng.shuffle(arms)
            key[cid] = dict(zip('ABC', arms))
            draw.text((5, top + 3), cid, font=font, fill='black')
            for k, line in enumerate(textwrap.wrap(case['prompt'], width=127)[:3]):
                draw.text((5, top + 23 + 18 * k), line, font=font, fill='black')
            for k, label in enumerate(['SOURCE', 'A', 'B', 'C']):
                path = ROOT / case['image'] if k == 0 else run / 'evaluation' / cid / (key[cid][label] + '.png')
                im = Image.open(path).convert('RGB')
                sheet.paste(im.resize((300, 300)), (k * 300, top + 108))
                draw.text((k * 300 + 5, top + 87), label, font=font, fill='black')
            form.append(dict(case_id=cid, source_instruction=case['prompt'], scores={a: None for a in 'ABC'}))
        sheet.save(out / f'page{start // 4:02}.jpg', quality=96)
    (out / 'unblinding_key.json').write_text(json.dumps(key, indent=2) + '\n')
    (out / 'blank_form.json').write_text(json.dumps(form, indent=2) + '\n')
    if args.attempt == 'r25_seed2_v1':
        boxes = {'478067': [0, 160, 210, 360], '351809': [260, 350, 430, 490],
                 '156301': [180, 95, 410, 320], '352264': [0, 70, 500, 330], '522661': [250, 260, 500, 500]}
        cases = {c['case_id']: c for c in spec['cases']}
        for sid, box in boxes.items():
            cid = 'expand_train_' + sid
            case = cases[cid]
            sheet = Image.new('RGB', (1600, 450), 'white')
            draw = ImageDraw.Draw(sheet)
            for col, label in enumerate(['SOURCE', 'A', 'B', 'C']):
                path = ROOT / case['image'] if col == 0 else run / 'evaluation' / cid / (key[cid][label] + '.png')
                im = Image.open(path).convert('RGB')
                w, h = im.size
                sw, sh = case['source_wh']
                crop = im.crop([round(box[0] * w / sw), round(box[1] * h / sh), round(box[2] * w / sw), round(box[3] * h / sh)])
                crop.thumbnail((400, 410))
                sheet.paste(crop, (col * 400, 35))
                draw.text((col * 400 + 4, 5), cid + ' ' + label, font=font, fill='black')
            sheet.save(out / (cid + '_detail.jpg'), quality=97)
    print('Anonymous pages ready; mapping not printed')


if __name__ == '__main__':
    main()
