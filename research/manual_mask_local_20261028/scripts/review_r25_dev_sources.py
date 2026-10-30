import json
import math
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'inputs/r25_dev_candidates_v1'
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['complete'] and len(manifest['records']) == 64
    out = source / 'review'
    assert not out.exists()
    out.mkdir()
    prior = json.loads((ROOT / 'data/localdev16_prior_source_signatures_v1.json').read_text())['records']
    for p in (ROOT / 'inputs').glob('localdev16*/*/provenance.json'):
        d = json.loads(p.read_text())
        if 'signature' in d:
            prior.append(d['signature'])
    old = json.loads((ROOT / 'inputs/budget_expansion_v1/manifest.json').read_text())
    prior.extend(dict(r['signature'], img_id=r['img_id']) for r in old['records'])
    records = []
    new_prior = []
    for row in manifest['records']:
        sig = row['signature']
        matches = []
        for index, other in enumerate(prior + new_prior):
            ah = (int(sig['average_hash'], 16) ^ int(other['average_hash'], 16)).bit_count()
            dh = (int(sig['difference_hash'], 16) ^ int(other['difference_hash'], 16)).bit_count()
            exact = sig['raw_sha256'] == other['raw_sha256'] or sig['rgb_sha256'] == other['rgb_sha256']
            near = ah <= 6 and dh <= 6 and abs(math.log(sig['width'] / sig['height'] / (other['width'] / other['height']))) <= .02
            if exact or near or ah <= 10 or dh <= 10:
                matches.append(dict(index=index, reference=other.get('path', other.get('img_id', 'prior_signature')),
                    exact=exact, near=near, ah_distance=ah, dh_distance=dh))
        new_prior.append(dict(sig, img_id=row['img_id']))
        records.append(dict(row, duplicate_matches=matches, eligible_before_visual=row['geometry_eligible'] and not matches))
    selected = [r for r in records if r['eligible_before_visual']]
    (out / 'qc.json').write_text(json.dumps(dict(records=records, prior_signatures=len(prior),
        conservative_gray_rejection=True, no_generated_images=True, test_media_opened=False), indent=2) + '\n')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    for start in range(0, len(selected), 12):
        sheet = Image.new('RGB', (1280, 1140), 'white')
        draw = ImageDraw.Draw(sheet)
        for k, row in enumerate(selected[start:start + 12]):
            left, top = (k % 4) * 320, (k // 4) * 380
            im = Image.open(ROOT / row['image']).convert('RGB').resize((300, 300))
            box = [v * 300 / row['source_wh'][j % 2] for j, v in enumerate(row['bbox'])]
            ImageDraw.Draw(im).rectangle(box, outline='red', width=2)
            sheet.paste(im, (left, top))
            draw.text((left, top + 302), f"{start+k}: {row['img_id']} bbox={row['bbox_fraction']:.2f}", font=font, fill='black')
            for j, line in enumerate(textwrap.wrap(row['instruction'], width=39)[:4]):
                draw.text((left, top + 320 + j * 14), line, font=font, fill='black')
        sheet.save(out / f'source_page{start//12:02}.jpg', quality=95)
    print(json.dumps(dict(total=64, eligible_before_visual=len(selected), prior_signatures=len(prior), selected_ids=[x['img_id'] for x in selected])))


if __name__ == '__main__':
    main()
