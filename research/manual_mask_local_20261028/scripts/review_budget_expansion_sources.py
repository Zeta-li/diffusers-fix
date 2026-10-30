import json
import math
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'inputs/budget_expansion_v1'
    manifest = json.loads((source / 'manifest.json').read_text())
    if not manifest['complete']:
        raise RuntimeError('Acquisition not complete')
    out = source / 'review'; out.mkdir(exist_ok=True)
    prior = json.loads((ROOT / 'data/localdev16_prior_source_signatures_v1.json').read_text())['records']
    for p in (ROOT / 'inputs').glob('localdev16*/*/provenance.json'):
        d = json.loads(p.read_text())
        if 'signature' in d:
            prior.append(d['signature'])
    records = []
    prior_new = []
    for row in manifest['records']:
        sig = row['signature']; matches = []
        for index, other in enumerate(prior + prior_new):
            ah = (int(sig['average_hash'], 16) ^ int(other['average_hash'], 16)).bit_count()
            dh = (int(sig['difference_hash'], 16) ^ int(other['difference_hash'], 16)).bit_count()
            exact = sig['raw_sha256'] == other['raw_sha256'] or sig['rgb_sha256'] == other['rgb_sha256']
            near = ah <= 6 and dh <= 6 and abs(math.log(sig['width']/sig['height']/(other['width']/other['height']))) <= .02
            if exact or near or ah <= 10 or dh <= 10:
                matches.append(dict(index=index, reference=other.get('path', other.get('img_id', 'prior_signature')),
                    exact=exact, near=near, ah_distance=ah, dh_distance=dh))
        prior_new.append(dict(sig, img_id=row['img_id']))
        records.append(dict(row, duplicate_matches=matches,
            eligible_before_visual=row['geometry_eligible'] and not any(m['exact'] or m['near'] for m in matches)))
    (out / 'qc.json').write_text(json.dumps(dict(records=records, prior_signatures=len(prior),
        no_generated_images=True, test_media_opened=False), indent=2) + '\n')
    for split in ('train', 'dev'):
        selected = [r for r in records if r['official_split'] == split and r['eligible_before_visual']]
        print(split, len(selected), flush=True)
        for start in range(0, len(selected), 16):
            page = Image.new('RGB', (1280, 1440), 'white'); draw = ImageDraw.Draw(page)
            for k, row in enumerate(selected[start:start+16]):
                left, top = (k % 4)*320, (k//4)*360
                im = Image.open(ROOT / row['image']).convert('RGB').resize((300, 300))
                box = [v*300/row['source_wh'][j%2] for j,v in enumerate(row['bbox'])]
                ImageDraw.Draw(im).rectangle(box, outline='red', width=2)
                page.paste(im, (left, top))
                label = f"{start+k}: {row['img_id']} bbox={row['bbox_fraction']:.2f}"
                draw.text((left,top+302), label, fill='black')
                for j,line in enumerate(textwrap.wrap(row['instruction'],width=46)[:3]):
                    draw.text((left,top+316+j*13),line,fill='black')
            page.save(out / (split+'_'+str(start//16).zfill(2)+'.jpg'), quality=92)


if __name__ == '__main__':
    main()
