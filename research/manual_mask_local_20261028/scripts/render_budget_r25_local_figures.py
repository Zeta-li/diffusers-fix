import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/budget_r25_v1'


def main():
    folder = RUN / 'blind_review'
    assert (folder / 'freeze_receipt.json').exists()
    key = json.loads((folder / 'unblinding_key.json').read_text())
    out = RUN / 'analysis'
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 21)
    positions = {'A': (512, 0), 'B': (1024, 0), 'C': (512, 545), 'D': (1024, 545)}
    sheet = Image.new('RGB', (1500, 730), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, (sid, title) in enumerate([
        ('351809', 'Barking: R25 retains the open-mouth repair'),
        ('478067', 'Failure retained: local candidates lack an unambiguous unicorn horn'),
    ]):
        cid = 'expand_train_' + sid
        page = Image.open(folder / (cid + '_detail.jpg'))
        draw.text((8, row * 365 + 5), sid + ' | ' + title, font=font, fill='black')
        # These are display panels derived from the previously reviewed anonymous JPEGs.
        # Native inference PNGs remain unchanged in the remote raw archive.
        source_size = (170, 140) if sid == '351809' else (210, 200)
        sw, sh = source_size
        source = page.crop((0, 33, sw, 33 + sh))
        source.thumbnail((300, 290))
        sheet.paste(source, (0, row * 365 + 68))
        draw.text((8, row * 365 + 39), 'SOURCE', font=font, fill='black')
        for col, arm in enumerate(('F', 'B', 'R25', 'C'), 1):
            label = next(k for k, value in key[cid].items() if value == arm)
            x, y = positions[label]
            w, h = ((348, 286) if sid == '351809' else (430, 410))
            crop = page.crop((x, y + 33, x + w, y + 33 + h))
            crop.thumbnail((300, 290))
            sheet.paste(crop, (col * 300, row * 365 + 68))
            draw.text((col * 300 + 8, row * 365 + 39), arm, font=font, fill='black')
    sheet.save(out / 'semantic_repair_and_failure.jpg', quality=96)
    print('Created labeled display figure after score freeze and unblinding')


if __name__ == '__main__':
    main()
