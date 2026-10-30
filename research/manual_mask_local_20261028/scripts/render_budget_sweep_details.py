import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/budget_sweep_v1'


def main():
    cases={c['case_id']:c for c in json.loads((RUN/'preregistration.json').read_text())['cases']}
    key=json.loads((RUN/'blind_review/unblinding_key.json').read_text())
    boxes={'478067':[0,160,210,360], '414492':[300,300,500,500],
        '351809':[260,350,430,490], '156301':[180,95,410,320],
        '352264':[0,70,500,330], '522661':[250,260,500,500]}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    for sid,box in boxes.items():
        cid='expand_train_'+sid;case=cases[cid]
        sheet=Image.new('RGB',(1536,1090),'white');draw=ImageDraw.Draw(sheet)
        for label,pos in [('SOURCE',(0,0)),('A',(512,0)),('B',(1024,0)),('C',(512,545)),('D',(1024,545))]:
            src=ROOT/case['image'] if label=='SOURCE' else RUN/'evaluation'/cid/(key[cid][label]+'.png')
            im=Image.open(src).convert('RGB');w,h=im.size;sw,sh=case['source_wh']
            rect=[round(box[0]*w/sw),round(box[1]*h/sh),round(box[2]*w/sw),round(box[3]*h/sh)]
            crop=im.crop(rect);crop.thumbnail((512,512))
            sheet.paste(crop,(pos[0],pos[1]+33));draw.text((pos[0]+5,pos[1]+3),cid+' '+label,font=font,fill='black')
        sheet.save(RUN/'blind_review'/(cid+'_detail.jpg'),quality=97)
    print('Six anonymous native-resolution detail sheets created; mapping not printed')


if __name__=='__main__':main()
