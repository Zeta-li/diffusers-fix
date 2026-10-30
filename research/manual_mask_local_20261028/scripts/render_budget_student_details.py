from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/budget_student_pilot_v1'


def main():
    spec=json.loads((RUN/'preregistration.json').read_text());cases={c['case_id']:c for c in spec['cases']}
    fonts=['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/System/Library/Fonts/Supplemental/Arial.ttf']
    font=next((ImageFont.truetype(p,22) for p in fonts if Path(p).exists()),ImageFont.load_default())
    for cid,box,label in [('free_275197',(20,80,390,730),'towel_detail'),('free_475277',(150,0,850,600),'mane_detail'),('local_train_15953_chocolate',(220,200,850,850),'cupcake_detail')]:
        arms=['SOURCE','F','ALL','LORA_ONLY','FRESH_B'];tiles=[]
        for a in arms:
            path=ROOT/cases[cid]['image'] if a=='SOURCE' else RUN/'evaluation'/cid/(a+'.png')
            im=Image.open(path).convert('RGB').resize((1024,1024)).crop(box);im.thumbnail((315,430));tiles.append(im)
        panel=Image.new('RGB',(1600,max(im.height for im in tiles)+45),'white');draw=ImageDraw.Draw(panel)
        for i,(a,im) in enumerate(zip(arms,tiles)):
            panel.paste(im,(i*320+(320-im.width)//2,40));draw.text((i*320+12,8),a,fill='black',font=font)
        panel.save(RUN/'analysis'/(label+'.png'))


if __name__=='__main__':main()
