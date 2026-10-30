import json

import numpy as np
from PIL import Image, ImageDraw

import run_s1 as common

ROOT=common.ROOT


def main():
    folder=ROOT/'runs/action_final1000_v1'
    assert json.loads((folder/'status.json').read_text())['phase']=='complete_quality_pending'
    spec=json.loads((folder/'preregistration.json').read_text());out=folder/'analysis';out.mkdir(exist_ok=True)
    rows=[];panels=[]
    for c in spec['cases']:
        cid=c['case_id'];f=Image.open(ROOT/'runs'/c['reference_F_run']/cid/'F.png').convert('RGB');fa=np.asarray(f,dtype=np.float32)/255
        paths={'FIX32':ROOT/'runs'/c['reference_FIX32_run']/cid/'FIX32.png',
            'CONTEXT64':ROOT/'runs/context_closed_v1'/cid/'CONTEXT64.png',
            'SELECTED_INIT':ROOT/'runs/action_closed_v1'/cid/'HEAD.png','FINAL1000':folder/cid/'HEAD.png'}
        images=[Image.open(ROOT/c['image']).convert('RGB'),f];names=['SOURCE','F'];metrics={}
        x0,y0,x1,y1=[int(v/500*1024) for v in c['evaluation_box']];region=np.zeros((1024,1024),bool);region[y0:y1,x0:x1]=True
        for arm,path in paths.items():
            im=Image.open(path).convert('RGB');e=np.abs(np.asarray(im,dtype=np.float32)/255-fa)
            metrics[arm]=dict(mae=float(e.mean()),outside_mae=float(e[~region].mean()),inside_mae=float(e[region].mean()),numeric_pass=bool(e.mean()<=.015 and e[~region].mean()<=.01))
            images.append(im);names.append(arm)
        meta=json.loads((folder/cid/'HEAD.json').read_text());sequence=''.join(d['action'] for d in meta['execution']['decisions'])
        rows.append(dict(case_id=cid,sequence=sequence,metrics=metrics,added_target_updates=meta['added_target_updates'],
            target_token_fraction=meta['target_token_fraction'],gain_vs_context=1-metrics['FINAL1000']['mae']/metrics['CONTEXT64']['mae'],
            exactly_context=bool(np.array_equal(np.asarray(images[-1]),np.asarray(images[3])))))
        panel=Image.new('RGB',(1740,315),'white');draw=ImageDraw.Draw(panel)
        for j,(im,name) in enumerate(zip(images,names)):
            panel.paste(im.resize((290,290)),(290*j,25));draw.text((290*j+3,6),c['source_group_id']+' '+name+(' '+sequence if name=='FINAL1000' else ''),fill='black')
        panel.save(out/(cid+'_panel.jpg'),quality=95);panels.append(panel)
    mean=lambda name:float(np.mean([r['metrics'][name]['mae'] for r in rows]))
    result=dict(rows=rows,mean_MAE={name:mean(name) for name in paths},mean_gain_vs_context=1-mean('FINAL1000')/mean('CONTEXT64'),
        improved_sources=sum(r['gain_vs_context']>0 for r in rows),exactly_context_sources=sum(r['exactly_context'] for r in rows),
        numeric_passes=sum(r['metrics']['FINAL1000']['numeric_pass'] for r in rows),diagnostic_only=True,
        changes_primary_selection=False,semantic_review_pending=True)
    sheet=Image.new('RGB',(1740,1260),'white')
    for j,p in enumerate(panels):sheet.paste(p,(0,j*315))
    sheet.save(out/'overview.jpg',quality=95);common.write(out/'metrics.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
