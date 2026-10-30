import argparse
import json
from collections import Counter

import numpy as np
import torch
from PIL import Image, ImageDraw

import run_s1 as common
from action_value_head import ActionValueHead, ACTIONS, FEATURE_NAMES
from train_action_value_head import load_trees, tree_rollout

ROOT=common.ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',required=True);p.add_argument('--fit',required=True);p.add_argument('--closed',required=True)
    args=p.parse_args();collection=ROOT/'runs'/args.collection;fit=ROOT/'runs'/args.fit;closed=ROOT/'runs'/args.closed
    if json.loads((closed/'status.json').read_text())['phase']!='complete_quality_pending':raise RuntimeError('closed run incomplete')
    spec=json.loads((collection/'preregistration.json').read_text());cs=json.loads((closed/'preregistration.json').read_text())
    trained=json.loads((fit/'result.json').read_text());ckpt=torch.load(fit/'head.pt',map_location='cpu',weights_only=True)
    trees,derived,provenance=load_trees(collection,spec);labels=json.loads((fit/'labels.json').read_text())
    if derived!=labels['rows'] or provenance!=labels['provenance']:raise RuntimeError('labels or provenance mismatch')
    fit_rows=[r for r in derived if r['role']=='fit'];x=torch.tensor([r['features'] for r in fit_rows]);mean=x.mean(0);std=x.std(0).clamp_min(1e-4)
    if not torch.allclose(mean,ckpt['mean'],atol=1e-6,rtol=2e-6) or not torch.allclose(std,ckpt['std'],atol=1e-6,rtol=2e-6):raise RuntimeError('normalization not fit-only')
    out=closed/'analysis';out.mkdir(exist_ok=True);rows=[];panels=[];replays=0;selected=trained['selected_constant']
    for case in cs['cases']:
        cid=case['case_id'];base=ROOT/'runs'/case['reference_F_run']/cid
        source=Image.open(ROOT/case['image']).convert('RGB');f=Image.open(base/'F.png').convert('RGB');fa=np.asarray(f,dtype=np.float32)/255
        x0,y0,x1,y1=[int(v/500*1024) for v in case['evaluation_box']];region=np.zeros((1024,1024),bool);region[y0:y1,x0:x1]=True
        if case['role'] in ('fit','calibration'):
            locations={'FIX32':collection/cid/'NN.png','CONTEXT64':collection/cid/'CC.png','FRESH64':collection/cid/'FF.png','CONSTANT':collection/cid/(selected+'.png')}
        else:
            locations={'FIX32':ROOT/'runs'/case['reference_FIX32_run']/cid/'FIX32.png',
                'CONTEXT64':ROOT/'runs/context_closed_v1'/cid/'CONTEXT64.png','FRESH64':ROOT/'runs/context_closed_v1'/cid/'FRESH64.png'}
            locations['CONSTANT']=locations[{'NN':'FIX32','CC':'CONTEXT64','FF':'FRESH64'}[selected]] if selected in ('NN','CC','FF') else closed/cid/'CONSTANT.png'
        locations['HEAD']=closed/cid/'HEAD.png'
        meta=json.loads((closed/cid/'HEAD.json').read_text());seq=''.join(d['action'] for d in meta['execution']['decisions'])
        if case['role'] in ('fit','calibration'):
            wanted=next(r for r in trained['selected_policy'][case['role']]['predictions'] if r['case_id']==cid)['sequence']
            if seq!=wanted:raise RuntimeError('online choice differs from tree prediction')
            raw=torch.load(closed/cid/'HEAD_raw.pt',map_location='cpu',weights_only=True)
            old=torch.load(collection/cid/(seq+'_raw.pt'),map_location='cpu',weights_only=True)
            if not all(torch.equal(a,b) for a,b in zip(raw['states'],old['states'])):raise RuntimeError('online rollout differs from action tree')
            replays+=1
        images=[source,f];names=['SOURCE','F'];measurements={}
        for name in ['FIX32','CONTEXT64','FRESH64','CONSTANT','HEAD']:
            im=Image.open(locations[name]).convert('RGB');e=np.abs(np.asarray(im,dtype=np.float32)/255-fa)
            measurements[name]=dict(mae=float(e.mean()),inside_mae=float(e[region].mean()),outside_mae=float(e[~region].mean()),
                numeric_pass=bool(e.mean()<=.015 and e[~region].mean()<=.01))
            if name in ['FIX32','CONTEXT64','CONSTANT','HEAD']:images.append(im);names.append(name)
        base_updates=meta['execution']['base_plan']['total_target_token_updates']
        added={'FIX32':0,'CONTEXT64':128,'FRESH64':128,'CONSTANT':64*sum(a!='N' for a in selected),'HEAD':meta['added_target_updates']}
        for name in measurements:measurements[name]['target_token_fraction']=(base_updates+added[name])/(8*4096)
        rows.append(dict(case_id=cid,source_group_id=case['source_group_id'],role=case['role'],head_sequence=seq,constant_sequence=selected,
            metrics=measurements,head_vs_context_gain=1-measurements['HEAD']['mae']/measurements['CONTEXT64']['mae'],
            head_vs_constant_gain=1-measurements['HEAD']['mae']/measurements['CONSTANT']['mae']))
        panel=Image.new('RGB',(6*290,315),'white');draw=ImageDraw.Draw(panel)
        for j,(im,name) in enumerate(zip(images,names)):
            panel.paste(im.resize((290,290)),(j*290,25));draw.text((j*290+3,6),case['source_group_id']+' '+name+(' '+seq if name=='HEAD' else ''),fill='black')
        panel.save(out/(cid+'_panel.jpg'),quality=95);panels.append((case['role'],panel))
    summary=[]
    for role in ('fit','calibration','seen_holdout'):
        subset=[r for r in rows if r['role']==role]
        for arm in ['FIX32','CONTEXT64','FRESH64','CONSTANT','HEAD']:
            summary.append(dict(role=role,arm=arm,n=len(subset),mean_mae=float(np.mean([r['metrics'][arm]['mae'] for r in subset])),
                mean_outside_mae=float(np.mean([r['metrics'][arm]['outside_mae'] for r in subset])),
                mean_target_token_fraction=float(np.mean([r['metrics'][arm]['target_token_fraction'] for r in subset])),
                numeric_passes=sum(r['metrics'][arm]['numeric_pass'] for r in subset)))
    hold=[r for r in rows if r['role']=='seen_holdout'];mean=lambda arm:float(np.mean([r['metrics'][arm]['mae'] for r in hold]))
    cc_gain=1-mean('HEAD')/mean('CONTEXT64');const_gain=1-mean('HEAD')/mean('CONSTANT')
    wins=sum(r['head_vs_context_gain']>0 for r in hold);worst=min(r['head_vs_context_gain'] for r in hold)
    result=dict(rows=rows,summary=summary,holdout_head_vs_context_mean_gain=cc_gain,holdout_head_vs_constant_mean_gain=const_gain,
        holdout_improved_sources=wins,holdout_worst_relative_gain=worst,
        numeric_continue_gate_pass=bool(cc_gain>=.01 and const_gain>=.01 and wins>=3 and worst>=-.1 and all(r['metrics']['HEAD']['numeric_pass'] for r in hold)),
        semantic_review_pending=True,independent_validation=False,online_fit_calibration_exact_replays=replays,
        selected_constant=selected,sequence_counts={role:dict(Counter(r['head_sequence'] for r in rows if r['role']==role)) for role in ('fit','calibration','seen_holdout')})
    for role in ('fit','calibration','seen_holdout'):
        group=[p for r,p in panels if r==role]
        for start in range(0,len(group),4):
            sheet=Image.new('RGB',(1740,315*len(group[start:start+4])),'white')
            for j,panel in enumerate(group[start:start+4]):sheet.paste(panel,(0,j*315))
            sheet.save(out/(role+'_'+str(start//4)+'.jpg'),quality=95)
    common.write(out/'metrics.json',result)
    common.write(ROOT/'data/action_fit_v1_label_audit.json',dict(status='PASS',labels=60,source_disjoint_fit_calibration=True,
        normalization_fit_only=True,online_fit_calibration_exact_replays=replays,checkpoint_sha256=common.sha(fit/'head.pt')))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows',)},indent=2))


if __name__=='__main__':main()
