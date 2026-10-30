import collections
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/budget_expansion_train_v1'


def main():
    torch.set_num_threads(4)
    spec=json.loads((RUN/'preregistration.json').read_text())
    status=json.loads((RUN/'status.json').read_text())
    if status['phase']!='complete_analysis_pending':raise RuntimeError('Training not complete')
    output=RUN/'analysis';output.mkdir(exist_ok=True);(output/'panels').mkdir(exist_ok=True)
    jobs=[json.loads(s) for s in (RUN/'calls.jsonl').read_text().splitlines()]
    counts=collections.Counter(j['kind'] for j in jobs)
    assert len(jobs)==status['completed_dit']==status['invoked_dit']
    expected=dict(contract=spec['planned_contract_dit'],prefix=spec['planned_prefix_dit'],
        teacher=spec['planned_teacher_dit'],gradient_student=spec['planned_gradient_student_dit'],evaluation=spec['planned_evaluation_dit'])
    assert dict(counts)=={k:v for k,v in expected.items() if v}
    logs=[json.loads(s) for s in (RUN/'training.jsonl').read_text().splitlines()]
    assert len(logs)==512 and logs[-1]['optimizer_updates']==128
    assert all(r['teacher_cache_unmodified'] for r in logs)
    for name,digest in spec['code_sha256'].items():
        assert hashlib.sha256((RUN/'code_snapshot'/name).read_bytes()).hexdigest()==digest
    zero=torch.load(RUN/'checkpoints/update000.pt',map_location='cpu',weights_only=True)
    final=torch.load(RUN/'checkpoints/update128.pt',map_location='cpu',weights_only=True)
    changes={kind:dict(changed_tensors=sum(not torch.equal(zero[kind][k],final[kind][k]) for k in zero[kind]),
        total_tensors=len(zero[kind]),delta_L2=sum(float((final[kind][k].float()-v.float()).square().sum()) for k,v in zero[kind].items())**.5)
        for kind in ('adapters',)}
    records=[]
    previous_bitwise_checks=[]
    case_map={c['case_id']:c for c in spec['cases']}
    arms=spec['evaluation_arms']
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    except OSError:font=ImageFont.load_default()
    sigmas=torch.load(RUN/'conditioning'/(spec['evaluation_sources'][0]+'.pt'),map_location='cpu',weights_only=True)['sigmas']
    for cid in spec['evaluation_sources']:
        case=case_map[cid];folder=RUN/'evaluation'/cid
        images={arm:Image.open(folder/(arm+'.png')).convert('RGB') for arm in arms}
        ref=np.asarray(images['F'],dtype=np.float32)/255
        width,height=images['F'].size
        x0,y0,x1,y1=case['bbox'];sw,sh=case['source_wh']
        box=[max(0,int((x0-32)*width/sw)),max(0,int((y0-32)*height/sh)),min(width,int(np.ceil((x1+32)*width/sw))),min(height,int(np.ceil((y1+32)*height/sh)))]
        inside=np.zeros((height,width),dtype=bool);inside[box[1]:box[3],box[0]:box[2]]=True
        for arm in arms:
            meta=json.loads((folder/(arm+'.json')).read_text())
            raw=torch.load(folder/(arm+'.pt'),weights_only=True,map_location='cpu')
            assert len(raw['states'])==8 and torch.equal(raw['final'],raw['states'][-1])
            assert torch.isfinite(raw['final']).all()
            if cid in spec['historical_sources'] and arm in ('F','ALL','PILOT'):
                old_arm='LORA_ONLY' if arm=='PILOT' else arm
                previous=ROOT/'runs/budget_student_pilot_v1/evaluation'/cid/(old_arm+'.json')
                assert meta['RGB_sha256']==json.loads(previous.read_text())['RGB_sha256'],('Prior baseline differs',cid,arm)
                previous_bitwise_checks.append([cid,arm])
            digest=hashlib.sha256(images[arm].tobytes()).hexdigest()
            assert digest==meta['RGB_sha256']
            error=np.abs(np.asarray(images[arm],dtype=np.float32)/255-ref)
            records.append(dict(case_id=cid,arm=arm,RGB_MAE=float(error.mean()),
                inside_MAE=float(error[inside].mean()),outside_MAE=float(error[~inside].mean()),
                latent_MAE=meta['latent_mae_vs_F'],role='calibration' if cid in spec['calibration_sources'] else
                'historical',
                training_source=cid in spec['training_sources']))
        panel=Image.new('RGB',(5*300,414),'white');draw=ImageDraw.Draw(panel)
        draw.text((8,5),cid,fill='black',font=font)
        for i,line in enumerate(textwrap.wrap(case['prompt'],width=150)[:3]):draw.text((8,27+20*i),line,fill='black',font=font)
        show=['SOURCE','F','ALL','PILOT','EXPAND']
        for i,arm in enumerate(show):
            image=Image.open(ROOT/case['image']).convert('RGB') if arm=='SOURCE' else images[arm]
            panel.paste(image.resize((300,300)),(i*300,110))
            draw.text((i*300+5,88),arm,fill='black',font=font)
        panel.save(output/'panels'/(cid+'.png'))
    summary={}
    for scope in ['all','calibration','historical']:
        scope_rows=records if scope=='all' else [r for r in records if r['role']==scope]
        summary[scope]={}
        for arm in arms:
            rows=[r for r in scope_rows if r['arm']==arm]
            summary[scope][arm]={k:float(np.mean([r[k] for r in rows])) for k in ('RGB_MAE','inside_MAE','outside_MAE','latent_MAE')}
            summary[scope][arm]['n']=len(rows)
            if arm!='F':
                base={r['case_id']:r for r in scope_rows if r['arm']=='ALL'}
                summary[scope][arm]['RGB_wins_vs_ALL']=sum(r['RGB_MAE']<base[r['case_id']]['RGB_MAE'] for r in rows)
                base_mean=summary[scope]['ALL']['RGB_MAE']
                summary[scope][arm]['relative_RGB_change_vs_ALL']=summary[scope][arm]['RGB_MAE']/base_mean-1
    for scope in summary:
        base=summary[scope]['PILOT']['RGB_MAE']
        summary[scope]['EXPAND']['relative_RGB_change_vs_PILOT']=summary[scope]['EXPAND']['RGB_MAE']/base-1
        matching=records if scope=='all' else [r for r in records if r['role']==scope]
        pilot={r['case_id']:r['RGB_MAE'] for r in matching if r['arm']=='PILOT'}
        summary[scope]['EXPAND']['RGB_wins_vs_PILOT']=sum(r['RGB_MAE']<pilot[r['case_id']] for r in matching if r['arm']=='EXPAND')
    rng=np.random.default_rng(2026103006)
    for scope in summary:
        matching=records if scope=='all' else [r for r in records if r['role']==scope]
        pilot={r['case_id']:r['RGB_MAE'] for r in matching if r['arm']=='PILOT'}
        expanded={r['case_id']:r['RGB_MAE'] for r in matching if r['arm']=='EXPAND'}
        ids=list(pilot);a=np.array([pilot[c] for c in ids]);b=np.array([expanded[c] for c in ids])
        indices=rng.integers(0,len(ids),(10000,len(ids)))
        ratios=b[indices].mean(1)/a[indices].mean(1)-1
        summary[scope]['EXPAND']['source_bootstrap95_relative_change_vs_PILOT']=np.percentile(ratios,[2.5,97.5]).tolist()
        summary[scope]['EXPAND']['worst_source_relative_change_vs_PILOT']=float((b/a-1).max())
    result=dict(status='NUMERICAL_ANALYSIS_COMPLETE_VISUAL_PENDING',actual_counts=status,call_kind_counts=dict(counts),
        updated_weights=changes,summary=summary,records=records,
        same_state_teacher_calls=512,second_step_teacher_branch_calls=128,prior_baseline_bitwise_checks=previous_bitwise_checks,
        training_sources=128,optimizer_updates=128,backward_passes=512,grad_student_calls=640,
        training_summary=json.loads((RUN/'training_summary.json').read_text()),net_latency_claim=False,
        independent_Dev32_opened=False,test64_opened=False)
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    ordinary=[r for r in logs if not r['full_regularization']]
    ax[0].plot([r['micro']+1 for r in ordinary],[r['loss'] for r in ordinary],alpha=.65)
    ax[0].set(xlabel='Microbatch',ylabel='Training loss',title='Mixed sources/windows; not a fixed-probe curve')
    names=['ALL','PILOT','EXPAND']
    vals=[summary['calibration'][a]['RGB_MAE'] for a in names]
    ax[1].bar(names,vals);ax[1].tick_params(axis='x',rotation=25)
    ax[1].set(ylabel='Mean RGB MAE versus F',title='New Cal32 sources; smaller is closer to F')
    fig.savefig(output/'training_and_error.png',dpi=160);plt.close(fig)
    print(json.dumps({'status':'PASS','counts':dict(counts),'weights':changes,'summary':summary},ensure_ascii=False))


if __name__=='__main__':main()
