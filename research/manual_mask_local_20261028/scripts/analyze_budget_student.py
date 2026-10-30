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
RUN=ROOT/'runs/budget_student_pilot_v1'


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
    assert dict(counts)==expected
    logs=[json.loads(s) for s in (RUN/'training.jsonl').read_text().splitlines()]
    assert len(logs)==100 and logs[-1]['optimizer_updates']==25
    assert all(r['teacher_cache_unmodified'] for r in logs)
    for name,digest in spec['code_sha256'].items():
        assert hashlib.sha256((RUN/'code_snapshot'/name).read_bytes()).hexdigest()==digest
    zero=torch.load(RUN/'checkpoints/update000.pt',map_location='cpu',weights_only=True)
    final=torch.load(RUN/'checkpoints/update025.pt',map_location='cpu',weights_only=True)
    changes={kind:dict(changed_tensors=sum(not torch.equal(zero[kind][k],final[kind][k]) for k in zero[kind]),
        total_tensors=len(zero[kind]),delta_L2=sum(float((final[kind][k].float()-v.float()).square().sum()) for k,v in zero[kind].items())**.5)
        for kind in ('adapters','spatial')}
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
            if cid in spec['diagnostic_sources'] and arm in ('F','ALL'):
                previous=ROOT/'runs/budget_preflight_v1/main'/cid/(arm+'_untiled_r0.pt')
                prior=torch.load(previous,map_location='cpu',weights_only=True)
                assert torch.equal(raw['final'],prior['final']),('Prior baseline differs',cid,arm)
                previous_bitwise_checks.append([cid,arm])
            digest=hashlib.sha256(images[arm].tobytes()).hexdigest()
            assert digest==meta['RGB_sha256']
            error=np.abs(np.asarray(images[arm],dtype=np.float32)/255-ref)
            records.append(dict(case_id=cid,arm=arm,RGB_MAE=float(error.mean()),
                inside_MAE=float(error[inside].mean()),outside_MAE=float(error[~inside].mean()),
                latent_MAE=meta['latent_mae_vs_F'],role='calibration' if cid in spec['calibration_sources'] else
                'internal_validation' if cid in spec['internal_validation_sources'] else 'seen_diagnostic',
                training_source=cid in spec['training_sources']))
        panel=Image.new('RGB',(8*256,370),'white');draw=ImageDraw.Draw(panel)
        draw.text((8,5),cid,fill='black',font=font)
        for i,line in enumerate(textwrap.wrap(case['prompt'],width=190)[:3]):draw.text((8,27+20*i),line,fill='black',font=font)
        show=['SOURCE','F','ALL','JOINT','VELOCITY_ONLY','LORA_ONLY','FRESH_B','FULL_STUDENT']
        for i,arm in enumerate(show):
            image=Image.open(ROOT/case['image']).convert('RGB') if arm=='SOURCE' else images[arm]
            panel.paste(image.resize((256,256)),(i*256,110))
            draw.text((i*256+5,88),arm,fill='black',font=font)
        panel.save(output/'panels'/(cid+'.png'))
    summary={}
    for scope in ['all','seen_diagnostic','calibration','internal_validation']:
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
    result=dict(status='NUMERICAL_ANALYSIS_COMPLETE_VISUAL_PENDING',actual_counts=status,call_kind_counts=dict(counts),
        updated_weights=changes,summary=summary,records=records,
        same_state_teacher_calls=100,second_step_teacher_branch_calls=25,prior_baseline_bitwise_checks=previous_bitwise_checks,
        training_sources=12,optimizer_updates=25,backward_passes=100,grad_student_calls=125,
        training_summary=json.loads((RUN/'training_summary.json').read_text()),net_latency_claim=False,
        contracts=dict(real_sources=2,real_contract_DiT=64,CPU_tests=3,CPU_tiny_DiT=77),
        inference_ablation_note='same joint checkpoint;not separately trained velocity-only/LoRA-only models')
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    ordinary=[r for r in logs if not r['full_regularization']]
    ax[0].plot([r['micro']+1 for r in ordinary],[r['loss'] for r in ordinary],alpha=.65)
    ax[0].set(xlabel='Microbatch',ylabel='Training loss',title='Mixed sources/windows; not a fixed-probe curve')
    names=['ALL','JOINT','VELOCITY_ONLY','LORA_ONLY','FRESH_B','FULL_STUDENT']
    vals=[summary['all'][a]['RGB_MAE'] for a in names]
    ax[1].bar(names,vals);ax[1].tick_params(axis='x',rotation=25)
    ax[1].set(ylabel='Mean RGB MAE versus F',title='15 seen development sources; smaller is closer to F')
    fig.savefig(output/'training_and_error.png',dpi=160);plt.close(fig)
    print(json.dumps({'status':'PASS','counts':dict(counts),'weights':changes,'summary':summary},ensure_ascii=False))


if __name__=='__main__':main()
