import argparse
import json
import shutil
import time
import traceback

import numpy as np
import torch
from PIL import Image

import run_s1 as common
from run_free_pilot import load_pipeline, make_plan
from free_region import run_free_region
from action_value_head import install_action_policy, Predictor, FEATURE_NAMES

ROOT=common.ROOT


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--attempt',required=True)
    args=p.parse_args();spec=json.loads((ROOT/args.protocol).read_text());out=ROOT/'runs'/args.attempt
    if out.exists() or '/' in args.attempt or spec['phase']!='FROZEN_BEFORE_GENERATION':raise RuntimeError('new frozen attempt required')
    for name,h in spec['code_sha256'].items():
        if common.sha(ROOT/'scripts'/name)!=h:raise RuntimeError('code changed: '+name)
    receipt=json.loads((ROOT/spec['cpu_receipt']).read_text())
    if receipt['exit_code']!=0:raise RuntimeError('CPU contracts failed')
    for name,h in receipt['code_sha256'].items():
        if common.sha(ROOT/'scripts'/name)!=h:raise RuntimeError('CPU receipt stale')
    if spec.get('checkpoint') and common.sha(ROOT/spec['checkpoint'])!=spec['checkpoint_sha256']:
        raise RuntimeError('checkpoint changed')
    planned=sum(len(c['arms']) for c in spec['cases'])
    if planned!=spec['planned_requests'] or planned*8!=spec['planned_dit']:raise RuntimeError('budget mismatch')
    out.mkdir();shutil.copy2(ROOT/args.protocol,out/'preregistration.json');(out/'code_snapshot').mkdir()
    for name in spec['code_sha256']:shutil.copy2(ROOT/'scripts'/name,out/'code_snapshot'/name)
    counts=dict(invoked_dit=0,completed_dit=0,successful_requests=0);active={}
    def status(phase,**kw):common.write(out/'status.json',dict(phase=phase,updated_unix=time.time(),**counts,**active,**kw))
    def pre(module,pos,fields):
        if pos or active['step']>=8:raise RuntimeError('unexpected call')
        counts['invoked_dit']+=1;common.append(out/'calls.jsonl',dict(event='invoked',**counts,**active))
    def post(module,pos,result):
        counts['completed_dit']+=1;common.append(out/'calls.jsonl',dict(event='completed',**counts,**active));active['step']+=1
    status('loading_model')
    try:
        pipe=load_pipeline();noise=torch.load(ROOT/'runs/s1_v2/explicit_noise.pt',weights_only=True,map_location='cpu')
        torch.save(noise,out/'explicit_noise.pt')
        common.write(out/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
            driver_gpu=torch.cuda.get_device_name(),user_confirmed_gpu='H800',scheduler=dict(pipe.scheduler.config),
            weights_receipt_sha256=common.sha('/workspace/model-weights/deployment_20261028.json')))
        predictor=Predictor(ROOT/spec['checkpoint']) if spec.get('checkpoint') else None
        bh=pipe.transformer.register_forward_pre_hook(pre,with_kwargs=True);ah=pipe.transformer.register_forward_hook(post)
        for case in spec['cases']:
            if common.sha(ROOT/case['image'])!=case['source_sha256']:raise RuntimeError('source changed')
            dst=out/case['case_id'];dst.mkdir();plan=make_plan(case,dict(kind='front',margin=32))
            ref=ROOT/'runs'/case['reference_F_run']/case['case_id']
            fdata=torch.load(ref/'F.pt',map_location='cpu',weights_only=True)
            fmeta=json.loads((ref/'F.json').read_text())['record'];fa=np.asarray(Image.open(ref/'F.png').convert('RGB'),dtype=np.float32)/255
            x0,y0,x1,y1=[int(v/500*1024) for v in case['evaluation_box']]
            region=np.zeros((1024,1024),bool);region[y0:y1,x0:x1]=True
            tree=[]
            for arm in case['arms']:
                name=arm['name'];active.update(case_id=case['case_id'],arm=name,step=0);status('running')
                sequence=arm.get('sequence')
                policy=predictor if name=='HEAD' else lambda f,k: sequence[0 if k==3 else 1]
                with install_action_policy(pipe,plan,policy,extra_tokens=64) as run:
                    result=run_free_region(pipe,Image.open(ROOT/case['image']).convert('RGB'),case['prompt'],noise.to('cuda'))
                record=pipe.last_free_region_run
                torch.save(dict(final=result.images.cpu(),states=record.step_latents,velocities=run['velocities']),dst/(name+'_raw.pt'))
                for key in ('initial_noise_sha256','reference_condition_sha256','text_sha256','text_mask_sha256'):
                    if getattr(record,key)!=fmeta[key]:raise RuntimeError('input mismatch')
                if not all(torch.equal(a,b) for a,b in zip(record.step_latents[:2],fdata['states'][:2])):raise RuntimeError('warmup mismatch')
                previous=noise
                for k,state in enumerate(record.step_latents):
                    dt=pipe.scheduler.sigmas[k+1]-pipe.scheduler.sigmas[k]
                    expected=(previous.to('cuda').float()+dt*run['velocities'][k].to('cuda')).to(state.dtype).cpu()
                    if not torch.equal(state,expected):raise RuntimeError('Euler mismatch')
                    previous=state
                added=sum(run['selected_target_per_step'])-plan['total_target_token_updates']
                if added!=sum(d['added_tokens'] for d in run['decisions']) or added>128:raise RuntimeError('budget exceeded')
                for d in run['decisions']:
                    if d['action']!='N' and not d['extra_kv_fresh']:raise RuntimeError('fresh KV failed')
                im=common.decode(pipe,result.images);im.save(dst/(name+'.png'))
                error=np.abs(np.asarray(im,dtype=np.float32)/255-fa)
                quality=dict(mae_vs_F=float(error.mean()),inside_mae_vs_F=float(error[region].mean()),outside_mae_vs_F=float(error[~region].mean()))
                controls=[]
                for control in case.get('controls',[]):
                    if ''.join(d['action'] for d in run['decisions'])!=control['sequence']:continue
                    folder=ROOT/'runs'/control['run']/case['case_id'];old=folder/control['raw']
                    raw=torch.load(old,map_location='cpu',weights_only=True)
                    if not all(torch.equal(a,b) for a,b in zip(record.step_latents,raw['states'])):raise RuntimeError('frozen control mismatch')
                    if not np.array_equal(np.asarray(im),np.asarray(Image.open(folder/control['png']))):raise RuntimeError('control PNG mismatch')
                    controls.append(control['sequence'])
                payload=dict(case_id=case['case_id'],source_group_id=case['source_group_id'],role=case['role'],arm=name,
                    quality=quality,record=record.manifest(),execution={k:v for k,v in run.items() if k!='velocities'},
                    checks=dict(all_Euler_bitwise=True,first2_F_bitwise=True,matched_archived_controls=controls),
                    added_target_updates=added,target_token_fraction=sum(run['selected_target_per_step'])/(8*4096),net_latency_claim=False)
                common.write(dst/(name+'.json'),payload);tree.append(payload)
                counts['successful_requests']+=1;common.append(out/'runs.jsonl',dict(**active,**counts,quality=quality));status('running')
            if spec['mode']=='tree_collection':
                if len(tree)!=9:raise RuntimeError('incomplete action tree')
                roots=[r['execution']['decisions'][0] for r in tree]
                groups=[roots]+[[r['execution']['decisions'][1] for r in tree if r['execution']['decisions'][0]['action']==a] for a in 'NCF']
                for group in groups:
                    if any(v['features']!=group[0]['features'] or v['state_signature']!=group[0]['state_signature'] for v in group):
                        raise RuntimeError('not a same-state action comparison')
                common.write(dst/'tree_contract.json',dict(root_branches=9,second_step_states=3,each_second_state_branches=3,
                    deterministic_shared_prefix=True,features_and_state_signatures_identical=True,feature_names=FEATURE_NAMES))
        bh.remove();ah.remove()
        if counts!=dict(invoked_dit=planned*8,completed_dit=planned*8,successful_requests=planned):raise RuntimeError('ledger mismatch')
        status('complete_quality_pending')
    except BaseException as exc:
        status('failed',error=type(exc).__name__+': '+str(exc));traceback.print_exc();raise


if __name__=='__main__':main()
