import argparse
import json
import shutil
import time
import traceback

import torch
from PIL import Image

import run_s1 as common
from run_free_pilot import load_pipeline, make_plan
from free_region import run_free_region
from context_refresh import install_context_refresh

ROOT = common.ROOT


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--attempt',required=True)
    args=parser.parse_args();protocol=ROOT/args.protocol;spec=json.loads(protocol.read_text());out=ROOT/'runs'/args.attempt
    if out.exists() or '/' in args.attempt or spec['phase']!='FROZEN_BEFORE_GENERATION':
        raise RuntimeError('frozen protocol and new attempt required')
    for name,digest in spec['code_sha256'].items():
        if common.sha(ROOT/'scripts'/name)!=digest:raise RuntimeError('code changed')
    receipt=json.loads((ROOT/spec['cpu_receipt']).read_text())
    if receipt['exit_code']!=0:raise RuntimeError('CPU contracts failed')
    planned=sum(len(c['arms']) for c in spec['cases'])
    if spec['planned_requests']!=planned or spec['planned_dit']!=planned*8:raise RuntimeError('wrong budget')
    out.mkdir();shutil.copy2(protocol,out/'preregistration.json');(out/'code_snapshot').mkdir()
    for name in spec['code_sha256']:shutil.copy2(ROOT/'scripts'/name,out/'code_snapshot'/name)
    counts=dict(invoked_dit=0,completed_dit=0,successful_requests=0);active={}
    def status(phase,**kw):common.write(out/'status.json',dict(phase=phase,updated_unix=time.time(),**counts,**active,**kw))
    def pre(module,pos,fields):
        if pos or active['step']>=8:raise RuntimeError('unexpected DiT call')
        counts['invoked_dit']+=1
        common.append(out/'calls.jsonl',dict(event='invoked',**counts,**active))
    def post(module,pos,result):
        counts['completed_dit']+=1
        common.append(out/'calls.jsonl',dict(event='completed',**counts,**active));active['step']+=1
    status('loading_model')
    try:
        pipe=load_pipeline();noise=torch.load(ROOT/'runs/s1_v2/explicit_noise.pt',weights_only=True,map_location='cpu')
        torch.save(noise,out/'explicit_noise.pt')
        common.write(out/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
            user_confirmed_gpu='H800',driver_name=torch.cuda.get_device_name(),scheduler=dict(pipe.scheduler.config),
            weights_receipt_sha256=common.sha('/workspace/model-weights/deployment_20261028.json')))
        before=pipe.transformer.register_forward_pre_hook(pre,with_kwargs=True)
        after=pipe.transformer.register_forward_hook(post)
        for case in spec['cases']:
            if common.sha(ROOT/case['image'])!=case['source_sha256']:raise RuntimeError('source changed')
            dst=out/case['case_id'];dst.mkdir()
            refs={}
            for arm,run in [('F',case['reference_F_run']),('FIX32',case['reference_FIX32_run'])]:
                path=ROOT/'runs'/run/case['case_id']
                refs[arm]=(torch.load(path/(arm+'.pt'),weights_only=True,map_location='cpu'),
                           json.loads((path/(arm+'.json')).read_text())['record'])
            plan=make_plan(case,dict(kind='front',margin=32));paired_counts=None
            for arm in case['arms']:
                name=arm['name'];active.update(case_id=case['case_id'],arm=name,step=0);status('running')
                with install_context_refresh(pipe,plan,preserve_extra_velocity=arm['preserve_extra_velocity'],
                    extra_tokens=spec['extra_tokens'],steps=spec['refresh_steps']) as run:
                    result=run_free_region(pipe,Image.open(ROOT/case['image']).convert('RGB'),case['prompt'],noise.to('cuda'))
                record=pipe.last_free_region_run
                torch.save(dict(final=result.images.cpu(),states=record.step_latents,velocities=run['velocities']),dst/(name+'_raw.pt'))
                for _,old in refs.values():
                    for key in ('initial_noise_sha256','reference_condition_sha256','text_sha256','text_mask_sha256'):
                        if getattr(record,key)!=old[key]:raise RuntimeError('paired input mismatch')
                for count,ref in [(2,refs['F'][0]),(3,refs['FIX32'][0])]:
                    if not all(torch.equal(a,b) for a,b in zip(record.step_latents[:count],ref['states'][:count])):
                        raise RuntimeError('prefix differs before intervention')
                previous=noise
                for k,state in enumerate(record.step_latents):
                    dt=pipe.scheduler.sigmas[k+1]-pipe.scheduler.sigmas[k]
                    expected=(previous.to('cuda').float()+dt*run['velocities'][k].to('cuda')).to(state.dtype).cpu()
                    if not torch.equal(expected,state):raise RuntimeError('Euler changed')
                    previous=state
                selected_counts=run['plan']['target_tokens_per_step']
                if paired_counts is not None and selected_counts!=paired_counts:raise RuntimeError('paired token budget mismatch')
                paired_counts=selected_counts
                if run['plan']['total_target_token_updates']!=plan['total_target_token_updates']+128:
                    raise RuntimeError('unexpected additional token budget')
                if not all(g['extra_kv_fresh'] for g in run['gates']):raise RuntimeError('extra KV not refreshed')
                if arm['preserve_extra_velocity'] and not all(g['extra_velocity_retained'] for g in run['gates']):
                    raise RuntimeError('gate failure')
                common.decode(pipe,result.images).save(dst/(name+'.png'))
                counts['successful_requests']+=1
                common.write(dst/(name+'.json'),dict(record=record.manifest(),
                    execution={k:v for k,v in run.items() if k!='velocities'},
                    checks=dict(first2_F_bitwise=True,first3_FIX32_bitwise=True,all_Euler_bitwise=True,
                                exact_added_tokens=128),net_latency_claim=False))
                common.append(out/'runs.jsonl',dict(**active,**counts));status('running')
        before.remove();after.remove()
        if counts!=dict(invoked_dit=planned*8,completed_dit=planned*8,successful_requests=planned):raise RuntimeError('final ledger mismatch')
        status('generation_complete_quality_pending')
    except BaseException as exc:
        status('failed',error=type(exc).__name__+': '+str(exc));traceback.print_exc();raise


if __name__=='__main__':main()
