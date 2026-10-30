import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import torch
from PIL import Image

import run_s1 as common
from run_free_pilot import load_pipeline
from budget_sweep_profiles import make_budget_plan
from trainable_budget_executor import TrainableBudgetExecutor
from cache_aware_adapters import CacheAdapters

ROOT=common.ROOT


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--attempt',required=True)
    args=parser.parse_args();spec=json.loads((ROOT/args.protocol).read_text());out=ROOT/'runs'/args.attempt
    if out.exists():raise RuntimeError('Preserve existing attempt')
    for n,d in spec['code_sha256'].items():
        if common.sha(ROOT/'scripts'/n)!=d:raise RuntimeError('Code changed')
    for key in ('expanded_checkpoint',):
        if common.sha(ROOT/spec[key])!=spec[key+'_sha256']:raise RuntimeError('Checkpoint changed')
    out.mkdir();shutil.copy2(ROOT/args.protocol,out/'preregistration.json');(out/'code_snapshot').mkdir()
    for n in spec['code_sha256']:shutil.copy2(ROOT/'scripts'/n,out/'code_snapshot'/n)
    counts=dict(invoked_dit=0,completed_dit=0,successful_requests=0)
    active={}
    def status(phase,**extra):common.write(out/'status.json',dict(phase=phase,active=active,**counts,**extra))
    try:
        status('loading_model');pipe=load_pipeline();pipe.vae.disable_tiling()
        model=pipe.transformer;adapters=CacheAdapters(model)
        assert common.sha(ROOT/'runs/s1_v2/explicit_noise.pt')==spec['noise_sha256']
        for c in spec['cases']:assert common.sha(ROOT/c['image'])==c['source_sha256']
        weights={arm:torch.load(ROOT/spec[key],weights_only=True,map_location='cpu')['adapters']
            for arm,key in [('B','expanded_checkpoint'),('B50','expanded_checkpoint'),('C','expanded_checkpoint')]}
        noise=torch.load(ROOT/'runs/s1_v2/explicit_noise.pt',weights_only=True,map_location='cuda')
        images={c['case_id']:Image.open(ROOT/c['image']).convert('RGB') for c in spec['cases']}
        cases={c['case_id']:c for c in spec['cases']}
        def before(*args):counts['invoked_dit']+=1
        def after(*args):counts['completed_dit']+=1
        pre=model.register_forward_pre_hook(before);post=model.register_forward_hook(after)
        original=model.forward;hashes={}
        for index,job in enumerate(spec['jobs']):
            active=dict(index=index,**job);status('running')
            cid,arm=job['case_id'],job['arm'];case=cases[cid]
            if arm in weights:adapters.load_state_dict(weights[arm])
            with adapters.mode(arm in weights):
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
                executor=None
                if arm!='F':executor=TrainableBudgetExecutor(model,make_budget_plan(case,arm))
                if executor:
                    if arm in ('B','B50','C'):
                        def gated_forward(*a, **kw):
                            with adapters.mode(executor.step not in executor.full_steps):
                                return executor.forward(*a, **kw)
                        model.forward = gated_forward
                    else:
                        model.forward=executor.forward
                try:
                    result=pipe(image=images[cid],prompt=case['prompt'],true_cfg_scale=1,num_inference_steps=8,
                        num_images_per_prompt=1,height=1024,width=1024,latents=noise.clone())
                    if executor and executor.step!=8:raise RuntimeError('Incomplete request')
                finally:
                    model.forward=original
                    if executor:executor.clear()
                torch.cuda.synchronize();seconds=time.perf_counter()-started;peak=torch.cuda.max_memory_allocated()
            im=result.images[0];digest=hashlib.sha256(im.tobytes()).hexdigest();key=(cid,arm)
            expected=json.loads((ROOT/spec['evaluation_run']/'evaluation'/cid/(arm+'.json')).read_text())
            if digest!=expected['RGB_sha256']:raise RuntimeError('Full pipeline differs from evaluated candidate')
            if key in hashes and hashes[key]!=digest:raise RuntimeError('Repeat output changed')
            hashes[key]=digest
            folder=out/cid;folder.mkdir(exist_ok=True)
            stem=arm+'_'+job['phase']+'_r'+str(job['repeat'])
            im.save(folder/(stem+'.png'));counts['successful_requests']+=1
            row=dict(**job,seconds=seconds,peak_GiB=peak/2**30,RGB_sha256=digest,**counts)
            common.write(folder/(stem+'.json'),row);common.append(out/'records.jsonl',row)
            print(json.dumps(dict(job=index+1,arm=arm,seconds=seconds,**counts)),flush=True)
        pre.remove();post.remove()
        assert counts['completed_dit']==spec['planned_DiT']
        status('complete_analysis_pending')
    except BaseException as exc:
        status('failed',exception=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
