import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

from PIL import Image
import torch

import run_s1 as common
from run_free_pilot import load_pipeline
from cache_aware_adapters import CacheAdapters
from local_edit_runtime import request_from_case, GeometryRefreshSelector, install_local_edit

ROOT=common.ROOT


@torch.no_grad()
def main():
    path=ROOT/'data/local_edit_runtime_latency_v1_preregistration.json'
    spec=json.loads(path.read_text());out=ROOT/'runs/local_edit_runtime_latency_v1';assert not out.exists()
    for n,h in spec['code_sha256'].items():assert common.sha(ROOT/'scripts'/n)==h,n
    for key in ('checkpoint','head','noise'):assert common.sha(ROOT/spec[key])==spec[key+'_sha256']
    out.mkdir();shutil.copy2(path,out/'preregistration.json')
    counts=dict(invoked_dit=0,completed_dit=0,successful_requests=0);active={}
    def status(phase,**kw):common.write(out/'status.json',dict(phase=phase,active=active,**counts,**kw))
    try:
        status('loading_model');pipe=load_pipeline();pipe.vae.disable_tiling();model=pipe.transformer
        adapters=CacheAdapters(model);adapters.load_state_dict(torch.load(ROOT/spec['checkpoint'],weights_only=True,map_location='cpu')['adapters'])
        selector=GeometryRefreshSelector(ROOT/spec['head'])
        noise=torch.load(ROOT/spec['noise'],weights_only=True,map_location='cuda')
        images={c['case_id']:Image.open(ROOT/c['image']).convert('RGB') for c in spec['cases']};cases={c['case_id']:c for c in spec['cases']}
        def before(*args):counts['invoked_dit']+=1
        def after(*args):counts['completed_dit']+=1
        pre=model.register_forward_pre_hook(before);post=model.register_forward_hook(after)
        hashes={};original=model.forward
        for idx,job in enumerate(spec['jobs']):
            active=dict(index=idx,**job);status('running')
            cid,arm=job['case_id'],job['arm'];c=cases[cid]
            start_count=counts['completed_dit']
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            with install_local_edit(pipe,adapters,request_from_case(c),arm,selector) as record:
                result=pipe(image=images[cid],prompt=c['prompt'],true_cfg_scale=1,num_inference_steps=8,
                    height=1024,width=1024,num_images_per_prompt=1,latents=noise.clone())
            torch.cuda.synchronize();seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
            assert model.forward==original and model._local_edit_controller_active is False
            assert record['completed'] and counts['completed_dit']-start_count==8
            im=result.images[0];digest=hashlib.sha256(im.tobytes()).hexdigest()
            expected=json.loads((ROOT/spec['evaluation_run']/'evaluation'/cid/(arm+'.json')).read_text())
            assert digest==expected['RGB_sha256'],(cid,arm)
            if (cid,arm) in hashes:assert hashes[(cid,arm)]==digest
            hashes[(cid,arm)]=digest
            folder=out/cid;folder.mkdir(exist_ok=True);stem=f"{arm}_{job['phase']}_r{job['repeat']}"
            im.save(folder/(stem+'.png'));counts['successful_requests']+=1
            row=dict(**job,seconds=seconds,peak_GiB=peak/2**30,RGB_sha256=digest,runtime_metadata=record['metadata'],**counts)
            common.write(folder/(stem+'.json'),row);common.append(out/'records.jsonl',row)
            print(json.dumps(dict(job=idx+1,arm=arm,seconds=seconds,**counts)),flush=True)
        pre.remove();post.remove()
        assert counts['invoked_dit']==counts['completed_dit']==spec['planned_dit'] and counts['successful_requests']==spec['planned_requests']
        status('complete_analysis_pending',all_outputs_match_quality=True,all_request_cache_scopes_restored=True)
    except BaseException as exc:
        status('failed',exception=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
