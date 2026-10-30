import copy
import gc
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
from train_budget_student import capture_condition, euler
from cache_aware_adapters import CacheAdapters
from trainable_budget_executor import TrainableBudgetExecutor
from controlled_mask import tensor_digest

ROOT = common.ROOT


@torch.no_grad()
def main():
    path = ROOT / 'data/budget_development_v1_preregistration.json'
    spec = json.loads(path.read_text()); out = ROOT / 'runs/budget_development_v1'
    if out.exists():
        raise RuntimeError('Preserve previous development run')
    gate = json.loads((ROOT / spec['gate']).read_text())
    if not gate['open_Dev32'] or common.sha(ROOT / spec['gate']) != spec['gate_sha256']:
        raise RuntimeError('Calibration gate absent or changed')
    for n,d in spec['code_sha256'].items():
        if common.sha(ROOT/'scripts'/n) != d:
            raise RuntimeError('Code changed')
    for key in ('pilot_checkpoint','expanded_checkpoint'):
        if common.sha(ROOT/spec[key]) != spec[key+'_sha256']:
            raise RuntimeError('Checkpoint changed')
    out.mkdir(); (out/'conditioning').mkdir(); (out/'evaluation').mkdir(); (out/'code_snapshot').mkdir()
    shutil.copy2(path,out/'preregistration.json')
    for n in spec['code_sha256']:shutil.copy2(ROOT/'scripts'/n,out/'code_snapshot'/n)
    counts=dict(invoked_dit=0,completed_dit=0,successful_requests=0,conditioning_only=0)
    active={}
    def status(phase,**extra):common.write(out/'status.json',dict(phase=phase,active=active,**counts,**extra))
    try:
        status('loading_model');pipe=load_pipeline();pipe.vae.disable_tiling();model=pipe.transformer
        noise=torch.load(ROOT/spec['noise'],map_location='cuda',weights_only=True)
        prepared={};cases={c['case_id']:c for c in spec['cases']}
        for cid,case in cases.items():
            if common.sha(ROOT/case['image']) != case['source_sha256']:raise RuntimeError('Source changed')
            active=dict(case_id=cid);status('preparing_conditions')
            item=capture_condition(pipe,Image.open(ROOT/case['image']).convert('RGB'),case['prompt'],noise)
            prepared[cid]=item;torch.save(item,out/'conditioning'/(cid+'.pt'));counts['conditioning_only']+=1
        pipe.text_encoder.to('cpu');gc.collect();torch.cuda.empty_cache()
        adapters=CacheAdapters(model)
        weights={arm:torch.load(ROOT/spec[key],map_location='cpu',weights_only=True)['adapters']
                 for arm,key in [('PILOT','pilot_checkpoint'),('LOCAL128','expanded_checkpoint')]}
        for cid,case in cases.items():
            folder=out/'evaluation'/cid;folder.mkdir();item=prepared[cid]
            fields={k:v.to('cuda') if isinstance(v,torch.Tensor) else copy.deepcopy(v)
                    for k,v in item.items() if k not in ('hidden_states','sigmas','timesteps','timestep')}
            condition=item['hidden_states'][:,4096:].to('cuda');sigmas=item['sigmas'].to('cuda');times=item['timesteps'].to('cuda')
            ref=None
            for arm in spec['evaluation_arms']:
                active=dict(case_id=cid,arm=arm);status('evaluation');executor=None
                if arm in weights:adapters.load_state_dict(weights[arm])
                with adapters.mode(arm in weights):
                    call=model.forward
                    if arm!='F':executor=TrainableBudgetExecutor(model,case['plan_B']);call=executor.forward
                    z=noise.clone();states=[];started=time.perf_counter()
                    for step in range(8):
                        counts['invoked_dit']+=1
                        with adapters.mode(arm in weights and (arm != 'LOCAL128' or step not in executor.full_steps)):
                            v=call(hidden_states=torch.cat([z,condition],1),timestep=times[step].reshape(1).to(z.dtype)/1000,**fields)[0][:,:4096]
                        if not torch.isfinite(v).all():raise RuntimeError('Nonfinite velocity')
                        counts['completed_dit']+=1;z=euler(z,v,sigmas,step);states.append(z.cpu().clone())
                    if executor:executor.clear()
                    if arm=='F':ref=z.float().clone()
                    im=common.decode(pipe,z);im.save(folder/(arm+'.png'))
                    torch.save(dict(final=z.cpu(),states=states),folder/(arm+'.pt'));counts['successful_requests']+=1
                    row=dict(case_id=cid,arm=arm,latent_mae_vs_F=float((z.float()-ref).abs().mean()),
                        RGB_sha256=hashlib.sha256(im.tobytes()).hexdigest(),final_sha256=tensor_digest(z),
                        diagnostic_seconds=time.perf_counter()-started,net_latency_claim=False,**counts)
                    common.write(folder/(arm+'.json'),row);common.append(out/'evaluation.jsonl',row)
                print(json.dumps(row),flush=True)
        assert counts['completed_dit']==counts['invoked_dit']==1024 and counts['successful_requests']==128
        status('complete_analysis_pending',training_updates=0,RL_updates=0,test64_opened=False)
    except BaseException as exc:
        status('failed',exception=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':
    main()
