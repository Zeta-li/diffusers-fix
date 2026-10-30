import argparse
import contextlib
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from PIL import Image
import torch

import run_s1 as common
from run_free_pilot import load_pipeline
from budget_profiles import make_budget_plan
from budget_cache_executor import BudgetCacheExecutor
from trainable_budget_executor import TrainableBudgetExecutor
from cache_aware_adapters import CacheAdapters
from spatial_velocity_student import SpatialVelocityStudent, region_groups, distillation_loss
from controlled_mask import tensor_digest

ROOT = common.ROOT


class ConditioningCaptured(Exception):
    pass


@torch.no_grad()
def capture_condition(pipe, image, prompt, noise):
    captured = {}
    original = pipe.transformer.forward
    def capture(*args, **kwargs):
        if args: raise RuntimeError('Unexpected positional input')
        for k, v in kwargs.items():
            captured[k] = v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
        raise ConditioningCaptured()
    pipe.transformer.forward = capture
    try:
        pipe(image=image, prompt=prompt, true_cfg_scale=1, num_inference_steps=8,
             latents=noise.clone(), height=1024, width=1024, output_type='latent')
    except ConditioningCaptured:
        pass
    finally:
        pipe.transformer.forward = original
    if not captured: raise RuntimeError('No conditioning captured')
    captured['sigmas'] = pipe.scheduler.sigmas.detach().cpu().clone()
    captured['timesteps'] = pipe.scheduler.timesteps.detach().cpu().clone()
    return captured


def euler(state, velocity, sigmas, step):
    return (state.float() + (sigmas[step + 1] - sigmas[step]) * velocity).to(state.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='data/budget_student_pilot_v1_preregistration.json')
    parser.add_argument('--attempt', default='budget_student_pilot_v1')
    args = parser.parse_args()
    spec_path = ROOT / args.protocol
    spec = json.loads(spec_path.read_text())
    out = ROOT / 'runs' / args.attempt
    if out.exists() or Path(args.attempt).name != args.attempt:
        raise RuntimeError('New attempt name required')
    for name, digest in spec['code_sha256'].items():
        if common.sha(ROOT/'scripts'/name) != digest: raise RuntimeError('Code changed: '+name)
    for c in spec['cases']:
        if common.sha(ROOT/c['image']) != c['source_sha256']: raise RuntimeError('Source changed')
    if common.sha(ROOT/spec['noise']) != spec['noise_sha256']: raise RuntimeError('Noise changed')
    out.mkdir(parents=True)
    (out/'code_snapshot').mkdir(); (out/'conditioning').mkdir(); (out/'checkpoints').mkdir()
    shutil.copy2(spec_path, out/'preregistration.json')
    for name in spec['code_sha256']: shutil.copy2(ROOT/'scripts'/name, out/'code_snapshot'/name)
    counts = dict(invoked_dit=0, completed_dit=0, teacher_dit=0, prefix_dit=0,
                  gradient_student_dit=0, contract_dit=0, evaluation_dit=0,
                  optimizer_updates=0, backward_passes=0, successful_requests=0, conditioning_only=0)
    active = {}
    def status(phase, **extra):
        common.write(out/'status.json',dict(phase=phase, updated_unix=time.time(), active=active, **counts, **extra))
    def counted(call, kind, *a, **kw):
        counts['invoked_dit'] += 1
        common.append(out/'calls.jsonl',dict(event='invoked',kind=kind,active=active,call=counts['invoked_dit']))
        value = call(*a, **kw)[0]
        if not torch.isfinite(value).all(): raise RuntimeError('Nonfinite velocity')
        counts['completed_dit'] += 1; counts[kind+'_dit'] += 1
        return value[:, :4096]
    pipe = None
    try:
        status('loading_model')
        pipe = load_pipeline(); pipe.vae.disable_tiling()
        model = pipe.transformer
        noise = torch.load(ROOT/spec['noise'],weights_only=True,map_location='cuda')
        prepared = {}
        cases = {c['case_id']:c for c in spec['cases']}
        for cid, case in cases.items():
            active = dict(case_id=cid,stage='conditioning_only'); status('preparing_conditions')
            item = capture_condition(pipe,Image.open(ROOT/case['image']).convert('RGB'),case['prompt'],noise)
            if not torch.equal(item['hidden_states'][:,:4096],noise.cpu()): raise RuntimeError('Condition noise differs')
            torch.save(item,out/'conditioning'/(cid+'.pt'))
            prepared[cid] = item
            counts['conditioning_only'] += 1
        pipe.text_encoder.to('cpu'); pipe.vae.to('cpu'); gc.collect(); torch.cuda.empty_cache()
        torch.manual_seed(spec['initialization_seed'])
        torch.cuda.manual_seed_all(spec['initialization_seed'])
        adapters = CacheAdapters(model,spec['lora_rank'])
        spatial = SpatialVelocityStudent(width=spec['spatial_width'],blocks=spec['spatial_blocks']).to('cuda')
        parameter_counts = dict(lora=sum(p.numel() for p in adapters.parameters()),spatial=sum(p.numel() for p in spatial.parameters()),
            frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad))
        common.write(out/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
            driver_gpu=torch.cuda.get_device_name(),user_confirmed_gpu='H800',parameters=parameter_counts,
            training_encoders_on_cpu=True,inference_timing_claim=False))
        def inputs(cid):
            item = prepared[cid]
            fields = {k:(v.to('cuda') if isinstance(v,torch.Tensor) else copy.deepcopy(v))
                      for k,v in item.items() if k not in ('hidden_states','sigmas','timesteps','timestep')}
            condition = item['hidden_states'][:,4096:].to('cuda')
            return fields, condition, item['sigmas'].to('cuda'), item['timesteps'].to('cuda')
        def call_step(call, kind, z, condition, fields, times, step):
            return counted(call,kind,hidden_states=torch.cat([z,condition],1),
                           timestep=times[step].reshape(1).to(z.dtype)/1000,**fields)
        def trajectory(cid, call, kind):
            fields,condition,sigmas,times=inputs(cid)
            z=noise.clone();states=[];velocities=[]
            for step in range(8):
                v=call_step(call,kind,z,condition,fields,times,step)
                z=euler(z,v,sigmas,step);states.append(z.detach().cpu().clone());velocities.append(v.detach().cpu().clone())
            counts['successful_requests']+=1
            return z,states,velocities
        with torch.no_grad():
            for cid in spec['diagnostic_sources'][:2]:
                active=dict(stage='contracts',case_id=cid);status('contracts')
                f,fs,fv=trajectory(cid,model.forward,'contract')
                full=TrainableBudgetExecutor(model,make_budget_plan(cases[cid],'FULL'),spatial=spatial)
                ff,fss,fvv=trajectory(cid,full.forward,'contract');full.clear()
                if not all(torch.equal(a,b) for a,b in zip(fs+fv,fss+fvv)):raise RuntimeError('Native/full contract failed')
                old=BudgetCacheExecutor(model,cases[cid]['plan_B'])
                b,bs,bv=trajectory(cid,old.forward,'contract');old.clear()
                new=TrainableBudgetExecutor(model,cases[cid]['plan_B'],spatial=spatial)
                bb,bss,bvv=trajectory(cid,new.forward,'contract');new.clear()
                if not all(torch.equal(a,b) for a,b in zip(bs+bv,bss+bvv)):raise RuntimeError('Cached/zero student contract failed')
                prior=ROOT/'runs/budget_preflight_v1/vae'/cid/'F_untiled_r0.pt'
                saved=torch.load(prior,weights_only=True,map_location='cpu')
                if not torch.equal(f.cpu(),saved['final']):raise RuntimeError('Direct loop differs from previous native pipeline')
                torch.save(dict(F=fs,ALL=bs),out/(cid+'_contract.pt'))
        sums=[0.,0.,0.]; sizes=[0,0,0]
        for cid in spec['scale_sources']:
            raw=torch.load(ROOT/'runs/budget_preflight_v1/vae'/cid/'F_untiled_r0.pt',weights_only=True,map_location='cpu')
            for step in range(2,7):
                active_mask=torch.zeros(1,4096,1);active_mask[:,cases[cid]['plan_B']['selected_indices_per_step'][step]]=1
                for k,group in enumerate(region_groups(active_mask,(64,64))):
                    value=raw['velocities'][step][:,:4096][:,group].float()
                    sums[k]+=float(value.square().sum());sizes[k]+=value.numel()
        scales=[max((s/n)**.5,1e-3) for s,n in zip(sums,sizes)]
        common.write(out/'loss_scales.json',dict(scales=scales,sums=sums,counts=sizes,sources=spec['scale_sources']))
        optimizer=torch.optim.AdamW([dict(params=adapters.parameters(),lr=spec['lora_lr']),
            dict(params=spatial.parameters(),lr=spec['spatial_lr'])],weight_decay=spec['weight_decay'])
        def save_checkpoint(tag):
            path=out/'checkpoints'/(tag+'.pt')
            torch.save(dict(adapters=adapters.state_dict(),spatial={k:v.detach().cpu().clone() for k,v in spatial.state_dict().items()},
                optimizer=optimizer.state_dict(),counts=copy.deepcopy(counts),scales=scales,parameters=parameter_counts,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),protocol_sha256=common.sha(spec_path)),path)
            return path
        save_checkpoint('update000')
        optimizer.zero_grad(set_to_none=True)
        micro_records=[]
        training_started=time.perf_counter()
        for job in spec['jobs']:
            active=dict(stage='training',**job);status('training')
            fields,condition,sigmas,times=inputs(job['case_id'])
            executor=TrainableBudgetExecutor(model,cases[job['case_id']]['plan_B'],spatial=spatial)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
            z=noise.clone()
            with torch.no_grad():
                for step in range(job['start']):
                    v=call_step(executor.forward,'prefix',z,condition,fields,times,step)
                    z=euler(z,v,sigmas,step)
            executor.detach_window(); z=z.detach()
            cache_before={k:tuple(v._version for v in kv) for k,kv in executor.layer_cache.items()}
            teacher_start=tensor_digest(z)
            with torch.no_grad(),adapters.mode(False):
                tv=call_step(model.forward,'teacher',z,condition,fields,times,job['start'])
                teacher_end=euler(z,tv,sigmas,job['start'])
                if job['window']==2:
                    tv2=call_step(model.forward,'teacher',teacher_end,condition,fields,times,job['start']+1)
                    teacher_end=euler(teacher_end,tv2,sigmas,job['start']+1)
            if cache_before!={k:tuple(v._version for v in kv) for k,kv in executor.layer_cache.items()}:
                raise RuntimeError('Teacher modified student cache')
            if teacher_start!=tensor_digest(z):raise RuntimeError('Teacher modified student state')
            sv=call_step(executor.forward,'gradient_student',z,condition,fields,times,job['start'])
            loss,parts=distillation_loss(sv,tv,z,float(sigmas[job['start']]),executor.last_active,(64,64),scales)
            if job['full_regularization']:
                loss=0.1*loss
            end=euler(z,sv,sigmas,job['start'])
            roll=loss.detach()*0
            if job['window']==2:
                sv2=call_step(executor.forward,'gradient_student',end,condition,fields,times,job['start']+1)
                end=euler(end,sv2,sigmas,job['start']+1)
                roll=(end.float()-teacher_end.float()).abs().mean()/sum(scales)*3
                loss=loss+0.5*roll
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            (loss/spec['accumulation']).backward();counts['backward_passes']+=1
            grad_norm=None
            if (job['micro']+1)%spec['accumulation']==0:
                trainable=adapters.parameters()+list(spatial.parameters())
                grad_norm=float(torch.nn.utils.clip_grad_norm_(trainable,spec['grad_clip'],error_if_nonfinite=True))
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise RuntimeError('Frozen base has gradient')
                optimizer.step();optimizer.zero_grad(set_to_none=True);counts['optimizer_updates']+=1
            torch.cuda.synchronize()
            row=dict(**job,loss=float(loss.detach()),roll=float(roll.detach()),parts=parts,grad_norm=grad_norm,
                seconds=time.perf_counter()-started,peak_GiB=torch.cuda.max_memory_allocated()/2**30,
                same_start_sha256=teacher_start,teacher_cache_unmodified=True,**counts)
            common.append(out/'training.jsonl',row);micro_records.append(row)
            print(json.dumps(dict(micro=job['micro']+1,updates=counts['optimizer_updates'],loss=row['loss'],
                seconds=row['seconds'],peak_GiB=row['peak_GiB'],actual_dit=counts['completed_dit'])),flush=True)
            executor.clear()
            del z,sv,tv,end,teacher_end,loss,executor
            if job['window']==2:del sv2,tv2
            if counts['optimizer_updates'] in (5,10,15,20,25) and (job['micro']+1)%4==0:
                save_checkpoint('update'+str(counts['optimizer_updates']).zfill(3))
        common.write(out/'training_summary.json',dict(seconds=time.perf_counter()-training_started,
            max_peak_GiB=max(r['peak_GiB'] for r in micro_records),mean_micro_seconds=float(np.mean([r['seconds'] for r in micro_records])),
            optimizer_updates=counts['optimizer_updates'],microbatches=len(micro_records),parameters=parameter_counts,
            prefix_dit=counts['prefix_dit'],teacher_dit=counts['teacher_dit'],gradient_student_dit=counts['gradient_student_dit']))
        gc.collect();torch.cuda.empty_cache();pipe.vae.to('cuda')
        eval_dir=out/'evaluation';eval_dir.mkdir()
        with torch.no_grad():
            for cid in spec['evaluation_sources']:
                folder=eval_dir/cid;folder.mkdir()
                ref=None
                for arm in spec['evaluation_arms']:
                    active=dict(stage='evaluation',case_id=cid,arm=arm);status('evaluation')
                    train_adapters=arm in ('JOINT','LORA_ONLY','FULL_STUDENT')
                    student_head=spatial if arm in ('JOINT','VELOCITY_ONLY') else None
                    executor=None
                    with adapters.mode(train_adapters):
                        if arm in ('F','FULL_STUDENT'):
                            call=model.forward
                        else:
                            executor=TrainableBudgetExecutor(model,cases[cid]['plan_B'],
                                mode='fresh' if arm=='FRESH_B' else 'all',spatial=student_head)
                            call=executor.forward
                        started=time.perf_counter()
                        final,states,vels=trajectory(cid,call,'evaluation')
                        if executor:executor.clear()
                        if arm=='F':ref=final.detach().float().clone()
                        final_mae=float((final.float()-ref).abs().mean())
                        im=common.decode(pipe,final)
                        im.save(folder/(arm+'.png'))
                        torch.save(dict(final=final.cpu(),states=states),folder/(arm+'.pt'))
                        record=dict(case_id=cid,arm=arm,latent_mae_vs_F=final_mae,diagnostic_seconds=time.perf_counter()-started,
                            net_latency_claim=False,final_sha256=tensor_digest(final),RGB_sha256=hashlib.sha256(im.tobytes()).hexdigest(),**counts)
                        common.write(folder/(arm+'.json'),record);common.append(out/'evaluation.jsonl',record)
                    print(json.dumps(dict(evaluation=cid,arm=arm,latent_mae=final_mae,actual_dit=counts['completed_dit'])),flush=True)
                    del final,states,vels
        expected=spec['planned_prefix_dit']+spec['planned_teacher_dit']+spec['planned_gradient_student_dit']+spec['planned_contract_dit']+spec['planned_evaluation_dit']
        if counts['completed_dit']!=expected or counts['optimizer_updates']!=25:
            raise RuntimeError('Final accounting differs')
        status('complete_analysis_pending',new_RL_updates=0,test64_opened=False)
    except BaseException as exc:
        status('failed',exception=repr(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
