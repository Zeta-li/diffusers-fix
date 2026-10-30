import argparse
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
from trainable_budget_executor import TrainableBudgetExecutor
from cache_aware_adapters import CacheAdapters
from spatial_velocity_student import distillation_loss
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
    parser.add_argument('--protocol', default='data/budget_expansion_train_v1_preregistration.json')
    parser.add_argument('--attempt', default='budget_expansion_train_v1')
    args = parser.parse_args()
    spec_path = ROOT / args.protocol
    spec = json.loads(spec_path.read_text())
    out = ROOT / 'runs' / args.attempt
    if out.exists() or Path(args.attempt).name != args.attempt:
        raise RuntimeError('New attempt name required')
    assert len(spec['training_sources']) == 128
    assert len(spec['jobs']) == 512 and spec['optimizer_updates'] == 128
    assert not set(spec['training_sources']) & set(spec['evaluation_sources'])
    assert set(j['case_id'] for j in spec['jobs']) == set(spec['training_sources'])
    assert len(set(j['seed'] for j in spec['jobs'])) == 512
    for name, digest in spec['code_sha256'].items():
        if common.sha(ROOT/'scripts'/name) != digest: raise RuntimeError('Code changed: '+name)
    for c in spec['cases']:
        if common.sha(ROOT/c['image']) != c['source_sha256']: raise RuntimeError('Source changed')
    if common.sha(ROOT/spec['initial_checkpoint']) != spec['initial_checkpoint_sha256']: raise RuntimeError('Initial checkpoint changed')
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
        spatial = None
        initial = torch.load(ROOT/spec['initial_checkpoint'], weights_only=True, map_location='cpu')
        adapters.load_state_dict(initial['adapters'])
        initial_adapters = {k:v.clone() for k,v in initial['adapters'].items()}
        scales = initial['scales']
        del initial
        parameter_counts = dict(lora=sum(p.numel() for p in adapters.parameters()),spatial=0,
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
        common.write(out/'loss_scales.json', dict(scales=scales, inherited_from=spec['initial_checkpoint'], scope='historical training sources only'))
        optimizer=torch.optim.AdamW(adapters.parameters(),lr=spec['lora_lr'],weight_decay=spec['weight_decay'])
        def save_checkpoint(tag):
            path=out/'checkpoints'/(tag+'.pt')
            torch.save(dict(adapters=adapters.state_dict(),spatial={},
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
            z=torch.randn(noise.shape, generator=torch.Generator(device='cuda').manual_seed(job['seed']), device='cuda', dtype=noise.dtype)
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
                trainable=adapters.parameters()
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
            if counts['optimizer_updates'] in spec['save_updates'] and (job['micro']+1)%4==0:
                save_checkpoint('update'+str(counts['optimizer_updates']).zfill(3))
        common.write(out/'training_summary.json',dict(seconds=time.perf_counter()-training_started,
            max_peak_GiB=max(r['peak_GiB'] for r in micro_records),mean_micro_seconds=float(np.mean([r['seconds'] for r in micro_records])),
            optimizer_updates=counts['optimizer_updates'],microbatches=len(micro_records),parameters=parameter_counts,
            prefix_dit=counts['prefix_dit'],teacher_dit=counts['teacher_dit'],gradient_student_dit=counts['gradient_student_dit']))
        gc.collect();torch.cuda.empty_cache();pipe.vae.to('cuda')
        eval_dir=out/'evaluation';eval_dir.mkdir()
        final_adapters=adapters.state_dict()
        torch.save(dict(adapters=final_adapters,spatial={},parameters=parameter_counts,scales=scales,counts=copy.deepcopy(counts),protocol_sha256=common.sha(spec_path)),out/'checkpoints/student_inference.pt')
        with torch.no_grad():
            for cid in spec['evaluation_sources']:
                folder=eval_dir/cid;folder.mkdir()
                ref=None
                for arm in spec['evaluation_arms']:
                    active=dict(stage='evaluation',case_id=cid,arm=arm);status('evaluation')
                    train_adapters=arm in ('PILOT','EXPAND')
                    if train_adapters: adapters.load_state_dict(initial_adapters if arm=='PILOT' else final_adapters)
                    student_head=spatial if arm in ('JOINT','VELOCITY_ONLY') else None
                    executor=None
                    with adapters.mode(train_adapters):
                        if arm == 'F':
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
        if counts['completed_dit']!=expected or counts['optimizer_updates']!=spec['optimizer_updates']:
            raise RuntimeError('Final accounting differs')
        status('complete_analysis_pending',new_RL_updates=0,test64_opened=False)
    except BaseException as exc:
        status('failed',exception=repr(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
