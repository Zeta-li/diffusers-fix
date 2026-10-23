import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline

from calibration_collector import CalibrationCollector,digest
from moments import save_npz


ROOT=Path(__file__).resolve().parents[1]
RESEARCH=ROOT.parent
PILOT=RESEARCH/'lightning8_preflight_20261022'
PREVIOUS=RESEARCH/'local_executor_boundary_20261022'
sys.path.insert(0,str(PILOT/'scripts'))
from run_pilot import BASE,LORA,LORA_FILE,SHIFT3,SEED


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')


def update(phase,**values):
    value=dict(phase=phase,time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),**values)
    save(ROOT/'status.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['engineering','train8'],required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    for folder in ['data','outputs','moments','latents']:(ROOT/folder).mkdir(exist_ok=True)
    ledger=ROOT/'data'/f'runs_{args.phase}.jsonl'
    existing=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    if existing and not args.resume:raise RuntimeError('Existing ledger requires explicit --resume')
    done={r['case_id']:r for r in existing if r['status']=='complete'}
    for row in done.values():
        assert sha(ROOT/row['output'])==row['output_sha256'] and sha(ROOT/row['moments'])==row['moments_sha256']
    if args.phase=='engineering':
        c=json.loads((PREVIOUS/'cases.json').read_text())['cases'][0]
        ref=next(json.loads(line) for line in (PREVIOUS/'data/runs.jsonl').read_text().splitlines()
            if json.loads(line).get('run_id')==f"R2__{c['case_id']}__measurement_0")
        region=json.loads((PREVIOUS/'masks/regions.json').read_text())['cases'][c['case_id']]['regions']['R2']['indices']
        cases=[dict(case_id=c['case_id'],image=str(PREVIOUS/c['image']),prompt=c['prompt'],indices=region,
            target_hw=ref['target_hw'],condition_count=ref['condition_count'],reference_sha256=ref['output_sha256'],
            latent=str(PILOT/'latents'/f"{c['case_id']}.pt"))]
        manifest_hash=sha(PREVIOUS/'cases.json')
    else:
        manifest=ROOT/'cases_train8.json'
        data=json.loads(manifest.read_text())
        cases=data['cases']
        if len(cases)!=8 or len({c['case_id'] for c in cases})!=8:raise ValueError('A frozen eight-source-image manifest is required')
        manifest_hash=sha(manifest)
        assets=json.loads((ROOT/'data/train8_assets.json').read_text())
        if assets.get('status')!='PASS' or assets.get('ready_for_calibration') is not True:raise ValueError('Dataset extraction/QC must pass before Train8')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    attempt=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    env=dict(phase=args.phase,attempt=attempt,seed=SEED,dtype='bfloat16',cfg=1,steps=8,baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),physical_gpu='H800 per user',
        packages={n:importlib.metadata.version(n) for n in ['torch','diffusers','transformers','peft','numpy']},
        data_manifest_sha256=manifest_hash,scripts={p.name:sha(p) for p in (ROOT/'scripts').glob('*.py')},
        planned_requests=len(cases),planned_student_forwards=len(cases)*8,planned_dense_shadow_forwards=len(cases)*6,
        formal_training_images=0 if args.phase=='engineering' else 8,student='uncorrected fixed core',sample_count=128,
        sampling='Fixed per-image, per-layer inactive indices reused across six shadow steps and first anchor; seed 2026102302',
        timing_scope='Diagnostic collection including oracle, hashes, checks, GPU reductions and CPU transfer; not deployment latency')
    save(ROOT/'data'/f'environment_{args.phase}_{attempt}.json',env)
    update('loading_model',phase_name=args.phase)
    started=time.perf_counter()
    pipe=QwenImageEditPipeline.from_pretrained(BASE,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    pipe.vae.enable_tiling();pipe.transformer.set_attention_backend('native');pipe.set_progress_bar_config(disable=True)
    for module in [pipe.transformer,pipe.text_encoder,pipe.vae]:module.eval()
    pipe.load_lora_weights(LORA,weight_name=LORA_FILE,adapter_name='lightning8',local_files_only=True)
    pipe.fuse_lora(adapter_names=['lightning8'],safe_fusing=True);pipe.unload_lora_weights()
    assert not any('lora_' in n for n,_ in pipe.transformer.named_parameters())
    torch.cuda.synchronize()
    env['load_fuse_seconds']=time.perf_counter()-started
    env['layers']=len(pipe.transformer.transformer_blocks);env['width']=pipe.transformer.inner_dim
    save(ROOT/'data'/f'environment_{args.phase}_{attempt}.json',env)
    collector=CalibrationCollector(pipe.transformer)
    collector.install()
    base_forward=collector.forward
    def forward(*a,**kw):
        collector.sigmas=pipe.scheduler.sigmas
        if collector.condition_count is None:
            collector.condition_count=kw['hidden_states'].shape[1]-collector.target_count
            collector.conditions=torch.arange(collector.target_count,kw['hidden_states'].shape[1],device=kw['hidden_states'].device)
        return base_forward(*a,**kw)
    pipe.transformer.forward=forward
    for case in cases:
        cid=case['case_id']
        if cid in done:continue
        started=time.perf_counter()
        update('collecting',phase_name=args.phase,case_id=cid,completed=len(done))
        image_path=Path(case['image'])
        if not image_path.is_absolute():image_path=ROOT/image_path
        image=Image.open(image_path).convert('RGB')
        height,width=case['target_hw'];n=(height//16)*(width//16)
        if case.get('latent'):
            latent=torch.load(case['latent'],weights_only=True).to('cuda')
        else:
            latent=torch.randn((1,n,pipe.transformer.config.in_channels),device='cuda',dtype=torch.bfloat16,
                generator=torch.Generator(device='cuda').manual_seed(SEED))
        assert latent.shape[1]==n
        latent_hash=digest(latent)
        latent_path=ROOT/'latents'/f'{cid}.pt'
        if latent_path.exists():assert digest(torch.load(latent_path,weights_only=True))==latent_hash
        else:torch.save(latent.cpu(),latent_path)
        pipe.scheduler=FlowMatchEulerDiscreteScheduler.from_config(SHIFT3)
        initial_cond=case.get('condition_count',n)
        samples=collector.begin_collection(cid,n,initial_cond,case['indices'])
        if 'condition_count' not in case:collector.condition_count=None
        save(ROOT/'data'/f'samples_{args.phase}_{cid}.json',samples)
        def callback(p,step,t,values):
            collector.confirm_update(values['latents'])
            return {}
        try:
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                result=pipe(image=image,prompt=case['prompt'],negative_prompt=' ',true_cfg_scale=1,num_inference_steps=8,
                    height=height,width=width,generator=torch.Generator(device='cuda').manual_seed(SEED),latents=latent.clone(),
                    callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents']).images[0]
            torch.cuda.synchronize()
            assert collector.step==8 and collector.local_calls==collector.local_completed==8
            assert collector.oracle_calls==collector.oracle_completed==6
            assert len(collector.anchor_seen)==env['layers'] and len(collector.pair_seen)==6*env['layers']
            assert all(row['student_only_euler_bitwise_equal'] for row in collector.records)
            buffer=io.BytesIO();result.save(buffer,format='PNG');content=buffer.getvalue()
            output=ROOT/'outputs'/f'{args.phase}__{cid}.png';output.write_bytes(content)
            if case.get('reference_sha256'):assert sha(output)==case['reference_sha256']
            moments_path=ROOT/'moments'/f'{args.phase}__{cid}.npz'
            collector.moments.data['metadata'].update(phase=args.phase,manifest_sha256=manifest_hash,seed=SEED,
                initial_latent_sha256=latent_hash,source_image_sha256=sha(image_path),student='uncorrected fixed core',
                target_count=n,condition_count=collector.condition_count,real_anchor=0,sample_indices_file=f'data/samples_{args.phase}_{cid}.json')
            save_npz(moments_path,collector.moments)
            row=dict(status='complete',phase=args.phase,attempt=attempt,case_id=cid,output=str(output.relative_to(ROOT)),output_sha256=sha(output),
                moments=str(moments_path.relative_to(ROOT)),moments_sha256=sha(moments_path),moments_bytes=moments_path.stat().st_size,
                local_calls=collector.local_calls,oracle_calls=collector.oracle_calls,real_dit_forwards=14,formal_training_image=args.phase=='train8',
                diagnostic_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                initial_latent_sha256=latent_hash,target_hw=case['target_hw'],source_image_sha256=sha(image_path),
                old_R2_PNG_exact=sha(output)==case['reference_sha256'] if case.get('reference_sha256') else None,
                records=collector.records,sigmas=pipe.scheduler.sigmas.cpu().tolist(),timesteps=pipe.scheduler.timesteps.cpu().tolist(),
                cache_check_scope='Sampled K/V at all layers plus complete velocity, ages, input and RNG; not full-cache byte equality')
            with ledger.open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            done[cid]=row
            collector.clear();collector.moments=None
            update('image_complete',phase_name=args.phase,case_id=cid,completed=len(done),real_dit_forwards=sum(r['real_dit_forwards'] for r in done.values()))
        except Exception as error:
            row=dict(status='failed',phase=args.phase,attempt=attempt,case_id=cid,error=repr(error),traceback=traceback.format_exc(),
                local_invoked=collector.local_calls,local_completed=collector.local_completed,oracle_invoked=collector.oracle_calls,oracle_completed=collector.oracle_completed)
            with ledger.open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            update('failed',failure_record=row)
            raise
    collector.close()
    update('complete',phase_name=args.phase,successful_requests=len(done),real_dit_forwards=sum(r['real_dit_forwards'] for r in done.values()),
        formal_training_images=0 if args.phase=='engineering' else len(done))


if __name__=='__main__':main()
