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

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from PIL import Image
from local_executor import LocalTargetExecutor
from budget_policy import build_policy

ROOT = Path(__file__).resolve().parent.parent
PREVIOUS = Path('/workspace/diffusers-fix/research/local_executor_boundary_20261022')
PILOT = Path('/workspace/diffusers-fix/research/lightning8_preflight_20261022')
sys.path.insert(0, str(PILOT / 'scripts'))
from run_pilot import BASE, LORA, LORA_FILE, SEED, SHIFT3, Recorder, tensor_hash
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
import diffusers

CONFIGS = ['F', 'R2', 'B125', 'B250', 'B333', 'B500']


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def status(phase, **fields):
    value = dict(phase=phase, time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **fields)
    save(ROOT / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats != 3:
        raise ValueError('Frozen pilot requires exactly three timing repeats')
    for name in ['data', 'outputs', 'figures']:
        (ROOT / name).mkdir(exist_ok=True)
    ledger = ROOT / 'data/runs_primary.jsonl'
    rows = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    if rows and not args.resume:
        raise RuntimeError('Existing ledger requires explicit --resume')
    previous = {r['run_id']: r for r in rows if r['status'] == 'complete'}
    for row in previous.values():
        assert hashlib.sha256((ROOT / row['output']).read_bytes()).hexdigest() == row['output_sha256']
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    plan_path = ROOT / 'masks/budget_plan.json'
    plan = json.loads(plan_path.read_text())
    cases = json.loads((PREVIOUS / 'cases.json').read_text())['cases']
    old = [json.loads(x) for x in (PREVIOUS / 'data/runs.jsonl').read_text().splitlines()]
    baseline = [json.loads(x) for x in (PILOT / 'data/runs.jsonl').read_text().splitlines()]
    schedule = next(r for r in old if r.get('config') == 'F' and r.get('kind') == 'measurement')
    expected_times = (torch.tensor(schedule['timesteps']).to(torch.bfloat16) / 1000).float().tolist()
    attempt = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    props = torch.cuda.get_device_properties(0)
    env = dict(phase='primary', attempt=attempt, previous_root=str(PREVIOUS), baseline_root=str(PILOT),
        source=diffusers.__file__, source_version=diffusers.__version__,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        packages={k: importlib.metadata.version(k) for k in ['torch','transformers','peft','huggingface_hub','torchao']},
        gpu_driver_name=props.name, physical_gpu_user_confirmed='H800', capability=torch.cuda.get_device_capability(),
        cuda_total_memory_bytes=props.total_memory, configs=CONFIGS, repeats=3, planned_requests=72,
        dtype='bfloat16', steps=8, cfg=1, seed=SEED, vae_tiling=True, backend='native SDPA',
        region_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').glob('*.py')},
        timing_scope='Memory source image and supplied R2 core plus pre-encoded background blocks to PIL; includes per-request CPU eight-step budget selection, executor indices/cache setup, encoding, all DiT, scheduler, decode and cache cleanup. Excludes model load, initial block partition, reference checks, PNG and metrics.',
        budget_selector='Oldest real refresh first, SHA tie-break, whole 2x2 background blocks, greedy cap, no clipping of core or blocks',
        background_budgets={'R2':[0,1],'B125':[1,8],'B250':[1,4],'B333':[1,3],'B500':[1,2]},
        selection_precomputation_reused_for_candidates=False, static_policies_used_only_for_F_R2=True, block_partition_reused=True,
        extra_dense_oracles_in_performance=0, training_images_used=0,
        value_correction=False, adaptive_quality_risk=False, full_fallback_other_than_endpoints=False,
        expected_model_timesteps=expected_times, expected_raw_timesteps=schedule['timesteps'])
    save(ROOT/'data/environment_primary.json', env)
    status('loading_model', group='primary')
    started_load = time.perf_counter()
    pipe = QwenImageEditPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.to('cuda')
    pipe.vae.enable_tiling()
    pipe.transformer.set_attention_backend('native')
    pipe.set_progress_bar_config(disable=True)
    for module in [pipe.transformer, pipe.text_encoder, pipe.vae]:
        module.eval()
    pipe.load_lora_weights(LORA, weight_name=LORA_FILE, adapter_name='lightning8', local_files_only=True)
    pipe.fuse_lora(adapter_names=['lightning8'], safe_fusing=True)
    pipe.unload_lora_weights()
    assert not any('lora_' in n for n,_ in pipe.transformer.named_parameters())
    torch.cuda.synchronize()
    env['model_load_and_fuse_seconds'] = time.perf_counter()-started_load
    executor = LocalTargetExecutor(pipe.transformer)
    executor.install()
    model_times = []
    def capture_time(module, inputs):
        model_times.append(inputs[0].detach().clone())
    timestep_handle = pipe.transformer.time_text_embed.register_forward_pre_hook(capture_time)
    env['timestep_contract_instrumentation'] = 'Eight small GPU timestep clones in every config; checked after request timing without additional model forward'
    save(ROOT/'data/environment_primary.json', env)
    samples = {}
    for case in cases:
        cid = case['case_id']
        ref = next(r for r in baseline if r.get('run_id') == f'E_lightning8_fused__{cid}__measurement_0')
        z = torch.load(PILOT/'latents'/f'{cid}.pt', weights_only=True).to('cuda')
        assert tensor_hash(z) == ref['initial_latent_sha256']
        samples[cid] = dict(latent=z, latent_hash=tensor_hash(z), image=Image.open(PREVIOUS/case['image']).convert('RGB'),
            target_hw=ref['target_hw'], target_count=z.shape[1], condition_count=ref['first_cond_shapes']['hidden_states'][1]-z.shape[1])
    recorder = Recorder()
    for obj, attribute, label, shape in [(pipe,'encode_prompt','prompt_encode',False),
        (pipe,'_encode_vae_image','vae_encode',False),(pipe.transformer,'forward','dit',True),(pipe.vae,'decode','vae_decode',False)]:
        recorder.wrap(obj,attribute,label,shapes=shape)
    total_done = len(previous)
    def request(case, config, kind, repeat):
        nonlocal total_done
        cid = case['case_id']
        run_id = f'{config}__{cid}__{kind}_{repeat}'
        if run_id in previous:
            return
        sample = samples[cid]
        frozen = plan['cases'][cid]['configs'][config]
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(SHIFT3)
        recorder.wrap(pipe.scheduler,'step','scheduler')
        recorder.reset()
        model_times.clear()
        status('running',group='primary',run_id=run_id,completed=total_done)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        recorder.active = True
        try:
            if config in ['F','R2']:
                policy = frozen
                selection_seconds = 0.0
            else:
                selector_started = time.perf_counter()
                policy = build_policy(plan['cases'][cid],config)
                selection_seconds = time.perf_counter()-selector_started
            mapping = {int(k):v for k,v in policy['per_step_indices'].items()}
            executor.begin(sample['target_count'],sample['condition_count'],policy['indices'],
                full_steps=[v-1 for v in policy['full_steps_1based']],native=policy['native'],per_step_indices=mapping)
            with torch.inference_mode():
                result = pipe(image=sample['image'],prompt=case['prompt'],negative_prompt=' ',true_cfg_scale=1,
                    num_inference_steps=8,height=sample['target_hw'][0],width=sample['target_hw'][1],
                    generator=torch.Generator(device='cuda').manual_seed(SEED),latents=sample['latent'].clone()).images[0]
            trace, age_trace = executor.trace.copy(),executor.age_trace.copy()
            cache_bytes = executor.cache_bytes()
            assert executor.step == 8 and len(recorder.shapes) == 8
            executor.clear()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            recorder.active = False
            assert policy == frozen, f'Per-request budget selection differs from preregistration: {run_id}'
            observed = torch.cat(model_times).float().cpu().tolist()
            assert observed == expected_times
            assert pipe.scheduler.timesteps.float().cpu().tolist() == schedule['timesteps']
            output = f'outputs/primary__{run_id}.png'
            buffer = io.BytesIO()
            result.save(buffer,format='PNG')
            (ROOT/output).write_bytes(buffer.getvalue())
            output_hash = hashlib.sha256(buffer.getvalue()).hexdigest()
            control = next(r for r in old if r.get('run_id') == f'{config}__{cid}__measurement_0') if config in ['F','R2'] else None
            if control:
                assert output_hash == control['output_sha256'], f'Old control output changed: {run_id}'
            row = dict(status='complete',run_id=run_id,phase='primary',config=config,case_id=cid,kind=kind,repeat=repeat,
                attempt=attempt,seed=SEED,initial_latent_sha256=sample['latent_hash'],target_hw=sample['target_hw'],
                target_count=sample['target_count'],condition_count=sample['condition_count'],e2e_seconds=elapsed,
                budget_selection_seconds=selection_seconds,budget_selection_rebuilt=config not in ['F','R2'],budget_selection_metadata=policy.get('budgetmeta',{}),
                policy_matches_frozen=True,per_step_selected_target_counts=[len(mapping[k]) for k in range(8)],
                cuda_event_seconds=recorder.seconds(),dit_calls=len(recorder.shapes),full_steps_1based=policy['full_steps_1based'],
                output=output,output_sha256=output_hash,trace=trace,age_trace=age_trace,cache_bytes=cache_bytes,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                old_output_exact=output_hash==control['output_sha256'] if control else None,
                model_timestep_contract_exact=True,observed_model_timesteps=observed,
                timesteps=pipe.scheduler.timesteps.cpu().tolist(),sigmas=pipe.scheduler.sigmas.cpu().tolist())
            with ledger.open('a') as f:
                f.write(json.dumps(row)+'\n')
            previous[run_id]=row
            total_done += 1
            status('request_complete',group='primary',run_id=run_id,completed=total_done,seconds=elapsed)
        except Exception as error:
            recorder.active=False
            with ledger.open('a') as f:
                f.write(json.dumps(dict(status='failed',phase='primary',run_id=run_id,attempt=attempt,
                    dit_calls_completed=len(recorder.shapes),error=repr(error),traceback=traceback.format_exc()))+'\n')
            executor.clear()
            raise
    for case in cases:
        for config in CONFIGS:
            request(case,config,'warmup',0)
    for repeat in range(3):
        for case_index,case in enumerate(cases):
            offset=(repeat+case_index)%len(CONFIGS)
            for config in CONFIGS[offset:]+CONFIGS[:offset]:
                request(case,config,'measurement',repeat)
    assert len(previous)==72
    timestep_handle.remove()
    status('complete',group='primary',requests=len(previous),total_model_forwards=sum(r['dit_calls'] for r in previous.values()))


if __name__=='__main__':
    main()
