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

ROOT = Path(__file__).resolve().parent.parent
PREVIOUS = Path('/workspace/diffusers-fix/research/local_executor_boundary_20261022')
PILOT = Path('/workspace/diffusers-fix/research/lightning8_preflight_20261022')
sys.path.insert(0, str(PILOT / 'scripts'))
from run_pilot import BASE, LORA, LORA_FILE, SEED, SHIFT3, Recorder, tensor_hash
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
import diffusers


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def status(phase, **fields):
    value = dict(phase=phase, time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **fields)
    save(ROOT / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['optimization'], default='optimization')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    configs = ['F', 'R2', 'FIX0', 'FIX1', 'FIX2', 'ROT'] if args.phase == 'primary' else ['F', 'R2', 'SHIFT', 'AFFINE']
    for name in ['data', 'outputs', 'figures']:
        (ROOT / name).mkdir(exist_ok=True)
    ledger = ROOT / 'data' / f'runs_{args.phase}.jsonl'
    rows = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    if rows and not args.resume:
        raise RuntimeError('Existing ledger requires --resume')
    previous = {r['run_id']: r for r in rows if r['status'] == 'complete'}
    for row in previous.values():
        assert hashlib.sha256((ROOT / row['output']).read_bytes()).hexdigest() == row['output_sha256']
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    plan_path = ROOT / 'masks/rotation_plan.json'
    plan = json.loads(plan_path.read_text())
    cases = json.loads((PREVIOUS / 'cases.json').read_text())['cases']
    old = [json.loads(x) for x in (PREVIOUS / 'data/runs.jsonl').read_text().splitlines()]
    baseline = [json.loads(x) for x in (PILOT / 'data/runs.jsonl').read_text().splitlines()]
    attempt = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    env = dict(phase=args.phase, attempt=attempt, previous_root=str(PREVIOUS), baseline_root=str(PILOT),
        source=diffusers.__file__, source_version=diffusers.__version__,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        packages={k: importlib.metadata.version(k) for k in ['torch', 'transformers', 'peft', 'huggingface_hub', 'torchao']},
        gpu_driver_name=torch.cuda.get_device_name(), physical_gpu_user_confirmed='H800', capability=torch.cuda.get_device_capability(),
        configs=configs, repeats=args.repeats, planned_requests=len(configs)*len(cases)*(args.repeats+1),
        dtype='bfloat16', steps=8, cfg=1, seed=SEED, vae_tiling=True, backend='native SDPA',
        region_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        scripts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'scripts').glob('*.py')},
        timing_scope='memory source image and supplied ROI to PIL; includes cache setup, correction, cleanup, encode, all DiT, scheduler, decode; excludes model load, static table preparation, PNG, metrics, precomputed region plan',
        extra_dense_oracles_in_performance=0)
    save(ROOT / 'data' / f'environment_{args.phase}.json', env)
    status('loading_model', group=args.phase)
    start = time.perf_counter()
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
    assert not any('lora_' in n for n, _ in pipe.transformer.named_parameters())
    torch.cuda.synchronize()
    env['model_load_and_fuse_seconds'] = time.perf_counter() - start
    if args.phase == 'optimization':
        from value_time_correction import ValueTimeExecutor
        executor = ValueTimeExecutor(pipe.transformer)
        schedule = next(r for r in old if r.get('config') == 'F' and r.get('kind') == 'measurement')
        with torch.inference_mode():
            table_metadata = executor.prepare_tables(schedule['timesteps'])
        env['static_correction_tables'] = table_metadata
    else:
        executor = LocalTargetExecutor(pipe.transformer)
    executor.install()
    model_time_inputs = []
    def capture_model_timestep(module, inputs):
        model_time_inputs.append(inputs[0].detach().clone())
    timestep_handle = pipe.transformer.time_text_embed.register_forward_pre_hook(capture_model_timestep)
    env['timestep_contract_instrumentation'] = 'Eight tiny device clones in all configurations; compare actual model time inputs after request timing, no extra model forward'
    save(ROOT / 'data' / f'environment_{args.phase}.json', env)
    samples = {}
    for case in cases:
        cid = case['case_id']
        ref = next(r for r in baseline if r.get('run_id') == f'E_lightning8_fused__{cid}__measurement_0')
        z = torch.load(PILOT / 'latents' / f'{cid}.pt', weights_only=True).to('cuda')
        assert tensor_hash(z) == ref['initial_latent_sha256']
        samples[cid] = dict(latent=z, latent_hash=tensor_hash(z), image=Image.open(PREVIOUS / case['image']).convert('RGB'),
            target_hw=ref['target_hw'], target_count=z.shape[1], condition_count=ref['first_cond_shapes']['hidden_states'][1]-z.shape[1])
    recorder = Recorder()
    for obj, attribute, label, shape in [(pipe, 'encode_prompt', 'prompt_encode', False),
        (pipe, '_encode_vae_image', 'vae_encode', False), (pipe.transformer, 'forward', 'dit', True),
        (pipe.vae, 'decode', 'vae_decode', False)]:
        recorder.wrap(obj, attribute, label, shapes=shape)
    total_done = len(previous)
    def request(case, config, kind, repeat):
        nonlocal total_done
        cid = case['case_id']
        run_id = f'{config}__{cid}__{kind}_{repeat}'
        if run_id in previous:
            return
        sample = samples[cid]
        policy = plan['cases'][cid]['configs']['R2' if config in ['SHIFT', 'AFFINE'] else config]
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(SHIFT3)
        recorder.wrap(pipe.scheduler, 'step', 'scheduler')
        recorder.reset()
        model_time_inputs.clear()
        if args.phase == 'optimization':
            executor.correction_mode = {'SHIFT': 'shift', 'AFFINE': 'affine'}.get(config, 'none')
        status('running', group=args.phase, run_id=run_id, completed=total_done)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        recorder.active = True
        try:
            mapping = {int(k): v for k, v in policy['per_step_indices'].items()}
            executor.begin(sample['target_count'], sample['condition_count'], policy['indices'],
                full_steps=[x-1 for x in policy['full_steps_1based']], native=policy['native'], per_step_indices=mapping)
            with torch.inference_mode():
                result = pipe(image=sample['image'], prompt=case['prompt'], negative_prompt=' ', true_cfg_scale=1,
                    num_inference_steps=8, height=sample['target_hw'][0], width=sample['target_hw'][1],
                    generator=torch.Generator(device='cuda').manual_seed(SEED), latents=sample['latent'].clone()).images[0]
            trace, age_trace = executor.trace.copy(), executor.age_trace.copy()
            correction_trace = getattr(executor, 'correction_trace', []).copy()
            cache_bytes = executor.cache_bytes()
            assert executor.step == 8 and len(recorder.shapes) == 8
            if config == 'ROT':
                assert len(age_trace) == 8 and max(t['max_read_age'] for t in age_trace) <= 2
            executor.clear()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            recorder.active = False
            observed_model_timesteps = torch.cat(model_time_inputs).float().cpu().tolist()
            assert observed_model_timesteps == table_metadata['model_timesteps']
            assert pipe.scheduler.timesteps.float().cpu().tolist() == table_metadata['raw_scheduler_timesteps']
            output = f'outputs/{args.phase}__{run_id}.png'
            buffer = io.BytesIO()
            result.save(buffer, format='PNG')
            (ROOT / output).write_bytes(buffer.getvalue())
            sha = hashlib.sha256(buffer.getvalue()).hexdigest()
            control = next(r for r in old if r.get('run_id') == f'{config}__{cid}__measurement_0') if config in ['F', 'R2'] else None
            if control:
                assert sha == control['output_sha256'], f'Control output changed: {run_id}'
            current = dict(status='complete', run_id=run_id, phase=args.phase, config=config, case_id=cid, kind=kind,
                repeat=repeat, attempt=attempt, seed=SEED, initial_latent_sha256=sample['latent_hash'],
                target_hw=sample['target_hw'], target_count=sample['target_count'], condition_count=sample['condition_count'],
                e2e_seconds=elapsed, cuda_event_seconds=recorder.seconds(), dit_calls=len(recorder.shapes),
                full_steps_1based=policy['full_steps_1based'], output=output, output_sha256=sha,
                trace=trace, age_trace=age_trace, correction_trace=correction_trace, cache_bytes=cache_bytes,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                old_output_exact=sha == control['output_sha256'] if control else None,
                model_timestep_contract_exact=True, observed_model_timesteps=observed_model_timesteps,
                timesteps=pipe.scheduler.timesteps.cpu().tolist(), sigmas=pipe.scheduler.sigmas.cpu().tolist())
            with ledger.open('a') as f:
                f.write(json.dumps(current) + '\n')
            previous[run_id] = current
            total_done += 1
            status('request_complete', group=args.phase, run_id=run_id, completed=total_done, seconds=elapsed)
        except Exception as error:
            recorder.active = False
            with ledger.open('a') as f:
                f.write(json.dumps(dict(status='failed', phase=args.phase, run_id=run_id, attempt=attempt,
                    dit_calls_completed=len(recorder.shapes), error=repr(error), traceback=traceback.format_exc()))+'\n')
            executor.clear()
            raise
    for case in cases:
        for config in configs:
            request(case, config, 'warmup', 0)
    for repeat in range(args.repeats):
        for case_index, case in enumerate(cases):
            shift = (repeat + case_index) % len(configs)
            for config in configs[shift:] + configs[:shift]:
                request(case, config, 'measurement', repeat)
    assert len(previous) == env['planned_requests']
    timestep_handle.remove()
    status('complete', group=args.phase, requests=len(previous), total_model_forwards=sum(r['dit_calls'] for r in previous.values()))


if __name__ == '__main__':
    main()
