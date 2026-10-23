import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from PIL import Image

from attribution_executor import AttributionExecutor

PILOT = Path('/workspace/diffusers-fix/research/lightning8_preflight_20261022')
PREVIOUS = Path('/workspace/diffusers-fix/research/local_executor_boundary_20261022')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PILOT / 'scripts'))
from run_pilot import BASE, LORA, LORA_FILE, SEED, SHIFT3, tensor_hash
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def status(phase, **kwargs):
    value = dict(phase=phase, time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **kwargs)
    write_json(ROOT / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    for name in ['data', 'outputs', 'captures']:
        (ROOT / name).mkdir(exist_ok=True)
    assert not (ROOT / 'data/runs.jsonl').exists()
    attempt = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    env = dict(attempt=attempt, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        packages={n: importlib.metadata.version(n) for n in ['torch', 'diffusers', 'transformers', 'peft', 'huggingface_hub', 'torchao']},
        gpu_driver_name=torch.cuda.get_device_name(), physical_gpu_user_confirmed='H800', capability=torch.cuda.get_device_capability(),
        dtype='bfloat16', steps=8, cfg=1, seed=SEED, model=str(BASE), lora=str(Path(LORA) / LORA_FILE),
        previous_root=str(PREVIOUS), primary_root=str(PREVIOUS), local_pilot_root=str(PREVIOUS), baseline_root=str(PILOT),
        region='R2', full_steps_1based=[1, 8], planned_requests=12, planned_model_forwards=168,
        oracle_used_for_deployment_timing=False,
        error_tensor_scope='active_error/inactive_error/combined_error: raw local before mixing vs same-current-state dense; returned_error: mixed returned target output vs same-state dense',
        main_attribution_trajectory='B', independent_trajectory_outputs=['D', 'B', 'V', 'K'],
        scripts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'scripts').glob('*.py')})
    write_json(ROOT / 'data/environment.json', env)
    status('loading_model')
    started = time.perf_counter()
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
    env['model_load_and_fuse_seconds'] = time.perf_counter() - started
    write_json(ROOT / 'data/environment.json', env)
    executor = AttributionExecutor(pipe.transformer)
    executor.install()
    regions = json.loads((PREVIOUS / 'masks/regions.json').read_text())['cases']
    cases = json.loads((PREVIOUS / 'cases.json').read_text())['cases']
    old = [json.loads(x) for x in (PREVIOUS / 'data/runs.jsonl').read_text().splitlines()]
    baseline = [json.loads(x) for x in (PILOT / 'data/runs.jsonl').read_text().splitlines()]
    counts = dict(local=0, oracle=0, native=0, requests=0)
    for case in cases:
        cid = case['case_id']
        ref = next(r for r in baseline if r.get('run_id') == f'E_lightning8_fused__{cid}__measurement_0')
        latent = torch.load(PILOT / 'latents' / f'{cid}.pt', weights_only=True).to('cuda')
        assert tensor_hash(latent) == ref['initial_latent_sha256']
        n = latent.shape[1]
        c = ref['first_cond_shapes']['hidden_states'][1] - n
        image = Image.open(PREVIOUS / case['image']).convert('RGB')
        for mode in ['D', 'B', 'V', 'K']:
            run_id = f'{mode}__{cid}'
            status('running', run_id=run_id, **counts)
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(SHIFT3)
            executor.begin_attribution(n, c, regions[cid]['regions']['R2']['indices'], mode, None)
            post_latents = []
            def callback(p, step, t, values):
                post_latents.append(values['latents'].detach().cpu().clone())
                if mode != 'D':
                    assert torch.equal(post_latents[-1], executor.captures[-1]['expected_post_latent'])
                    executor.records[-1]['actual_euler_update_bitwise_equal'] = True
                return {}
            original_forward = executor.forward
            def with_sigmas(*a, **kw):
                executor.sigmas = pipe.scheduler.sigmas
                return original_forward(*a, **kw)
            pipe.transformer.forward = with_sigmas
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                result = pipe(image=image, prompt=case['prompt'], negative_prompt=' ', true_cfg_scale=1,
                    num_inference_steps=8, height=ref['target_hw'][0], width=ref['target_hw'][1],
                    generator=torch.Generator(device='cuda').manual_seed(SEED), latents=latent.clone(),
                    callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents']).images[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert executor.step == 8 and len(post_latents) == 8
            output = f'outputs/{run_id}.png'
            buffer = io.BytesIO()
            result.save(buffer, format='PNG')
            (ROOT / output).write_bytes(buffer.getvalue())
            sha = hashlib.sha256(buffer.getvalue()).hexdigest()
            old_ref = next(r for r in old if r.get('run_id') == f'{"F" if mode == "D" else "R2"}__{cid}__measurement_0') if mode in ['D', 'B'] else None
            if old_ref:
                assert sha == old_ref['output_sha256'], f'{run_id} changed vs previous: {sha}'
            capture = f'captures/{run_id}.pt'
            torch.save(dict(steps=executor.captures, post_latents=post_latents), ROOT / capture)
            for step_record, cap in zip(executor.records, executor.captures):
                row = dict(case_id=cid, mode=mode, run_id=run_id, current_latent_sha256=tensor_hash(cap['current_latent']), **step_record)
                with (ROOT / 'data/step_metrics.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
            row = dict(status='complete', case_id=cid, mode=mode, run_id=run_id, output=output, output_sha256=sha,
                e2e_seconds=elapsed, timing_scope='diagnostic including oracle, synchronization, captures and metric computation; not deployment timing',
                local_calls=executor.local_calls, oracle_calls=executor.oracle_calls, native_calls=executor.native_calls,
                full_steps_1based=[1, 8] if mode != 'D' else list(range(1, 9)), region='R2',
                target_hw=ref['target_hw'], target_count=n, condition_count=c, initial_latent_sha256=tensor_hash(latent),
                selected_partial_target_count=len(regions[cid]['regions']['R2']['indices']),
                timesteps=pipe.scheduler.timesteps.cpu().tolist(), sigmas=pipe.scheduler.sigmas.cpu().tolist(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), capture=capture,
                old_output_exact=sha == old_ref['output_sha256'] if old_ref else None,
                local_seconds_sum=sum(r['local_seconds'] for r in executor.records), oracle_seconds_sum=sum(r['oracle_seconds'] for r in executor.records))
            with (ROOT / 'data/runs.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            counts['requests'] += 1
            for key in ['local', 'oracle', 'native']:
                counts[key] += row[f'{key}_calls']
            executor.clear()
            executor.captures.clear()
            executor.records.clear()
            del post_latents
            status('request_complete', run_id=run_id, output_sha256=sha, **counts)
    assert counts == dict(local=72, oracle=72, native=24, requests=12)
    status('complete', **counts, total_model_forwards=sum(counts[x] for x in ['local', 'oracle', 'native']))


if __name__ == '__main__':
    main()
