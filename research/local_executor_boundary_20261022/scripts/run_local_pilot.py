import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import statistics
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

PILOT = Path('/workspace/diffusers-fix/research/lightning8_preflight_20261022')
sys.path.insert(0, str(PILOT / 'scripts'))
from run_pilot import BASE, LORA, LORA_FILE, SEED, SHIFT3, Recorder, tensor_hash
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
import diffusers

ROOT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def status(phase, **kwargs):
    v = dict(phase=phase, time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **kwargs)
    write_json(ROOT / 'status.json', v)
    print(json.dumps(v, ensure_ascii=False), flush=True)


def compare(reference, current):
    assert len(reference) == len(current) == 8
    out = []
    for step, (a, b) in enumerate(zip(reference, current)):
        assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
        af, bf = a.float(), b.float()
        delta = af - bf
        out.append(dict(step=step + 1, max_abs=delta.abs().max().item(), relative_l2=(delta.norm() / af.norm().clamp_min(1e-12)).item(),
            exact=torch.equal(a, b), reference_sha256=tensor_hash(a), current_sha256=tensor_hash(b)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--safe', action='store_true')
    parser.add_argument('--closure', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    for name in ['data', 'outputs', 'correctness']:
        (ROOT / name).mkdir(exist_ok=True)
    assert not (args.safe and args.closure)
    rows_path = ROOT / ('data/closure_runs.jsonl' if args.closure else 'data/runs.jsonl')
    old = [json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
    if args.closure:
        old = [json.loads(x) for x in (ROOT/'data/runs.jsonl').read_text().splitlines()] + old
    if old and not args.resume:
        raise RuntimeError('Existing data require --resume')
    previous = {x['run_id']: x for x in old if x['status'] == 'complete'}
    output_by_hash = {x['output_sha256']: x['output'] for x in previous.values()}
    for row in previous.values():
        assert hashlib.sha256((ROOT / row['output']).read_bytes()).hexdigest() == row['output_sha256']
    attempt = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    env = dict(attempt=attempt, python=sys.version, diffusers_version=diffusers.__version__, source=diffusers.__file__,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        packages={k: importlib.metadata.version(k) for k in ['torch', 'transformers', 'peft', 'huggingface_hub', 'torchao', 'safetensors']},
        gpu_driver_name=torch.cuda.get_device_name(), physical_gpu_user_confirmed='H800', capability=torch.cuda.get_device_capability(),
        scripts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'scripts').glob('*.py')},
        region_manifest_sha256=hashlib.sha256((ROOT / 'masks/regions.json').read_bytes()).hexdigest(), repeats=args.repeats,
        dtype='bfloat16', attention='native PyTorch SDPA', vae_tiling=True, cfg=1, steps=8,
        fresh_request_cache=True, automatic_region_locator=False, region_generation_in_request_timing=False,
        request_scope='memory image and preprovided ROI to PIL, includes cache setup/cleanup, excludes PNG and metrics')
    write_json(ROOT / 'data' / f'environment_{attempt}.json', env)
    tiny = json.loads((ROOT / 'data/tiny_correctness.json').read_text())
    assert tiny['status'] == 'pass'
    status('loading_base')
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
    assert not pipe.transformer.zero_cond_t
    torch.cuda.synchronize()
    env['model_load_and_fuse_seconds'] = time.perf_counter() - started
    write_json(ROOT / 'data' / f'environment_{attempt}.json', env)
    regions = json.loads((ROOT / 'masks/regions.json').read_text())['cases']
    cases = json.loads((ROOT / 'cases.json').read_text())['cases']
    prepared = {}
    for case in cases:
        cid = case['case_id']
        latent = torch.load(PILOT / 'latents' / f'{cid}.pt', weights_only=True).to('cuda')
        previous_row = next(json.loads(x) for x in (PILOT / 'data/runs.jsonl').read_text().splitlines() if json.loads(x).get('run_id') == f'E_lightning8_fused__{cid}__measurement_0')
        assert tensor_hash(latent) == previous_row['initial_latent_sha256']
        prepared[cid] = dict(latent=latent, latent_hash=tensor_hash(latent), image=Image.open(ROOT / case['image']).convert('RGB'),
            height=previous_row['target_hw'][0], width=previous_row['target_hw'][1], target_count=latent.shape[1], condition_count=previous_row['first_cond_shapes']['hidden_states'][1] - latent.shape[1])
    executor = LocalTargetExecutor(pipe.transformer)
    executor.install()
    recorder = Recorder()
    recorder.wrap(pipe, 'encode_prompt', 'prompt_encode')
    recorder.wrap(pipe, '_encode_vae_image', 'vae_encode')
    recorder.wrap(pipe.transformer, 'forward', 'dit', shapes=True)
    recorder.wrap(pipe.vae, 'decode', 'vae_decode')
    active_capture = {'enabled': False, 'predictions': []}
    def capture_prediction(module, module_args, output):
        if active_capture['enabled']:
            active_capture['predictions'].append(output[0][:, :active_capture['count']].detach().clone())
    handle = pipe.transformer.register_forward_hook(capture_prediction)

    def request(case, config, kind, repeat=0, capture=False, shape_audit=False):
        run_id = f'{config}__{case["case_id"]}__{kind}_{repeat}'
        capture_path = ROOT / 'correctness' / f'{run_id}.pt'
        if run_id in previous:
            row = previous[run_id]
            saved = torch.load(capture_path, weights_only=True) if capture else None
            return row, saved
        sample = prepared[case['case_id']]
        n, c = sample['target_count'], sample['condition_count']
        index = list(range(n)) if config in ['F', 'F_safe', 'F_closure', 'G', 'G_reverse'] else regions[case['case_id']]['regions']['R0' if config == 'Rdynamic' else config.split('_')[0]]['indices']
        if config == 'P25' or config == 'P50':
            raise ValueError('Area sweeps are separate diagnostics, not this primary matrix')
        full = list(range(8)) if config in ['F', 'F_safe', 'F_closure', 'G', 'G_reverse'] else ([0, 1, 3, 5, 7] if config.endswith('_safe') else [0, 7])
        per_step = None
        if config == 'Rdynamic':
            index = regions[case['case_id']]['regions']['R0']['indices']
            per_step = {step: regions[case['case_id']]['regions']['R0' if step < 4 else 'R2']['indices'] for step in range(8)}
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(SHIFT3)
        recorder.wrap(pipe.scheduler, 'step', 'scheduler')
        recorder.reset()
        latents = []
        active_capture.update(enabled=capture, predictions=[], count=n)
        def callback(p, i, t, values):
            if capture:
                latents.append(values['latents'].detach().clone())
            return {}
        shape_rows = []
        hooks = []
        if shape_audit:
            raw_context = {}
            def capture_raw(module, module_args, module_kwargs):
                raw_context['hidden'] = module_kwargs['hidden_states']
            hooks.append(pipe.transformer.register_forward_pre_hook(capture_raw, with_kwargs=True))
            def audit_gather(module, inputs):
                ids = executor.all_targets if executor.step in executor.full_steps else executor.indices[executor.step]
                ids = torch.cat([ids, executor.conditions])
                expected_raw = raw_context['hidden'].index_select(1, ids)
                equal = torch.equal(inputs[0], expected_raw)
                assert equal
                shape_rows.append(dict(step=executor.step+1,layer=-1,operator='img_in_current_raw_gather',input_shape=list(inputs[0].shape),exact_current_raw_source=equal))
            hooks.append(pipe.transformer.img_in.register_forward_pre_hook(audit_gather))
            for bi in [0, len(pipe.transformer.transformer_blocks) // 2, len(pipe.transformer.transformer_blocks) - 1]:
                block = pipe.transformer.transformer_blocks[bi]
                for label, module in [('image_to_q',block.attn.to_q),('image_to_k',block.attn.to_k),('image_to_v',block.attn.to_v),('image_mlp',block.img_mlp),('text_mlp',block.txt_mlp)]:
                    def hook(mod, inputs, layer=bi, name=label):
                        shape_rows.append(dict(step=executor.step + 1, layer=layer, operator=name, input_shape=list(inputs[0].shape)))
                    hooks.append(module.register_forward_pre_hook(hook))
        status('running', run_id=run_id)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        recorder.active = True
        started = time.perf_counter()
        try:
            executor.begin(n, c, index, full_steps=full, native=config in ['F','F_safe','F_closure'], reverse_full=config == 'G_reverse', per_step_indices=per_step)
            with torch.inference_mode():
                image = pipe(image=sample['image'], prompt=case['prompt'], negative_prompt=' ', true_cfg_scale=1,
                    num_inference_steps=8, height=sample['height'], width=sample['width'],
                    generator=torch.Generator(device='cuda').manual_seed(SEED), latents=sample['latent'].clone(),
                    callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents']).images[0]
            trace = executor.trace.copy()
            cache_bytes = executor.cache_bytes()
            final_age = executor.age.tolist()
            assert executor.step == 8
            executor.clear()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            recorder.active = False
            assert len(recorder.shapes) == 8
            encoded = io.BytesIO()
            image.save(encoded, format='PNG')
            image_bytes = encoded.getvalue()
            sha = hashlib.sha256(image_bytes).hexdigest()
            output = output_by_hash.get(sha)
            if output is None:
                output = f'outputs/{run_id}.png'
                (ROOT / output).write_bytes(image_bytes)
                output_by_hash[sha] = output
            row = dict(status='complete', run_id=run_id, attempt=attempt, case_id=case['case_id'], config=config, kind=kind, repeat=repeat,
                seed=SEED, initial_latent_sha256=sample['latent_hash'], target_hw=[sample['height'], sample['width']],
                target_count=n, condition_count=c, selected_partial_target_count=len(index), per_step_selected_target_count={str(t['step']):t['selected_target'] for t in trace if t['layer']==0}, full_steps_1based=[i+1 for i in full],
                e2e_seconds=elapsed, cuda_event_seconds=recorder.seconds(), dit_calls=len(recorder.shapes),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                cache_bytes=cache_bytes, cache_final_age=final_age, trace=trace, actual_module_shapes=shape_rows,
                timing_instrumentation='CUDA stage events; hooks only in shape_audit; per-step capture only in correctness',
                output=output, output_sha256=sha, scheduler_config=dict(pipe.scheduler.config),
                timesteps=pipe.scheduler.timesteps.cpu().tolist(), sigmas=pipe.scheduler.sigmas.cpu().tolist())
            if capture:
                saved = dict(predictions=[x.cpu() for x in active_capture['predictions']], latents=[x.cpu() for x in latents])
                torch.save(saved, capture_path)
                row['capture'] = str(capture_path.relative_to(ROOT))
            else:
                saved = None
            with rows_path.open('a') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            previous[run_id] = row
            print(json.dumps(dict(run_id=run_id, seconds=elapsed, dit=row['dit_calls'], cache_GiB=cache_bytes/2**30)),flush=True)
            return row, saved
        except Exception as error:
            executor.clear()
            recorder.active = False
            with rows_path.open('a') as f:
                f.write(json.dumps(dict(status='failed',run_id=run_id,attempt=attempt,error=f'{type(error).__name__}: {error}'))+'\n')
            raise
        finally:
            active_capture['enabled'] = False
            active_capture['predictions'] = []
            for h in hooks:
                h.remove()

    correctness = dict(status='running', comparisons=[], tolerances=dict(relative_l2=1e-3, max_abs=.02))
    reference = {}
    for case in cases:
        row, base_capture = request(case, 'F', 'correctness', capture=True)
        reference[case['case_id']] = (row, base_capture)
        g_row, custom = request(case, 'G', 'correctness', capture=True)
        comparison = dict(case_id=case['case_id'], order='canonical', predictions=compare(base_capture['predictions'], custom['predictions']), latents=compare(base_capture['latents'], custom['latents']))
        correctness['comparisons'].append(comparison)
        write_json(ROOT / 'data/real_correctness.json', correctness)
        assert all(x['max_abs'] <= .02 and x['relative_l2'] <= 1e-3 for group in ['predictions','latents'] for x in comparison[group])
    row, reverse = request(cases[0], 'G_reverse', 'correctness', capture=True)
    comparison = dict(case_id=cases[0]['case_id'], order='reverse', predictions=compare(reference[cases[0]['case_id']][1]['predictions'], reverse['predictions']), latents=compare(reference[cases[0]['case_id']][1]['latents'], reverse['latents']))
    correctness['comparisons'].append(comparison)
    assert all(x['max_abs'] <= .02 and x['relative_l2'] <= 1e-3 for group in ['predictions','latents'] for x in comparison[group])
    a1,_ = request(cases[0], 'R0', 'reset_a1')
    b,_ = request(cases[-1], 'R0', 'reset_b')
    a2,_ = request(cases[0], 'R0', 'reset_a2')
    correctness['request_reset_aba'] = dict(a1=a1['run_id'], b=b['run_id'], a2=a2['run_id'], output_equal=a1['output_sha256'] == a2['output_sha256'])
    assert correctness['request_reset_aba']['output_equal']
    dynamic,_ = request(cases[1], 'Rdynamic', 'dynamic_audit', shape_audit=True)
    shape,_ = request(cases[-1], 'R0', 'shape_audit', shape_audit=True)
    for row in [dynamic, shape]:
        for entry in row['actual_module_shapes']:
            if entry['operator'].startswith('image_'):
                expected = next(t['image_qkv_tokens'] for t in row['trace'] if t['step']==entry['step'] and t['layer']==entry['layer'])
                assert entry['input_shape'][1] == expected
    expected_new = len(set(regions[cases[1]['case_id']]['regions']['R2']['indices']) - set(regions[cases[1]['case_id']]['regions']['R0']['indices']))
    assert next(t['newly_activated_from_current_latent'] for t in dynamic['trace'] if t['step']==5 and t['layer']==0) == expected_new
    correctness['dynamic_new_token_count'] = expected_new
    correctness['dynamic_audit_run'] = dynamic['run_id']
    correctness['shape_audit_run'] = shape['run_id']
    correctness['status'] = 'passed'
    write_json(ROOT / 'data/real_correctness.json', correctness)
    status('correctness_passed')
    if args.closure:
        probe_path=ROOT/'masks/closure_probe.json'
        probe=json.loads(probe_path.read_text())
        regions['remove_hair']['regions'].update(probe['cases']['remove_hair']['regions'])
        cases=[case for case in cases if case['case_id']=='remove_hair']
        env['closure_probe_sha256']=hashlib.sha256(probe_path.read_bytes()).hexdigest()
        write_json(ROOT/'data'/f'environment_{attempt}.json',env)
    configs = ['F_closure','R4','R5'] if args.closure else (['F_safe','R2_safe','R3_safe'] if args.safe else ['F','R0','R1','R2','R3'])
    write_json(ROOT / ('data/closure_plan.json' if args.closure else ('data/fallback_plan.json' if args.safe else 'data/performance_plan.json')),dict(configs=configs,repeats=args.repeats,cases=[x['case_id'] for x in cases],
        warmups_per_case_config=1, planned_requests=len(cases)*len(configs)*(1+args.repeats), full_steps_1based=([1,2,4,6,8] if args.safe else [1,8]), native_baseline='F_closure' if args.closure else ('F_safe' if args.safe else 'F'),
        deviations=['P25/P50 sweeps deferred; R0 fractions 9-28% provide low-area evidence, no 50% area trend claim', 'G used for correctness, no standalone five-repeat overhead benchmark', 'No mid-step branch clone; only request reset and dynamic expansion tested']))
    for case in cases:
        for config in configs:
            request(case, config, 'warmup')
        for repeat in range(args.repeats):
            order = configs[repeat % len(configs):] + configs[:repeat % len(configs)]
            for config in order:
                request(case, config, 'measurement', repeat)
    handle.remove()
    status('complete', experiment_mode='closure_probe' if args.closure else ('safe_diagnostic' if args.safe else 'primary'), successful_requests=len(previous), dit_calls=sum(x['dit_calls'] for x in previous.values()))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
        raise
