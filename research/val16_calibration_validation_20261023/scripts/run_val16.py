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

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT.parent / 'fewshot_calibration_start_20261023'
PILOT = ROOT.parent / 'lightning8_preflight_20261022'
sys.path.insert(0, str(TRAIN / 'scripts'))
os.environ['VALUE_CHANNEL_STAGE4_SCRIPTS'] = str(TRAIN / 'scripts')
from local_executor import LocalTargetExecutor
from value_channel_correction import ValueChannelExecutor
from value_fitted_correction import ValueFittedExecutor
from moments import load_npz, deployment_route_b
sys.path.insert(0, str(PILOT / 'scripts'))
from run_pilot import BASE, LORA, LORA_FILE, SEED, SHIFT3, Recorder, tensor_hash
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline

CONFIGS = ['F', 'R2', 'PRIOR', 'A', 'B']
CAL_SHA = '6df60e49ca8d618c5458063a80ce5bf3a5b02b0c28c7e618b04b76733964518f'
MODEL_ID = 'Qwen-Image-Edit@ac7f9318f633fc4b5778c59367c8128225f1e3de|Lightning8@36ee4ff860856559b9c581eaa69414900488f3de8be2f2448586daec77e201b9|bf16-fused-unloaded|73faca368c0da77645ea58836aa8780d2d8c5353'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def status(phase, **values):
    value = dict(phase=phase, time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **values)
    save(ROOT / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def execution(cases):
    planned = [(c['case_id'], config, 'warmup', 0) for c in cases for config in CONFIGS]
    for repeat in range(3):
        for i, case in enumerate(cases):
            offset = (repeat + i) % len(CONFIGS)
            planned += [(case['case_id'], config, 'measurement', repeat) for config in CONFIGS[offset:] + CONFIGS[:offset]]
    return [dict(case_id=cid, config=config, kind=kind, repeat=repeat,
        run_id=f'{config}__{cid}__{kind}_{repeat}') for cid, config, kind, repeat in planned]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    for name in ['data', 'outputs', 'latents', 'logs']:
        (ROOT / name).mkdir(exist_ok=True)
    cases_path = ROOT / 'cases_val16.json'
    cases = json.loads(cases_path.read_text())['cases']
    if len(cases) != 16 or len({c['case_id'] for c in cases}) != 16 or any(c['status'] != 'PASS' for c in cases):
        raise ValueError('Exactly sixteen frozen QC-passed sources required; no reselection')
    calibration_path = TRAIN / 'data/calibration_train8.npz'
    assert sha(calibration_path) == CAL_SHA
    calibration = load_npz(calibration_path)
    train_cases_path = TRAIN / 'cases_train8.json'
    train_cases = json.loads(train_cases_path.read_text())['cases']
    assert not ({c['img_id'] for c in cases} & {c['img_id'] for c in train_cases})
    assert set(calibration['source_image_ids']) == {c['case_id'] for c in train_cases}
    frozen_plan = execution(cases)
    source_scripts = [TRAIN / 'scripts' / f for f in ['local_executor.py', 'value_time_correction.py',
        'value_channel_correction.py', 'value_fitted_correction.py', 'moments.py']]
    prereg = dict(version=1, cases_sha256=sha(cases_path), calibration_sha256=CAL_SHA,
        evaluation_regions_path='masks/evaluation_regions.json',
        evaluation_regions_sha256=sha(ROOT / 'masks/evaluation_regions.json'),
        case_ids=[c['case_id'] for c in cases], configs=CONFIGS, planned_requests=320, planned_dit_forwards=2560,
        warmups=80, measurements=240, repeats=3, quality_repeat=0, seed=SEED,
        planned_execution_order=[row['run_id'] for row in frozen_plan], planned_execution=frozen_plan,
        sources={c['case_id']:dict(source_sha256=c['source_sha256'], mask_sha256=c['mask_sha256'],
            core_indices_sha256=hashlib.sha256(json.dumps(c['indices']).encode()).hexdigest()) for c in cases},
        model_identity=MODEL_ID, scripts={str(p):sha(p) for p in source_scripts + [Path(__file__)]},
        quality_gate=dict(lpips_to_F_max=.02, extra_far_background_over_F_max=1/255), latency_target_saving=.10,
        evaluation_region='R0 is annotation-positive bbox rounded outward to target 16px token grid; band r=dilate(R0,r)&dilate(~R0,r), r=1/2/4; R2=R0 bbox plus clipped 2-token halo; far background=complement of R2; all configs share these regions',
        fitting_on_val=False, target_column_used=False, test_used=False, rl=False, new_gradient_steps=0,
        timing_scope='Memory source image and supplied ROI to PIL output; includes executor begin/index copies, encoding, all 8 DiT, correction reads/checks, Euler steps, decode and request cache cleanup. Excludes model load, source file I/O, source/ROI annotation, static coefficient preparation and table switching, initial latent creation, PNG and metrics; no subtraction of runtime correction cost',
        region_scope='Annotation-assisted executor/calibration validation; not automatic semantic ROI detection or its end-to-end latency')
    prereg_path = ROOT / 'data/preregistration.json'
    if prereg_path.exists():
        if json.loads(prereg_path.read_text()) != prereg:
            raise RuntimeError('Frozen preregistration changed')
    else:
        save(prereg_path, prereg)
    ledger = ROOT / 'data/runs_primary.jsonl'
    old = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    if old and not args.resume:
        raise RuntimeError('Use --resume to preserve previous attempts')
    done = {r['run_id']:r for r in old if r['status'] == 'complete'}
    assert len(done) == sum(r['status'] == 'complete' for r in old)
    for row in done.values():
        assert sha(ROOT / row['output']) == row['output_sha256']
    assert [r['run_id'] for r in old if r['status'] == 'complete'] == [r['run_id'] for r in frozen_plan[:len(done)]]
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    expected = json.loads((TRAIN / 'data/runs_train8.jsonl').read_text().splitlines()[0])
    raw_times = expected['timesteps']
    expected_times = (torch.tensor(raw_times).to(torch.bfloat16) / 1000).float().tolist()
    attempt = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    props = torch.cuda.get_device_properties(0)
    env = dict(attempt=attempt, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        gpu_driver_name=props.name, physical_gpu_user_confirmed='H800', capability=torch.cuda.get_device_capability(),
        total_memory_bytes=props.total_memory, packages={k:importlib.metadata.version(k) for k in ['torch','diffusers','transformers','peft','numpy']},
        model_identity=MODEL_ID, calibration_sha256=CAL_SHA, preregistration_sha256=sha(prereg_path),
        model_dtype='bfloat16', cfg=1, steps=8, backend='native SDPA', vae_tiling=True,
        expected_model_timesteps=expected_times, expected_raw_timesteps=raw_times,
        timing_scope=prereg['timing_scope'], all_correction_tables_resident_for_all_configs=True,
        new_teacher_or_shadow_forwards=0, val_refit=False)
    assert sha(Path(LORA) / LORA_FILE) == MODEL_ID.split('Lightning8@')[1].split('|')[0]
    env['lora_sha256_verified'] = True
    env['base_weight_provenance'] = 'Same frozen local snapshot as Train8; this run does not rehash all base weight shards'
    env_path = ROOT / 'data' / f'environment_{attempt}.json'
    save(env_path, env)
    status('loading_model', attempt=attempt, completed=len(done))
    started = time.perf_counter()
    pipe = QwenImageEditPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16, local_files_only=True).to('cuda')
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
    env['model_load_fuse_seconds'] = time.perf_counter() - started
    native = pipe.transformer.forward
    executors = dict(R2=LocalTargetExecutor(pipe.transformer), PRIOR=ValueChannelExecutor(pipe.transformer),
        A=ValueChannelExecutor(pipe.transformer), B=ValueFittedExecutor(pipe.transformer))
    assert all(ex.original_forward == native for ex in executors.values())
    with torch.inference_mode():
        prior_metadata = executors['PRIOR'].prepare_tables(raw_times)
        a_metadata = executors['A'].prepare_tables(raw_times, dict(mean=calibration['route_a']['mean'],
            variance=calibration['route_a']['variance'], source=str(calibration_path), source_hash=CAL_SHA))
        b = executors['B']
        contract = b.coefficient_contract(raw_times, model_identity=MODEL_ID)
        fitted = deployment_route_b(calibration, contract['heads'], contract['head_dim'], contract,
            str(calibration_path), CAL_SHA)
        b_metadata = b.prepare_fitted_tables(raw_times, fitted, model_identity=MODEL_ID)
    for name, ex in executors.items():
        if name != 'R2':
            ex.correction_mode = 'affine'
    env.update(tables=dict(PRIOR=prior_metadata, A=a_metadata, B=b_metadata),
        total_static_table_bytes=sum(ex.table_bytes() for name,ex in executors.items() if name!='R2'),
        static_allocated_bytes=torch.cuda.memory_allocated(), static_reserved_bytes=torch.cuda.memory_reserved())
    save(env_path, env)
    active = dict(config=None, executor=None, target_count=None, condition_count=None, invoked=0, completed=0)
    def dispatch(*args, **kwargs):
        active['invoked'] += 1
        hidden = kwargs['hidden_states']
        actual_condition = hidden.shape[1] - active['target_count']
        if active['condition_count'] is None:
            active['condition_count'] = actual_condition
            if active['executor'] is not None:
                active['executor'].condition_count = actual_condition
                active['executor'].conditions = torch.arange(active['target_count'],hidden.shape[1],device=hidden.device)
        assert active['condition_count'] == actual_condition
        result = native(*args, **kwargs) if active['config'] == 'F' else active['executor'].forward(*args, **kwargs)
        active['completed'] += 1
        return result
    pipe.transformer.forward = dispatch
    model_times = []
    handle = pipe.transformer.time_text_embed.register_forward_pre_hook(lambda module, inputs:model_times.append(inputs[0].detach().clone()))
    recorder = Recorder()
    for obj,attr,label,shape in [(pipe,'encode_prompt','prompt_encode',False),(pipe,'_encode_vae_image','vae_encode',False),
        (pipe.transformer,'forward','dit',True),(pipe.vae,'decode','vae_decode',False)]:
        recorder.wrap(obj,attr,label,shapes=shape)
    samples = {}
    for case in cases:
        cid = case['case_id']
        image_path = ROOT / case['image']
        assert sha(image_path) == case['source_sha256']
        image = Image.open(image_path).convert('RGB')
        h,w = case['target_hw']
        n = (h//16)*(w//16)
        z = torch.randn((1,n,pipe.transformer.config.in_channels),device='cuda',dtype=torch.bfloat16,
            generator=torch.Generator(device='cuda').manual_seed(SEED))
        latent_hash = tensor_hash(z)
        path = ROOT / 'latents' / f'{cid}.pt'
        if path.exists():
            assert tensor_hash(torch.load(path,weights_only=True)) == latent_hash
        else:
            torch.save(z.cpu(),path)
        samples[cid] = dict(image=image,latent=z,latent_hash=latent_hash,n=n)
    by_id = {c['case_id']:c for c in cases}
    for item in frozen_plan:
        run_id,cid,config,kind,repeat = [item[k] for k in ['run_id','case_id','config','kind','repeat']]
        if run_id in done:
            continue
        case,sample = by_id[cid],samples[cid]
        ex = None if config == 'F' else executors[config]
        assert all(not e.layer_cache and e.velocity is None for e in executors.values())
        active.update(config=config, executor=ex, target_count=sample['n'], condition_count=None, invoked=0, completed=0)
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(SHIFT3)
        recorder.wrap(pipe.scheduler,'step','scheduler')
        recorder.reset()
        model_times.clear()
        status('running', run_id=run_id, completed=len(done), planned=320)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        recorder.active = True
        try:
            if ex is not None:
                ex.begin(sample['n'],sample['n'],case['indices'],full_steps=[0,7],native=False)
            with torch.inference_mode():
                result = pipe(image=sample['image'],prompt=case['prompt'],negative_prompt=' ',true_cfg_scale=1,
                    num_inference_steps=8,height=case['target_hw'][0],width=case['target_hw'][1],
                    generator=torch.Generator(device='cuda').manual_seed(SEED),latents=sample['latent'].clone()).images[0]
            assert len(recorder.shapes) == active['completed'] == active['invoked'] == 8
            trace,age_trace,cache_bytes = (ex.trace.copy(),ex.age_trace.copy(),ex.cache_bytes()) if ex is not None else ([],[],0)
            if ex is not None:
                assert ex.step == 8
                ex.clear()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            recorder.active = False
            observed = torch.cat(model_times).float().cpu().tolist()
            assert observed == expected_times
            assert pipe.scheduler.timesteps.float().cpu().tolist() == raw_times
            assert all(not e.layer_cache and e.velocity is None for e in executors.values())
            buffer = io.BytesIO()
            result.save(buffer,format='PNG')
            content = buffer.getvalue()
            digest = hashlib.sha256(content).hexdigest()
            output = f'outputs/{config}__{cid}.png'
            path = ROOT / output
            if path.exists() and sha(path) != digest:
                output = f'outputs/{run_id}.png'
                path = ROOT / output
            if path.exists():
                assert sha(path) == digest
            else:
                path.write_bytes(content)
            row = dict(status='complete',**item,attempt=attempt,seed=SEED,
                initial_latent_sha256=sample['latent_hash'],target_hw=case['target_hw'],target_count=sample['n'],
                condition_count=active['condition_count'],e2e_seconds=elapsed,cuda_event_seconds=recorder.seconds(),
                dit_calls=len(recorder.shapes),oracle_calls=0,full_steps_1based=list(range(1,9)) if config=='F' else [1,8],
                output=output,output_sha256=digest,trace=trace,age_trace=age_trace,cache_bytes=cache_bytes,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                model_timestep_contract_exact=True,observed_model_timesteps=observed,
                timesteps=pipe.scheduler.timesteps.cpu().tolist(),sigmas=pipe.scheduler.sigmas.cpu().tolist(),
                correction_mode='none' if config in ['F','R2'] else 'affine',executor_class='native' if ex is None else type(ex).__name__,
                coefficient_sha256=CAL_SHA if config in ['A','B'] else None,
                source_sha256=case['source_sha256'],core_indices_sha256=prereg['sources'][cid]['core_indices_sha256'],
                request_reset=True,cleanup_all_executors_empty=True,first_cond_shapes=recorder.shapes[0])
            with ledger.open('a') as stream:
                stream.write(json.dumps(row,ensure_ascii=False) + '\n')
            done[run_id] = row
            status('request_complete',run_id=run_id,completed=len(done),planned=320,seconds=elapsed)
        except Exception as error:
            recorder.active = False
            row = dict(status='failed',**item,attempt=attempt,dit_calls_completed=active['completed'],dit_calls_invoked=active['invoked'],
                error=repr(error),traceback=traceback.format_exc())
            with ledger.open('a') as stream:
                stream.write(json.dumps(row,ensure_ascii=False) + '\n')
            for executor in executors.values():
                executor.clear()
            status('failed',failure_record=row)
            raise
    handle.remove()
    assert sha(calibration_path) == CAL_SHA
    assert len(done) == 320
    all_rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    failures = [row for row in all_rows if row['status'] != 'complete']
    status('complete',successful_requests=len(done),successful_dit_forwards=sum(row['dit_calls'] for row in done.values()),
        failed_requests=len(failures),failed_dit_completed=sum(row.get('dit_calls_completed',0) for row in failures),
        failed_dit_invoked=sum(row.get('dit_calls_invoked',row.get('dit_calls_completed',0)) for row in failures),
        oracle_calls=0,val_sources=16,calibration_sha256=CAL_SHA)


if __name__ == '__main__':
    main()
