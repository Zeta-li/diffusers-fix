import argparse
import inspect
import json
import shutil
import subprocess
import time
import traceback
from contextlib import nullcontext

import torch
from PIL import Image

import run_s1 as common
from context_refresh import install_context_refresh
from free_region import run_free_region
from run_free_pilot import load_pipeline, make_plan

ROOT = common.ROOT


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    source = ROOT / args.protocol
    spec = json.loads(source.read_text())
    out = ROOT / 'runs' / args.attempt
    if out.exists() or '/' in args.attempt or spec['phase'] != 'FROZEN_BEFORE_GENERATION':
        raise RuntimeError('New attempt and frozen protocol required')
    for name, digest in spec['code_sha256'].items():
        if common.sha(ROOT / 'scripts' / name) != digest:
            raise RuntimeError('Frozen code changed: ' + name)
    for receipt_path in spec['cpu_receipts']:
        receipt = json.loads((ROOT / receipt_path).read_text())
        if receipt['exit_code'] != 0:
            raise RuntimeError('CPU contracts failed')
        for name, digest in receipt['code_sha256'].items():
            if common.sha(ROOT / 'scripts' / name) != digest:
                raise RuntimeError('CPU-tested code changed: ' + name)
    arms = ['F', 'FIX32', 'FRESH64', 'CONTEXT64']
    planned = len(spec['cases']) * len(arms)
    if (planned, planned * 8) != (spec['planned_requests'], spec['planned_dit']):
        raise RuntimeError('Wrong declared budget')
    for case in spec['cases']:
        if common.sha(ROOT / case['image']) != case['source_sha256']:
            raise RuntimeError('Source changed')
    out.mkdir()
    shutil.copy2(source, out / 'preregistration.json')
    (out / 'code_snapshot').mkdir()
    for name in spec['code_sha256']:
        shutil.copy2(ROOT / 'scripts' / name, out / 'code_snapshot' / name)
    counts = dict(invoked_dit=0, completed_dit=0, successful_requests=0)
    active = {}

    def status(phase, **extra):
        common.write(out / 'status.json', dict(phase=phase, updated_unix=time.time(),
                     **counts, **{k:v for k,v in active.items() if k != 'velocities'}, **extra))

    def before(module, positional, fields):
        if positional or active['step'] >= 8 or fields['hidden_states'].shape != (1, 8192, 64):
            raise RuntimeError('Unexpected DiT geometry or step')
        counts['invoked_dit'] += 1
        common.append(out / 'calls.jsonl', dict(event='invoked', **counts,
                      **{k:v for k,v in active.items() if k != 'velocities'}))

    def after(module, positional, result):
        counts['completed_dit'] += 1
        active['velocities'].append(result[0][:, :4096].detach().cpu().clone())
        common.append(out / 'calls.jsonl', dict(event='completed', **counts,
                      **{k:v for k,v in active.items() if k != 'velocities'}))
        active['step'] += 1

    status('loading_model')
    try:
        pipe = load_pipeline()
        noise = torch.load(ROOT / 'runs/s1_v2/explicit_noise.pt', weights_only=True, map_location='cpu')
        if common.sha(ROOT / 'runs/s1_v2/explicit_noise.pt') != spec['noise_sha256']:
            raise RuntimeError('Frozen noise changed')
        torch.save(noise, out / 'explicit_noise.pt')
        common.write(out / 'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
            user_confirmed_gpu='H800', driver_name=torch.cuda.get_device_name(),
            scheduler=dict(pipe.scheduler.config), native_pipeline_sha256=common.sha(inspect.getfile(type(pipe))),
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            weights_receipt_sha256=common.sha('/workspace/model-weights/deployment_20261028.json')))
        pre = pipe.transformer.register_forward_pre_hook(before, with_kwargs=True)
        post = pipe.transformer.register_forward_hook(after)
        for case in spec['cases']:
            dst = out / case['case_id']
            dst.mkdir()
            image = Image.open(ROOT / case['image']).convert('RGB')
            if list(image.size) != case['source_wh']:
                raise RuntimeError('Source dimensions differ')
            plan = make_plan(case, dict(kind='front', margin=32))
            refs = {}
            budgets = {}
            for arm in arms:
                active = dict(case_id=case['case_id'], arm=arm, step=0, velocities=[])
                status('running')
                context = install_context_refresh(pipe, plan, preserve_extra_velocity=arm=='CONTEXT64',
                    extra_tokens=64, steps=(3, 5)) if arm in ('FRESH64','CONTEXT64') else nullcontext(None)
                with context as execution:
                    result = run_free_region(pipe, image, case['prompt'], noise.to('cuda'),
                                             plan=plan if arm=='FIX32' else None)
                record = pipe.last_free_region_run
                raw = dict(final=result.images.cpu(), states=record.step_latents, velocities=active['velocities'])
                torch.save(raw, dst / (arm + '_raw.pt'))
                previous = noise
                for k, state in enumerate(record.step_latents):
                    dt = pipe.scheduler.sigmas[k+1] - pipe.scheduler.sigmas[k]
                    expected = (previous.to('cuda').float() + dt * active['velocities'][k].to('cuda')).to(state.dtype).cpu()
                    if not torch.equal(expected, state):
                        raise RuntimeError('Euler mismatch')
                    previous = state
                if arm != 'F':
                    old, old_record = refs['F']
                    for key in ('initial_noise_sha256','reference_condition_sha256','text_sha256','text_mask_sha256'):
                        if getattr(record,key) != getattr(old_record,key):
                            raise RuntimeError('Paired inputs differ')
                    if not all(torch.equal(a,b) for a,b in zip(raw['states'][:2], old['states'][:2])):
                        raise RuntimeError('First two full states differ')
                if execution:
                    if not all(torch.equal(a,b) for a,b in zip(raw['states'][:3], refs['FIX32'][0]['states'][:3])):
                        raise RuntimeError('Intervention prefix differs')
                    if not all(g['extra_kv_fresh'] for g in execution['gates']):
                        raise RuntimeError('KV refresh failed')
                    if arm == 'CONTEXT64' and not all(g['extra_velocity_retained'] for g in execution['gates']):
                        raise RuntimeError('Velocity retention failed')
                    budgets[arm] = execution['plan']['target_tokens_per_step']
                    if sum(budgets[arm]) != plan['total_target_token_updates'] + 128:
                        raise RuntimeError('Extra budget differs')
                else:
                    budgets[arm] = [4096]*8 if arm == 'F' else plan['target_tokens_per_step']
                common.decode(pipe, result.images).save(dst / (arm + '.png'))
                common.write(dst / (arm + '.json'), dict(record=record.manifest(),
                    execution=None if execution is None else {k:v for k,v in execution.items() if k != 'velocities'},
                    target_tokens_per_step=budgets[arm], checks=dict(all_Euler_bitwise=True,
                    input_pairing=True, preintervention_prefix_bitwise=True), net_latency_claim=False))
                refs[arm] = raw, record
                counts['successful_requests'] += 1
                common.append(out / 'requests.jsonl', dict(case_id=case['case_id'],arm=arm,**counts))
                status('running')
                print(case['case_id'], arm, counts['successful_requests'], '/', planned, flush=True)
            if budgets['FRESH64'] != budgets['CONTEXT64']:
                raise RuntimeError('Equal-budget pairing failed')
        pre.remove()
        post.remove()
        if counts != dict(invoked_dit=planned*8, completed_dit=planned*8, successful_requests=planned):
            raise RuntimeError('Final ledger mismatch')
        status('generation_complete_quality_pending')
    except BaseException as exc:
        status('failed', error=type(exc).__name__ + ': ' + str(exc))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
