import argparse
import json
import shutil
import time
import traceback

import torch
from PIL import Image

import run_s1 as common
from run_free_pilot import load_pipeline, make_plan
from free_region import run_free_region
from refresh_supervision import install_probe

ROOT = common.ROOT


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    protocol = ROOT/args.protocol
    spec = json.loads(protocol.read_text())
    out = ROOT/'runs'/args.attempt
    if out.exists() or '/' in args.attempt or spec['phase'] != 'FROZEN_BEFORE_GENERATION':
        raise RuntimeError('frozen protocol and unique attempt required')
    for name, digest in spec['code_sha256'].items():
        if common.sha(ROOT/'scripts'/name) != digest:
            raise RuntimeError('code changed after freeze')
    if json.loads((ROOT/spec['cpu_receipt']).read_text())['exit_code'] != 0:
        raise RuntimeError('CPU probe contracts failed')
    per_case = 8+len(spec['teacher_steps'])+len(spec['probe_steps'])*len(spec['candidates'])
    if spec['planned_dit'] != per_case*len(spec['cases']):
        raise RuntimeError('unexpected collection budget')
    out.mkdir()
    shutil.copy2(protocol, out/'preregistration.json')
    (out/'code_snapshot').mkdir()
    for name in spec['code_sha256']:
        shutil.copy2(ROOT/'scripts'/name, out/'code_snapshot'/name)
    counts = dict(invoked_dit=0, completed_dit=0, successful_requests=0,
                  student_invoked=0, student_completed=0, teacher_invoked=0,
                  teacher_completed=0, counterfactual_invoked=0, counterfactual_completed=0)
    active = {}
    def status(phase, **kw):
        common.write(out/'status.json', dict(phase=phase, updated_unix=time.time(), **counts, **active, **kw))
    def call(event, kind, step, action):
        counts[event+'_dit'] += 1
        counts[kind+'_'+event] += 1
        common.append(out/'calls.jsonl', dict(event=event, kind=kind, step=step, action=action, **active, **counts))
    status('loading_model')
    try:
        pipe = load_pipeline()
        noise = torch.load(ROOT/'runs/s1_v2/explicit_noise.pt', weights_only=True, map_location='cpu')
        torch.save(noise, out/'explicit_noise.pt')
        common.write(out/'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
            user_confirmed_gpu='H800', driver_name=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()), scheduler=dict(pipe.scheduler.config),
            weights_receipt_sha256=common.sha('/workspace/model-weights/deployment_20261028.json')))
        for case in spec['cases']:
            if common.sha(ROOT/case['image']) != case['source_sha256']:
                raise RuntimeError('source changed')
            active.update(case_id=case['case_id'])
            status('running')
            dst = out/case['case_id']
            dst.mkdir()
            plan = make_plan(case, dict(kind='front', margin=32))
            common.write(dst/'FIX32_plan.json', plan)
            def observation(row):
                torch.save(row, dst/f"step{row['step']}.pt")
                compact = {k: v for k, v in row.items() if not isinstance(v, torch.Tensor) and k != 'actions'}
                compact['actions'] = [{k: v for k, v in a.items() if not isinstance(v, torch.Tensor)} for a in row['actions']]
                common.write(dst/f"step{row['step']}.json", compact)
            with install_probe(pipe, plan, call, observation, budget=spec['extra_tokens'], seed=spec['seed'],
                               teacher_steps=spec['teacher_steps'], probe_steps=spec['probe_steps'],
                               candidate_mode=spec.get('candidate_mode', 'diagnostic')) as receipt:
                result = run_free_region(pipe, Image.open(ROOT/case['image']).convert('RGB'),
                    case['prompt'], noise.to('cuda'), trace_latents=True)
            record = pipe.last_free_region_run
            torch.save(dict(final=result.images.cpu(), states=record.step_latents,
                velocities=receipt['velocities']), dst/'FIX32_raw.pt')
            ref = ROOT/'runs/free_fixed32_v1'/case['case_id']
            old = torch.load(ref/'FIX32.pt', weights_only=True, map_location='cpu')
            old_record = json.loads((ref/'FIX32.json').read_text())['record']
            for name in ('initial_noise_sha256', 'reference_condition_sha256', 'text_sha256', 'text_mask_sha256'):
                if old_record[name] != getattr(record, name):
                    raise RuntimeError('reference inputs differ')
            if not all(torch.equal(a, b) for a, b in zip(record.step_latents, old['states'])):
                raise RuntimeError('diagnosis changed FIX32 trajectory')
            previous = noise
            for k, state in enumerate(record.step_latents):
                dt = pipe.scheduler.sigmas[k+1]-pipe.scheduler.sigmas[k]
                expected = (previous.to('cuda').float()+dt*receipt['velocities'][k].to('cuda')).to(state.dtype).cpu()
                if not torch.equal(expected, state):
                    raise RuntimeError('Euler update changed')
                previous = state
            common.decode(pipe, result.images).save(dst/'FIX32.png')
            common.write(dst/'checks.json', dict(all_eight_states_bitwise_archived_FIX32=True,
                all_tokens_native_Euler=True, diagnostic_calls_do_not_affect_student=True,
                trace=receipt['trace'], age_trace=receipt['age_trace']))
            counts['successful_requests'] += 1
            status('running')
        if counts['completed_dit'] != spec['planned_dit'] or counts['invoked_dit'] != spec['planned_dit']:
            raise RuntimeError('final ledger mismatch')
        status('collection_complete_analysis_pending')
    except BaseException as exc:
        status('failed', error=type(exc).__name__+': '+str(exc))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
