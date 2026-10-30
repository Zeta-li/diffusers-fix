import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

from PIL import Image
import torch

import run_s1 as common
from run_free_pilot import load_pipeline
from train_budget_expansion import capture_condition, euler
from cache_aware_adapters import CacheAdapters
from trainable_budget_executor import TrainableBudgetExecutor
from controlled_mask import tensor_digest
from local_edit_runtime import GeometryRefreshSelector, request_from_case, resolve_request, export_backend_plan

ROOT = common.ROOT


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    path = ROOT / args.protocol
    spec = json.loads(path.read_text())
    out = ROOT / 'runs' / args.attempt
    assert Path(args.attempt).name == args.attempt and not out.exists()
    assert spec['evaluation_arms'] == ['F', 'R25', 'R6', 'SUP']
    assert spec['planned_requests'] == 4 * len(spec['cases'])
    assert spec['planned_dit'] == 8 * spec['planned_requests']
    for name, value in spec['code_sha256'].items():
        assert common.sha(ROOT / 'scripts' / name) == value, name
    for key in ('checkpoint', 'noise', 'head'):
        assert common.sha(ROOT / spec[key]) == spec[key + '_sha256']
    out.mkdir()
    for folder in ('evaluation', 'conditioning', 'code_snapshot'):
        (out / folder).mkdir()
    shutil.copy2(path, out / 'preregistration.json')
    for name in spec['code_sha256']:
        shutil.copy2(ROOT / 'scripts' / name, out / 'code_snapshot' / name)
    counts = dict(invoked_dit=0, completed_dit=0, successful_requests=0, conditioning_only=0)
    active = {}

    def status(phase, **extra):
        common.write(out / 'status.json', dict(phase=phase, active=active, **counts, **extra))

    try:
        status('loading_model')
        pipe = load_pipeline()
        pipe.vae.disable_tiling()
        model = pipe.transformer
        selector = GeometryRefreshSelector(ROOT / spec['head'])
        noise = torch.load(ROOT / spec['noise'], map_location='cuda', weights_only=True)
        assert noise.shape == (1, 4096, 64) and noise.dtype == torch.bfloat16
        if spec['conditioning_mode'] == 'capture':
            for case in spec['cases']:
                cid = case['case_id']
                assert common.sha(ROOT / case['image']) == case['source_sha256']
                active = dict(case_id=cid)
                status('preparing_conditions')
                item = capture_condition(pipe, Image.open(ROOT / case['image']).convert('RGB'), case['prompt'], noise)
                assert torch.equal(item['hidden_states'][:, :4096], noise.cpu())
                torch.save(item, out / 'conditioning' / (cid + '.pt'))
                counts['conditioning_only'] += 1
        pipe.text_encoder.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        adapters = CacheAdapters(model)
        adapters.load_state_dict(torch.load(ROOT / spec['checkpoint'], map_location='cpu', weights_only=True)['adapters'])
        for case in spec['cases']:
            cid = case['case_id']
            assert common.sha(ROOT / case['image']) == case['source_sha256']
            folder = out / 'evaluation' / cid
            folder.mkdir()
            if spec['conditioning_mode'] == 'cached':
                cp = ROOT / spec['conditioning_run'] / 'conditioning' / (cid + '.pt')
                assert common.sha(cp) == spec['conditioning_sha256'][cid]
            else:
                cp = out / 'conditioning' / (cid + '.pt')
            item = torch.load(cp, map_location='cpu', weights_only=True)
            assert item['hidden_states'].shape[1] == 8192
            fields = {k: v.to('cuda') if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                      for k, v in item.items() if k not in ('hidden_states', 'sigmas', 'timesteps', 'timestep')}
            condition = item['hidden_states'][:, 4096:].to('cuda')
            sigmas = item['sigmas'].to('cuda')
            times = item['timesteps'].to('cuda')
            assert len(times) == 8 and len(sigmas) == 9
            for arm in spec['evaluation_arms']:
                active = dict(case_id=cid, arm=arm)
                status('evaluation')
                plan, runtime_metadata = resolve_request(request_from_case(case), arm, selector)
                if arm != 'F':
                    assert plan == case['plans'][arm]
                if arm == 'SUP':
                    assert runtime_metadata == case['SUP_precomputed_source_only']
                common.write(folder / (arm + '_backend_plan.json'), export_backend_plan(plan, runtime_metadata))
                executor = None if plan is None else TrainableBudgetExecutor(model, plan)
                z = noise.clone()
                states = []
                enabled_steps = []
                started = time.perf_counter()
                for step in range(8):
                    if executor:
                        assert executor.step == step
                    enabled = executor is not None and step not in executor.full_steps
                    enabled_steps.append(enabled)
                    counts['invoked_dit'] += 1
                    common.append(out / 'calls.jsonl', dict(**active, step=step, adapter_enabled=enabled, invoked=counts['invoked_dit']))
                    with adapters.mode(enabled):
                        fn = model.forward if executor is None else executor.forward
                        v = fn(hidden_states=torch.cat([z, condition], 1), timestep=times[step].reshape(1).to(z.dtype) / 1000, **fields)[0][:, :4096]
                    assert torch.isfinite(v).all()
                    counts['completed_dit'] += 1
                    z = euler(z, v, sigmas, step)
                    states.append(z.cpu().clone())
                if executor:
                    executor.clear()
                if arm == 'F':
                    ref = z.float().clone()
                im = common.decode(pipe, z)
                im.save(folder / (arm + '.png'))
                torch.save(dict(final=z.cpu(), states=states), folder / (arm + '.pt'))
                counts['successful_requests'] += 1
                row = dict(**active, runtime_metadata=runtime_metadata, role=case['role'], seed=spec['seed'], enabled_steps=enabled_steps,
                    latent_mae_vs_F=float((z.float() - ref).abs().mean()), RGB_sha256=hashlib.sha256(im.tobytes()).hexdigest(),
                    final_sha256=tensor_digest(z), diagnostic_seconds=time.perf_counter() - started, net_latency_claim=False, **counts)
                common.write(folder / (arm + '.json'), row)
                common.append(out / 'evaluation.jsonl', row)
                print(json.dumps(row), flush=True)
        assert counts['invoked_dit'] == counts['completed_dit'] == spec['planned_dit']
        assert counts['successful_requests'] == spec['planned_requests']
        status('complete_analysis_pending', training_updates=0, RL_updates=0, test64_opened=False)
    except BaseException as exc:
        status('failed', exception=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
