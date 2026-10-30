import argparse
import hashlib
import json
import shutil
import time
import traceback

import torch
from PIL import Image

import run_s1 as common
from run_free_pilot import load_pipeline
from budget_profiles import make_budget_plan
from trainable_budget_executor import TrainableBudgetExecutor
from cache_aware_adapters import CacheAdapters

ROOT = common.ROOT


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='data/budget_merge_v1_preregistration.json')
    parser.add_argument('--attempt', default='budget_merge_v1')
    args = parser.parse_args()
    spec = json.loads((ROOT / args.protocol).read_text())
    out = ROOT / 'runs' / args.attempt
    if out.exists():
        raise RuntimeError('Preserve existing attempt')
    for n, d in spec['code_sha256'].items():
        if common.sha(ROOT / 'scripts' / n) != d:
            raise RuntimeError('Code changed')
    if common.sha(ROOT / spec['checkpoint']) != spec['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed')
    out.mkdir(); (out / 'code_snapshot').mkdir()
    shutil.copy2(ROOT / args.protocol, out / 'preregistration.json')
    for n in spec['code_sha256']:
        shutil.copy2(ROOT / 'scripts' / n, out / 'code_snapshot' / n)
    counts = dict(invoked_dit=0, completed_dit=0, successful_requests=0)
    active = {}
    def status(phase, **extra):
        common.write(out / 'status.json', dict(phase=phase, active=active, **counts, **extra))
    try:
        status('loading_model')
        pipe = load_pipeline(); pipe.vae.disable_tiling()
        model = pipe.transformer; adapters = CacheAdapters(model)
        ckpt = torch.load(ROOT / spec['checkpoint'], weights_only=True, map_location='cpu')
        adapters.load_state_dict(ckpt['adapters']); del ckpt
        original_weights = {}; merged_weights = {}; energies = []
        for name, layer in adapters.layers.items():
            original_weights[name] = layer.base.weight.detach().clone()
            delta = layer.lora_B @ layer.lora_A
            layer.merge()
            merged_weights[name] = layer.base.weight.detach().clone()
            actual = layer.base.weight.float() - original_weights[name].float()
            energies.append(dict(name=name, delta_squared=float(delta.square().sum()),
                                 rounding_error_squared=float((actual - delta).square().sum()),
                                 changed_fraction=float((actual != 0).float().mean())))
            del delta, actual
        common.write(out / 'merge_diagnostics.json', dict(layers=energies,
            swap_scaffolding_GiB=sum(t.numel() * t.element_size() for t in list(original_weights.values()) + list(merged_weights.values())) / 2**30,
            inference_precision='base and merged weights BF16; unmerged adapter matmuls FP32'))
        noise = torch.load(ROOT / 'runs/s1_v2/explicit_noise.pt', weights_only=True, map_location='cuda')
        images = {c['case_id']: Image.open(ROOT / c['image']).convert('RGB') for c in spec['cases']}
        cases = {c['case_id']: c for c in spec['cases']}
        def before(*args): counts['invoked_dit'] += 1
        def after(*args): counts['completed_dit'] += 1
        pre = model.register_forward_pre_hook(before); post = model.register_forward_hook(after)
        original_forward = model.forward; hashes = {}
        for index, job in enumerate(spec['jobs']):
            active = dict(index=index, **job); status('running')
            cid, arm = job['case_id'], job['arm']; case = cases[cid]
            for name, layer in adapters.layers.items():
                layer.base.weight.copy_((merged_weights if arm == 'MERGED' else original_weights)[name])
                layer.merged = arm == 'MERGED'
                layer.enabled = arm != 'F'
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            executor = None
            if arm != 'F':
                executor = TrainableBudgetExecutor(model, make_budget_plan(case, 'B'))
                model.forward = executor.forward
            try:
                result = pipe(image=images[cid], prompt=case['prompt'], true_cfg_scale=1, num_inference_steps=8,
                    num_images_per_prompt=1, height=1024, width=1024, latents=noise.clone())
                if executor and executor.step != 8:
                    raise RuntimeError('Incomplete request')
            finally:
                model.forward = original_forward
                if executor: executor.clear()
            torch.cuda.synchronize(); seconds = time.perf_counter() - started
            im = result.images[0]; digest = hashlib.sha256(im.tobytes()).hexdigest()
            key = (cid, arm)
            if arm in ('F', 'UNMERGED'):
                expected_arm = 'F' if arm == 'F' else 'LORA_ONLY'
                expected = json.loads((ROOT / 'runs/budget_student_pilot_v1/evaluation' / cid / (expected_arm + '.json')).read_text())
                if digest != expected['RGB_sha256']:
                    raise RuntimeError('Unmerged/base changed from prior validated result')
            if key in hashes and hashes[key] != digest:
                raise RuntimeError('Repeat output changed')
            hashes[key] = digest
            folder = out / cid; folder.mkdir(exist_ok=True)
            stem = arm + '_' + job['phase'] + '_r' + str(job['repeat'])
            im.save(folder / (stem + '.png')); counts['successful_requests'] += 1
            row = dict(**job, seconds=seconds, peak_GiB_including_swap_scaffolding=torch.cuda.max_memory_allocated()/2**30,
                       RGB_sha256=digest, **counts)
            common.write(folder / (stem + '.json'), row); common.append(out / 'records.jsonl', row)
            print(json.dumps(dict(job=index+1, arm=arm, seconds=seconds, **counts)), flush=True)
        pre.remove(); post.remove()
        assert counts['completed_dit'] == spec['planned_DiT']
        status('complete_analysis_pending')
    except BaseException as exc:
        status('failed', exception=repr(exc), traceback=traceback.format_exc()); raise


if __name__ == '__main__':
    main()
