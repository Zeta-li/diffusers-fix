import copy
import gc
import hashlib
import json
import shutil
import time
import traceback

import torch

import run_s1 as common
from run_free_pilot import load_pipeline
from train_budget_expansion import euler
from cache_aware_adapters import CacheAdapters
from trainable_budget_executor import TrainableBudgetExecutor
from controlled_mask import tensor_digest

ROOT = common.ROOT


@torch.no_grad()
def main():
    path = ROOT / 'data/budget_rescue_v1_preregistration.json'
    spec = json.loads(path.read_text())
    out = ROOT / 'runs/budget_rescue_v1'
    if out.exists():
        raise RuntimeError('New run directory required')
    for name, digest in spec['code_sha256'].items():
        assert common.sha(ROOT / 'scripts' / name) == digest
    for arm in spec['arms']:
        assert common.sha(ROOT / arm['checkpoint']) == arm['sha256']
    assert common.sha(ROOT / spec['noise']) == spec['noise_sha256']
    assert len(spec['cases']) == 32
    out.mkdir(); (out / 'evaluation').mkdir(); (out / 'code_snapshot').mkdir()
    shutil.copy2(path, out / 'preregistration.json')
    for name in spec['code_sha256']:
        shutil.copy2(ROOT / 'scripts' / name, out / 'code_snapshot' / name)
    counts = dict(invoked_dit=0, completed_dit=0, successful_requests=0)
    active = {}
    def status(phase, **extra):
        common.write(out / 'status.json', dict(phase=phase, active=active, **counts, **extra))
    try:
        status('loading_model')
        pipe = load_pipeline(); pipe.vae.disable_tiling()
        pipe.text_encoder.to('cpu'); gc.collect(); torch.cuda.empty_cache()
        model = pipe.transformer; adapters = CacheAdapters(model)
        weights = {a['name']: torch.load(ROOT / a['checkpoint'], map_location='cpu', weights_only=True)['adapters'] for a in spec['arms']}
        noise = torch.load(ROOT / spec['noise'], map_location='cuda', weights_only=True)
        for case in spec['cases']:
            cid = case['case_id']; folder = out / 'evaluation' / cid; folder.mkdir()
            assert common.sha(ROOT / case['image']) == case['source_sha256']
            condition_path = ROOT / spec['reference_run'] / 'conditioning' / (cid + '.pt')
            assert common.sha(condition_path) == spec['conditioning_sha256'][cid]
            item = torch.load(condition_path, map_location='cpu', weights_only=True)
            assert torch.equal(noise.cpu(), item['hidden_states'][:, :4096])
            fields = {k: v.to('cuda') if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                      for k, v in item.items() if k not in ('hidden_states', 'sigmas', 'timesteps', 'timestep')}
            condition = item['hidden_states'][:, 4096:].to('cuda')
            sigmas = item['sigmas'].to('cuda'); times = item['timesteps'].to('cuda')
            ref = torch.load(ROOT / spec['reference_run'] / 'evaluation' / cid / 'F.pt', map_location='cuda', weights_only=True)['final'].float()
            for arm in spec['arms']:
                name = arm['name']; active = dict(case_id=cid, arm=name); status('evaluation')
                adapters.load_state_dict(weights[name])
                executor = TrainableBudgetExecutor(model, case['plan_B'])
                z = noise.clone(); states = []; enabled_steps = []; started = time.perf_counter()
                for step in range(8):
                    assert executor.step == step
                    enabled = arm['lora_schedule'] == 'all' or step not in executor.full_steps
                    enabled_steps.append(enabled); counts['invoked_dit'] += 1
                    common.append(out / 'calls.jsonl', dict(**active, step=step, adapter_enabled=enabled, invoked=counts['invoked_dit']))
                    with adapters.mode(enabled):
                        v = executor.forward(hidden_states=torch.cat([z, condition], 1), timestep=times[step].reshape(1).to(z.dtype)/1000, **fields)[0][:, :4096]
                    if not torch.isfinite(v).all(): raise RuntimeError('Nonfinite velocity')
                    counts['completed_dit'] += 1
                    z = euler(z, v, sigmas, step); states.append(z.cpu().clone())
                if name == 'LOCAL128': assert enabled_steps == [False, False, True, True, True, True, True, False]
                executor.clear(); im = common.decode(pipe, z); im.save(folder / (name + '.png'))
                torch.save(dict(final=z.cpu(), states=states), folder / (name + '.pt'))
                counts['successful_requests'] += 1
                row = dict(**active, enabled_steps=enabled_steps, latent_mae_vs_F=float((z.float()-ref).abs().mean()),
                           RGB_sha256=hashlib.sha256(im.tobytes()).hexdigest(), final_sha256=tensor_digest(z),
                           diagnostic_seconds=time.perf_counter()-started, net_latency_claim=False, **counts)
                common.write(folder / (name + '.json'), row); common.append(out / 'evaluation.jsonl', row)
                print(json.dumps(row), flush=True)
        assert counts['invoked_dit'] == counts['completed_dit'] == 512
        assert counts['successful_requests'] == 64
        status('complete_analysis_pending', training_updates=0, RL_updates=0, Dev32_opened=False)
    except BaseException as exc:
        status('failed', exception=repr(exc), traceback=traceback.format_exc()); raise


if __name__ == '__main__':
    main()
