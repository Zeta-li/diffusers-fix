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
    protocol = ROOT / 'data/budget_r25_v1_preregistration.json'
    spec = json.loads(protocol.read_text()); out = ROOT / 'runs/budget_r25_v1'
    assert not out.exists()
    for name, digest in spec['code_sha256'].items(): assert common.sha(ROOT / 'scripts' / name) == digest
    assert common.sha(ROOT / spec['checkpoint']) == spec['checkpoint_sha256']
    assert common.sha(ROOT / spec['noise']) == spec['noise_sha256']
    out.mkdir(); (out / 'evaluation').mkdir(); (out / 'code_snapshot').mkdir()
    shutil.copy2(protocol, out / 'preregistration.json')
    for name in spec['code_sha256']: shutil.copy2(ROOT / 'scripts' / name, out / 'code_snapshot' / name)
    counts = dict(invoked_dit=0, completed_dit=0, successful_requests=0)
    active = {}
    def status(phase, **extra): common.write(out / 'status.json', dict(phase=phase, active=active, **counts, **extra))
    try:
        for entry in spec['reused']:
            folder = out / 'evaluation' / entry['case_id']; folder.mkdir(exist_ok=True)
            for ext, digest in entry['sha256'].items():
                src = ROOT / (entry['stem'] + '.' + ext)
                assert common.sha(src) == digest
                shutil.copy2(src, folder / (entry['arm'] + '.' + ext))
        status('loading_model', reused_requests=192)
        pipe = load_pipeline(); pipe.vae.disable_tiling()
        pipe.text_encoder.to('cpu'); gc.collect(); torch.cuda.empty_cache()
        model = pipe.transformer; adapters = CacheAdapters(model)
        adapters.load_state_dict(torch.load(ROOT / spec['checkpoint'], map_location='cpu', weights_only=True)['adapters'])
        noise = torch.load(ROOT / spec['noise'], map_location='cuda', weights_only=True)
        for case in spec['cases']:
            cid = case['case_id']; folder = out / 'evaluation' / cid; folder.mkdir(exist_ok=True)
            assert common.sha(ROOT / case['image']) == case['source_sha256']
            cp = ROOT / spec['conditioning_run'] / 'conditioning' / (cid + '.pt')
            assert common.sha(cp) == spec['conditioning_sha256'][cid]
            item = torch.load(cp, map_location='cpu', weights_only=True)
            assert torch.equal(noise.cpu(), item['hidden_states'][:, :4096])
            fields = {k: v.to('cuda') if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                for k,v in item.items() if k not in ('hidden_states', 'sigmas', 'timesteps', 'timestep')}
            condition = item['hidden_states'][:, 4096:].to('cuda')
            sigmas = item['sigmas'].to('cuda'); times = item['timesteps'].to('cuda')
            for job in [j for j in spec['jobs'] if j['case_id'] == cid]:
                arm = job['arm']; active = dict(case_id=cid, arm=arm); status('evaluation')
                executor = None if arm == 'F' else TrainableBudgetExecutor(model, case['plans'][arm])
                z = noise.clone(); states = []; enabled_steps = []; start = time.perf_counter()
                for step in range(8):
                    if executor: assert executor.step == step
                    enabled = executor is not None and step not in executor.full_steps
                    enabled_steps.append(enabled); counts['invoked_dit'] += 1
                    common.append(out / 'calls.jsonl', dict(**active, step=step, adapter_enabled=enabled, invoked=counts['invoked_dit']))
                    with adapters.mode(enabled):
                        fn = model.forward if executor is None else executor.forward
                        v = fn(hidden_states=torch.cat([z, condition], 1), timestep=times[step].reshape(1).to(z.dtype)/1000, **fields)[0][:, :4096]
                    if not torch.isfinite(v).all(): raise RuntimeError('Nonfinite velocity')
                    counts['completed_dit'] += 1
                    z = euler(z, v, sigmas, step); states.append(z.cpu().clone())
                if executor: executor.clear()
                if arm == 'F': ref = z.float().clone()
                else: ref = torch.load(folder / 'F.pt', map_location='cuda', weights_only=True)['final'].float()
                im = common.decode(pipe, z); im.save(folder / (arm + '.png'))
                torch.save(dict(final=z.cpu(), states=states), folder / (arm + '.pt'))
                counts['successful_requests'] += 1
                row = dict(**active, role=case['role'], enabled_steps=enabled_steps,
                    latent_mae_vs_F=float((z.float()-ref).abs().mean()), RGB_sha256=hashlib.sha256(im.tobytes()).hexdigest(),
                    final_sha256=tensor_digest(z), diagnostic_seconds=time.perf_counter()-start, net_latency_claim=False, **counts)
                common.write(folder / (arm + '.json'), row); common.append(out / 'evaluation.jsonl', row)
                print(json.dumps(row), flush=True)
        assert counts['invoked_dit'] == counts['completed_dit'] == spec['planned_dit'] == 384
        assert counts['successful_requests'] == spec['planned_requests'] == 48
        status('complete_analysis_pending', reused_requests=192, training_updates=0, RL_updates=0)
    except BaseException as exc:
        status('failed', exception=repr(exc), traceback=traceback.format_exc()); raise


if __name__ == '__main__': main()
