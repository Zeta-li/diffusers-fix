import datetime
import json

import torch

import run_s1 as common
from budget_r25_profiles import make_budget_plan

ROOT = common.ROOT


def main():
    path = ROOT / 'data/r25_seed2_v1_preregistration.json'
    assert not path.exists()
    old_path = ROOT / 'data/budget_r25_v1_preregistration.json'
    old = json.loads(old_path.read_text())
    noise_path = ROOT / 'data/r25_seed2_noise.pt'
    assert not noise_path.exists()
    seed = 2026103102
    noise = torch.randn((1, 4096, 64), generator=torch.Generator(device='cpu').manual_seed(seed), dtype=torch.bfloat16)
    assert not torch.equal(noise, torch.load(ROOT / old['noise'], map_location='cpu', weights_only=True))
    torch.save(noise, noise_path)
    cases = old['cases']
    assert len(cases) == 48
    for case in cases:
        for arm in ('B', 'R25'):
            assert case['plans'][arm] == make_budget_plan(case, arm)
        assert common.sha(ROOT / case['image']) == case['source_sha256']
    names = ['freeze_r25_seed2.py', 'evaluate_r25_validation.py', 'budget_r25_profiles.py',
             'budget_profiles.py', 'train_budget_expansion.py', 'cache_aware_adapters.py',
             'trainable_budget_executor.py', 'budget_cache_executor.py', 'run_free_pilot.py',
             'run_s1.py', 'controlled_mask.py', 'free_region.py']
    spec = dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='Second-noise-seed stability of frozen R25, same Train32 and Cal16',
        prior_protocol_sha256=common.sha(old_path), cases=cases,
        evaluation_arms=['F', 'B', 'R25'], visual_arms=['F', 'B', 'R25'],
        evaluation_sources=[c['case_id'] for c in cases], seed=seed,
        noise=str(noise_path.relative_to(ROOT)), noise_sha256=common.sha(noise_path),
        noise_generation='torch.randn CPU generator, BF16 packed shape1x4096x64; same noise for every arm and source',
        checkpoint=old['checkpoint'], checkpoint_sha256=old['checkpoint_sha256'],
        conditioning_mode='cached', conditioning_run=old['conditioning_run'], conditioning_sha256=old['conditioning_sha256'],
        conditioning_note='Source/text/scheduler fields reused; cached target-noise prefix is discarded and replaced by new explicit noise.',
        planned_requests=144, planned_dit=1152,
        code_sha256={name:common.sha(ROOT / 'scripts' / name) for name in names},
        gate=dict(Cal_mean_MAE_reduction_vs_B_at_least=0.10, Cal_improved_sources_at_least=10,
                  Cal_new_joint_regressions_vs_B_at_most=1, Train_mean_MAE_R25_lower_than_B=True),
        next_gate='If numeric and semantic stability passes, freeze new independentDev32 F/B/R25 without changing weights, plans or prompts based on generated outcomes.',
        visual_rule='Anonymous A/B/C labels; prior sources seen; one assistant, no human consensus. Sign relocation445972 is preservation failure; ambiguous item is not joint pass.',
        hard_mask=False, pixel_copy=False, training_updates=0, RL_updates=0,
        independent_validation=False, test64_opened=False, new_latency_claim=False)
    common.write(path, spec)
    print(json.dumps(dict(protocol=str(path), sha256=common.sha(path), seed=seed,
                         noise_sha256=spec['noise_sha256'], requests=144, DiT=1152)))


if __name__ == '__main__':
    main()
