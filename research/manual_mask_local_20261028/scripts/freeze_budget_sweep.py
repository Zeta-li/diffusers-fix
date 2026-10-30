import collections
import datetime
import json
from pathlib import Path

import run_s1 as common
from budget_profiles import make_budget_plan as old_plan
from budget_sweep_profiles import make_budget_plan

ROOT = common.ROOT


def main():
    dest = ROOT / 'data/budget_sweep_v1_preregistration.json'
    assert not dest.exists()
    split = json.loads((ROOT / 'data/budget_expansion_split_v1.json').read_text())
    prior = json.loads((ROOT / 'data/budget_expansion_train_v1_preregistration.json').read_text())
    candidate = json.loads((ROOT / 'data/budget_local128_candidate_v1.json').read_text())
    cases = [c for c in split['cases'] if c['role'] == 'train'][:32]
    cases += [c for c in split['cases'] if c['role'] == 'calibration'][:16]
    assert len(cases) == 48 and len({c['source_group_id'] for c in cases}) == 48
    frozen = {c['case_id']: c for c in prior['cases']}
    jobs, reused, conditioning = [], [], {}
    hist = collections.Counter()
    for case in cases:
        cid = case['case_id']
        assert common.sha(ROOT / case['image']) == case['source_sha256']
        plans = {arm: make_budget_plan(case, arm) for arm in ('B', 'B50', 'C')}
        assert plans['B'] == frozen[cid]['plan_B'] == old_plan(case, 'B')
        assert plans['C'] == old_plan(case, 'C')
        assert plans['B50']['full_steps_zero_based'] == [0, 1, 7]
        assert plans['C']['full_steps_zero_based'] == [0, 1, 4, 7]
        for step in (0, 1, 2, 3, 5, 6, 7):
            assert plans['B50']['selected_indices_per_step'][step] == plans['C']['selected_indices_per_step'][step]
        assert plans['B50']['effective_local_fraction'] >= plans['B']['effective_local_fraction']
        case['plans'] = plans
        for arm, plan in plans.items(): hist[(case['role'], arm, plan['effective_local_fraction'])] += 1
        cp = ROOT / 'runs/budget_expansion_train_v1/conditioning' / (cid + '.pt')
        conditioning[cid] = common.sha(cp)
        for arm in ('F', 'B', 'B50', 'C'):
            if case['role'] == 'calibration' and arm in ('F', 'B'):
                run, name = ('budget_expansion_train_v1', 'F') if arm == 'F' else ('budget_rescue_v1', 'LOCAL128')
                stem = 'runs/' + run + '/evaluation/' + cid + '/' + name
                reused.append(dict(case_id=cid, arm=arm, stem=stem,
                    sha256={ext: common.sha(ROOT / (stem + '.' + ext)) for ext in ('png', 'pt', 'json')}))
            else: jobs.append(dict(case_id=cid, arm=arm))
    names = ['freeze_budget_sweep.py', 'evaluate_budget_sweep.py', 'budget_sweep_profiles.py',
        'budget_profiles.py', 'train_budget_expansion.py', 'cache_aware_adapters.py',
        'trainable_budget_executor.py', 'budget_cache_executor.py', 'run_free_pilot.py',
        'run_s1.py', 'controlled_mask.py', 'free_region.py']
    checkpoint = 'runs/budget_expansion_train_v1/checkpoints/student_inference.pt'
    assert common.sha(ROOT / checkpoint) == '216794f27aacf696af7835612234dcdf04cb7dcc6dbfbc9ecffcfa3ef55332f7'
    spec = dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='Frozen LOCAL128 spatial-budget and middle-refresh comparison before budget policy fitting',
        selection='First32 frozen training sources and first16 frozen calibration sources in original order; no Dev selection',
        cases=cases, evaluation_sources=[c['case_id'] for c in cases], evaluation_arms=['F', 'B', 'B50', 'C'],
        jobs=jobs, reused=reused, planned_requests=len(jobs), planned_dit=len(jobs)*8,
        checkpoint=checkpoint, checkpoint_sha256=common.sha(ROOT / checkpoint),
        noise=prior['noise'], noise_sha256=prior['noise_sha256'],
        conditioning_run='runs/budget_expansion_train_v1', conditioning_sha256=conditioning,
        code_sha256={n: common.sha(ROOT / 'scripts' / n) for n in names},
        split_sha256=common.sha(ROOT / 'data/budget_expansion_split_v1.json'),
        adapter_schedule='enabled only when step not in plan full anchors; rank8 unmerged',
        hard_mask=False, pixel_copy=False, training_updates=0, RL_updates=0, test64_opened=False,
        equal_total_budget=False, novelty='B to B50 spatial minimum; B50 to C extra full anchor at step5',
        quality='Paired RGB MAE versus F is fidelity only; anonymous visual review for edit success and preservation; no independent generalization claim',
        selection_rule='Report every source and actual token budget. Train-only per-source labels; Cal only selects thresholds. Do not choose a policy using Dev32.',
        timing_claim=False, plan_contract=dict(cases=48, old_B_exact=48, old_C_exact=48, B50_C_shared_steps_exact=336),
        budget_histogram=[dict(role=k[0], arm=k[1], fraction=k[2], count=v) for k,v in hist.items()])
    assert spec['planned_dit'] == 1280 and spec['planned_requests'] == 160 and len(reused) == 32
    common.write(dest, spec)
    print(json.dumps(dict(path=str(dest), sha256=common.sha(dest), requests=len(jobs), DiT=len(jobs)*8, reused=len(reused), hist=spec['budget_histogram'])))


if __name__ == '__main__': main()
