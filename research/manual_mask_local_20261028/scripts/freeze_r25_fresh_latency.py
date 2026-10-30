import copy
import datetime
import json

import run_s1 as common
from budget_r25_profiles import make_budget_plan

ROOT = common.ROOT


def main():
    path = ROOT / 'data/r25_fresh_latency_v1_preregistration.json'
    assert not path.exists()
    quality_run = ROOT / 'runs/r25_fresh_dev32_v1'
    decision = json.loads((quality_run / 'analysis/final_decision.json').read_text())
    assert decision['pass_fixed_candidate_validation'] is True
    quality = json.loads((quality_run / 'preregistration.json').read_text())
    cases = copy.deepcopy(quality['cases'][:8])
    for case in cases:
        for arm in ('B', 'R25'):
            assert case['plans'][arm] == make_budget_plan(case, arm)
        case.pop('plans')
    arms = ['F', 'B', 'R25']
    jobs = [dict(case_id=cases[0]['case_id'], arm=arm, phase='warmup', repeat=0) for arm in arms]
    for repeat in range(3):
        for index, case in enumerate(cases):
            shift = (index + repeat) % 3
            for arm in arms[shift:] + arms[:shift]:
                jobs.append(dict(case_id=case['case_id'], arm=arm, phase='measured', repeat=repeat + 1))
    names = ['freeze_r25_fresh_latency.py', 'benchmark_budget_r25.py', 'budget_r25_profiles.py', 'budget_profiles.py',
        'cache_aware_adapters.py', 'trainable_budget_executor.py', 'budget_cache_executor.py', 'run_free_pilot.py',
        'run_s1.py', 'controlled_mask.py', 'free_region.py']
    spec = dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), candidate='R25',
        expanded_checkpoint=quality['checkpoint'], expanded_checkpoint_sha256=quality['checkpoint_sha256'],
        cases=cases, jobs=jobs, arms=arms, noise_sha256=quality['noise_sha256'], evaluation_run='runs/r25_fresh_dev32_v1',
        quality_decision_sha256=common.sha(quality_run / 'analysis/final_decision.json'), planned_DiT=600, planned_requests=75,
        code_sha256={n: common.sha(ROOT / 'scripts' / n) for n in names},
        source_selection='first8 in previously frozen32source order, not selected by quality or speed; all cases remain in quality denominator',
        timing_scope='resident models; source RGB/noise loaded; online plan, input preprocessing, text encode, source VAE, 8DiT, decode, PIL and cache cleanup included; model loading and file IO excluded',
        CUDA_synchronization='before and after complete request', CPU_background='no parallel archive or heavy analysis during timing',
        quality_status='passes frozen single-assistant gate;3 common uncertain sources require human followup, one new failure versusF',
        decision_rule='Report measured paired source mean speedup; repeated hashes must match frozen quality outputs; no quality retuning',
        diagnostic_only=False, test64_opened=False, training_updates=0, RL_updates=0)
    common.write(path, spec)
    print(json.dumps(dict(protocol=str(path), sha256=common.sha(path), requests=75, DiT=600)))


if __name__ == '__main__':
    main()
