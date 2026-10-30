import argparse
import copy
import datetime
import json

import run_s1 as common
from budget_r25_profiles import validate_budget_plan

ROOT = common.ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', choices=['pilot8', 'remaining40'], required=True)
    args = parser.parse_args()
    path = ROOT / ('data/refresh_timing_' + args.subset + '_v1_preregistration.json')
    assert not path.exists()
    parent = ROOT / 'data/budget_r25_v1_preregistration.json'
    old = json.loads(parent.read_text())
    cases = copy.deepcopy(old['cases'][:8] if args.subset == 'pilot8' else old['cases'][8:])
    reused = []
    for case in cases:
        cid = case['case_id']
        assert common.sha(ROOT / case['image']) == case['source_sha256']
        for arm, middle in [('R4', 3), ('R6', 5)]:
            plan = copy.deepcopy(case['plans']['B'])
            plan.update(kind='budget_' + arm, profile=arm, full_steps_zero_based=[0, 1, middle, 7])
            plan['selected_indices_per_step'][middle] = list(range(4096))
            plan['target_tokens_per_step'] = list(map(len, plan['selected_indices_per_step']))
            plan['total_target_token_updates'] = sum(plan['target_tokens_per_step'])
            validate_budget_plan(plan)
            assert plan['total_target_token_updates'] == case['plans']['R25']['total_target_token_updates']
            for step in set(range(8)) - {middle, 4}:
                assert plan['selected_indices_per_step'][step] == case['plans']['R25']['selected_indices_per_step'][step]
            case['plans'][arm] = plan
        for arm in ('F', 'B', 'R25'):
            stem = 'runs/budget_r25_v1/evaluation/' + cid + '/' + arm
            reused.append(dict(case_id=cid, arm=arm, stem=stem,
                sha256={ext: common.sha(ROOT / (stem + '.' + ext)) for ext in ('png', 'pt', 'json')}))
    names = list(dict.fromkeys(list(old['code_sha256']) + ['freeze_refresh_timing.py', 'evaluate_refresh_timing.py', 'analyze_refresh_timing.py']))
    spec = dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), subset=args.subset,
        parent_protocol_sha256=common.sha(parent), cases=cases, reused=reused,
        jobs=[dict(case_id=c['case_id'], arm=a) for c in cases for a in ('R4', 'R6')],
        planned_requests=2 * len(cases), planned_dit=16 * len(cases), evaluation_arms=['F', 'B', 'R4', 'R25', 'R6'],
        checkpoint=old['checkpoint'], checkpoint_sha256=old['checkpoint_sha256'],
        noise=old['noise'], noise_sha256=old['noise_sha256'], conditioning_run=old['conditioning_run'],
        conditioning_sha256={c['case_id']: old['conditioning_sha256'][c['case_id']] for c in cases},
        code_sha256={n: common.sha(ROOT / 'scripts' / n) for n in names},
        contract='Four full steps for R4/R25/R6; identical per-source total target updates and same local per-step indices outside swapped refresh steps; no new weights or masks.',
        continuation_rule='Pilot: expand if best fixed MAE improves at least 2% vs R25 OR oracle improves at least 2% vs best fixed. Otherwise retain R25 and report low opportunity.',
        policy_rule='Train sources only; source-group CV; calibration16 used for a single frozen policy comparison. No tuning on fresh Dev32 or Test64.',
        feature_rule='Only geometry, source/text conditions and common first two full states. No final F/candidate output at inference.',
        training_updates=0, RL_updates=0, net_latency_claim=False, test64_opened=False)
    common.write(path, spec)
    print(json.dumps(dict(path=str(path), sha256=common.sha(path), requests=spec['planned_requests'], DiT=spec['planned_dit'])))


if __name__ == '__main__':
    main()
