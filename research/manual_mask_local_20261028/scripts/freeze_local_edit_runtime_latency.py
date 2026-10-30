import datetime
import json

import run_s1 as common

ROOT = common.ROOT


def main():
    run = ROOT / 'runs/refresh_dev16_v1'
    spec = json.loads((run / 'preregistration.json').read_text())
    numeric = json.loads((run / 'analysis/results.json').read_text())
    visual = json.loads((run / 'blind_review/unblinded_results.json').read_text())
    summary = visual['summary']['all']
    gates = {}
    for arm in ('R6', 'SUP'):
        changes = visual['discordance']['all']['R25_to_' + arm]
        gates[arm] = (numeric['numeric_gates'][arm]
                      and len(changes['regressions']) <= 1
                      and summary[arm]['joint'] >= summary['R25']['joint'])
    passed = [a for a in gates if gates[a]]
    selected = min(passed, key=lambda a: numeric['summary'][a]['surrogate_loss']) if passed else 'R25'
    decision = dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    gates=gates, selected_for_latency=selected,
                    promotion='development_candidate_only; statistical superiority not established',
                    inputs_sha256={str(p.relative_to(ROOT)): common.sha(p) for p in
                        (run/'preregistration.json', run/'analysis/results.json', run/'blind_review/unblinded_results.json')})
    target = ROOT / 'data/local_edit_runtime_latency_v1_preregistration.json'
    assert not target.exists()
    arms = ['F', 'R25'] + ([selected] if selected != 'R25' else [])
    cases = spec['cases'][:8]
    jobs = [dict(case_id=cases[0]['case_id'], arm=a, phase='warmup', repeat=0) for a in arms]
    for repeat in range(3):
        for i, c in enumerate(cases):
            offset = (repeat + i) % len(arms)
            for a in arms[offset:] + arms[:offset]:
                jobs.append(dict(case_id=c['case_id'], arm=a, phase='measure', repeat=repeat))
    names = list(spec['code_sha256']) + ['benchmark_local_edit_runtime.py', 'freeze_local_edit_runtime_latency.py', 'analyze_local_edit_runtime_latency.py']
    result = dict(frozen_at=decision['frozen_at'], cases=cases, arms=arms, jobs=jobs,
                  selected=selected, decision=decision, evaluation_run='runs/refresh_dev16_v1',
                  planned_requests=len(jobs), planned_dit=8*len(jobs),
                  code_sha256={n:common.sha(ROOT/'scripts'/n) for n in names},
                  scope='model resident, source file IO excluded, prompt/source VAE/plan/head/8DiT/untiled decode/PIL/cache cleanup included',
                  statistics='paired source means of3 repetitions; ratio of means; source bootstrap10000 draws; warmups excluded',
                  all_quality_failures_retained=True, training_updates=0, RL_updates=0, test64_opened=False)
    for k in ('checkpoint', 'checkpoint_sha256', 'head', 'head_sha256', 'noise', 'noise_sha256'):
        result[k] = spec[k]
    common.write(run/'analysis/gate_decision.json', decision)
    common.write(target, result)
    print(json.dumps(dict(gates=gates, selected=selected, requests=len(jobs), DiT=8*len(jobs), protocol_sha256=common.sha(target))))


if __name__ == '__main__':
    main()
