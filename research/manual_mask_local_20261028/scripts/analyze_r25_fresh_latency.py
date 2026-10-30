import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / 'runs/r25_fresh_latency_v1'
    spec = json.loads((run / 'preregistration.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    rows = [json.loads(line) for line in (run / 'records.jsonl').read_text().splitlines()]
    assert status['phase'] == 'complete_analysis_pending'
    assert status['successful_requests'] == len(rows) == 75
    assert status['invoked_dit'] == status['completed_dit'] == 600
    assert len([r for r in rows if r['phase'] == 'warmup']) == 3
    pairs = [r for r in rows if r['phase'] == 'measured']
    arms = spec['arms']
    matrix = np.empty((8, 3))
    source = []
    for index, case in enumerate(spec['cases']):
        cid = case['case_id']
        group = {}
        for j, arm in enumerate(arms):
            items = [r for r in pairs if r['case_id'] == cid and r['arm'] == arm]
            assert len(items) == 3 and len({r['RGB_sha256'] for r in items}) == 1
            expected = json.loads((ROOT / spec['evaluation_run'] / 'evaluation' / cid / (arm + '.json')).read_text())
            assert all(r['RGB_sha256'] == expected['RGB_sha256'] for r in rows if r['case_id'] == cid and r['arm'] == arm)
            times = np.array([r['seconds'] for r in items])
            matrix[index, j] = times.mean()
            group[arm] = dict(mean_seconds=float(times.mean()), seconds=times.tolist(), CV=float(times.std(ddof=1) / times.mean()))
        source.append(dict(case_id=cid, arms=group))
    rng = np.random.default_rng(2026103108)
    ix = rng.integers(0, 8, (10000, 8))
    summary = {}
    for j, arm in enumerate(arms):
        values = np.array([r['seconds'] for r in pairs if r['arm'] == arm])
        ratios = matrix[ix, 0].mean(1) / matrix[ix, j].mean(1)
        summary[arm] = dict(mean_seconds=float(values.mean()), median_seconds=float(np.median(values)),
            P90_seconds=float(np.percentile(values, 90)), speedup_vs_F=float(matrix[:, 0].mean() / matrix[:, j].mean()),
            time_saved_fraction=float(1 - matrix[:, j].mean() / matrix[:, 0].mean()),
            speedup_source_bootstrap95=np.percentile(ratios, [2.5, 97.5]).tolist(),
            peak_GiB=max(r['peak_GiB'] for r in pairs if r['arm'] == arm),
            mean_within_source_CV=float(np.mean([r['arms'][arm]['CV'] for r in source])))
    result = dict(status='COMPLETE', actual_requests=75, actual_DiT=600, warmups=3, paired_requests=72,
        summary=summary, per_source=source, R25_time_change_vs_B=summary['R25']['mean_seconds']/summary['B']['mean_seconds']-1,
        outputs_verified_against_quality=75, all_repeat_outputs_identical=True,
        source_bootstrap_note='8 fixed new-dev sources, 10000 source resamples; not a population guarantee',
        timing_scope=spec['timing_scope'], policy_updates=0, RL_updates=0)
    out = run / 'analysis';out.mkdir(exist_ok=True)
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'final_decision.json').write_text(json.dumps(dict(status='COMPLETE_MEASURED_NET_LATENCY',quality_gate_source=spec['evaluation_run'],
        passed_output_identity=True, R25_measured_faster_than_F=summary['R25']['speedup_vs_F']>1),indent=2)+'\n')
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
