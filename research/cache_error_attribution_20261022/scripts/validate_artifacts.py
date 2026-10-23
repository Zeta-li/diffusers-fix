import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    plan = json.loads((ROOT / 'data/preregistration.json').read_text())
    environment = json.loads((ROOT / 'data/environment.json').read_text())
    previous = Path(environment['previous_root'])
    assert sha(ROOT / '01_实验协议.md') == plan['protocol_sha256']
    for name, expected in plan['source_hashes'].items():
        assert sha(ROOT / 'scripts' / name) == expected
    assert sha(previous / 'masks/regions.json') == plan['prior_regions_sha256']
    assert sha(previous / 'data/runs.jsonl') == plan['prior_runs_sha256']
    run = rows(ROOT / 'data/runs.jsonl')
    steps = rows(ROOT / 'data/step_metrics.jsonl')
    assert len(run) == 12 and all(r['status'] == 'complete' for r in run)
    assert {(r['case_id'], r['mode']) for r in run} == {(cid, m) for cid in plan['case_ids'] for m in plan['modes']}
    assert len(steps) == 72
    assert {(r['case_id'], r['mode'], r['step']) for r in steps} == {(cid, m, s) for cid in plan['case_ids'] for m in ['B', 'V', 'K'] for s in range(1, 9)}
    assert sum(r['local_calls'] for r in run) == 72
    assert sum(r['oracle_calls'] for r in run) == 72
    assert sum(r['native_calls'] for r in run) == 24
    assert all(r['actual_euler_update_bitwise_equal'] for r in steps)
    full = [r for r in steps if r['full_step']]
    assert len(full) == 18 and all(r['combined_error']['squared_error_sum'] == 0 for r in full)
    assert all(r['factorization_max_abs'] == r['orthogonality_cross_term'] == 0 for r in steps)
    assert all(r['oracle_sampled_cache_unchanged'] for r in steps)
    old_exact = [r for r in run if r['mode'] in ['D', 'B']]
    assert len(old_exact) == 6 and all(r['old_output_exact'] for r in old_exact)
    for row in run:
        assert sha(ROOT / row['output']) == row['output_sha256']
        assert row['sigmas'] == run[0]['sigmas'] and row['timesteps'] == run[0]['timesteps']
    analysis = json.loads((ROOT / 'data/summary.json').read_text())
    assert analysis['status'] == 'complete' and not analysis['warnings']
    derived = {(r['case_id'], r['mode'], r['step']): r for r in analysis['same_state_step_errors']}
    for row in steps:
        current = derived[(row['case_id'], row['mode'], row['step'])]
        for region in ['active', 'inactive']:
            assert current[f'{region}_count'] == row[f'{region}_count']
            assert current[f'{region}_scalar_count'] == row[f'{region}_error']['count']
    tiny = [json.loads((ROOT / 'data/attribution_tiny_correctness.json').read_text()),
            json.loads((ROOT / 'attempts/tiny_v1/attribution_tiny_correctness.json').read_text())]
    assert all(t['status'] == 'pass' and t['passed_tests'] == 8 and t['completed_actual_tiny_model_forwards'] == 96 for t in tiny)
    for name, expected in tiny[0]['source_hashes'].items():
        assert sha(ROOT / 'scripts' / name) == expected
    value = dict(status='pass', requests=12, total_real_model_forwards=168, local_forwards=72, oracle_forwards=72,
        native_forwards=24, same_state_records=72, full_step_exact_controls=18, actual_euler_bitwise_checks=72,
        prior_exact_output_controls=6, tiny_test_rounds=2, tiny_test_cases=16, tiny_model_forwards=192,
        failed_real_model_requests=0, failed_tiny_tests=0, peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in run),
        diagnostic_request_seconds_sum=sum(r['e2e_seconds'] for r in run),
        local_seconds_sum=sum(r['local_seconds_sum'] for r in run), oracle_seconds_sum=sum(r['oracle_seconds_sum'] for r in run),
        deployment_timing_measured=False, preregistration_sha256=sha(ROOT / 'data/preregistration.json'),
        runs_sha256=sha(ROOT / 'data/runs.jsonl'), step_metrics_sha256=sha(ROOT / 'data/step_metrics.jsonl'))
    (ROOT / 'data/final_validation_summary.json').write_text(json.dumps(value, ensure_ascii=False, indent=2))
    manifest = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(ROOT.rglob('*'))
                if p.is_file() and p.name != 'artifact_manifest.json' and '__pycache__' not in p.parts}
    (ROOT / 'data/artifact_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
