import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    result = dict(status='running', phases={})
    for phase, count, configs, protocol in [
        ('primary', 72, ['F','R2','FIX0','FIX1','FIX2','ROT'], '01_实验协议.md'),
        ('optimization', 48, ['F','R2','SHIFT','AFFINE'], '04_Value时间校正算法与执行协议.md')]:
        plan = json.loads((ROOT / 'data' / f'preregistration_{phase}.json').read_text())
        assert sha(ROOT / protocol) == plan['protocol_sha256']
        for name, expected in plan['scripts'].items():
            assert sha(ROOT / 'scripts' / name) == expected
        summary = json.loads((ROOT / 'data' / f'summary_{phase}.json').read_text())
        assert summary['status'] == 'complete' and not summary['warnings']
        assert summary['runtime_age_audits_all_passed']
        assert summary['execution_order_audit']['complete_order_matches']
        assert summary['repeated_output_sha_audit_all_passed']
        assert len(summary['baseline_F_R2_audits']) == 6
        assert all(x['png_sha256_equal'] and x['pixel_identical'] for x in summary['baseline_F_R2_audits'])
        raw = rows(ROOT / 'data' / f'runs_{phase}.jsonl')
        assert len(raw) == count and all(r['status'] == 'complete' for r in raw)
        assert len({r['run_id'] for r in raw}) == count
        assert sum(r['dit_calls'] for r in raw) == count * 8
        for row in raw:
            assert row['config'] in configs and row['dit_calls'] == 8
            assert sha(ROOT / row['output']) == row['output_sha256']
            if row['config'] in ['F', 'R2']:
                assert row['old_output_exact']
            if row['config'] == 'ROT':
                assert max(v['max_read_age'] for v in row['age_trace']) <= 2
            if phase == 'optimization':
                assert row['model_timestep_contract_exact']
        for cid in ['blocks_text','shirt_color','remove_hair']:
            for config in configs:
                group = [r for r in raw if r['case_id'] == cid and r['config'] == config]
                assert len(group) == 4 and len({r['output_sha256'] for r in group}) == 1
                assert len([r for r in group if r['kind'] == 'warmup']) == 1
                assert {r['repeat'] for r in group if r['kind'] == 'measurement'} == {0,1,2}
        result['phases'][phase] = dict(requests=count, dit_forwards=count*8, oracle_forwards=0,
            measured_requests=sum(r['kind']=='measurement' for r in raw), warmup_requests=sum(r['kind']=='warmup' for r in raw),
            all_prior_F_R2_outputs_exact=True, prior_output_controls=sum(r['config'] in ['F','R2'] for r in raw),
            peak_allocated_GiB=max(r['peak_allocated_bytes'] for r in raw)/2**30,
            request_seconds_sum=sum(r['e2e_seconds'] for r in raw), runs_sha256=sha(ROOT / 'data' / f'runs_{phase}.jsonl'))
    tiny_rotation = json.loads((ROOT / 'data/rotation_correctness.json').read_text())
    tiny_value = json.loads((ROOT / 'data/value_time_tiny_correctness.json').read_text())
    assert tiny_rotation['status'] == 'pass' and tiny_rotation['actual_tiny_model_forwards'] == 20
    assert tiny_value['status'] == 'pass' and tiny_value['actual_tiny_forward_calls']['completed'] == 172
    assert tiny_value['actual_tiny_forward_calls']['failed'] == 0
    for name, expected in tiny_value['sources'].items():
        assert sha(ROOT / 'scripts' / name) == expected
    env = json.loads((ROOT / 'data/environment_optimization.json').read_text())
    table = env['static_correction_tables']
    assert table['finite'] and table['alpha_shape'] == [60,8,8] and table['delta_shape'] == [60,8,8,24,128]
    result.update(status='pass', total_requests=120, total_real_dit_forwards=960, extra_dense_oracles_in_performance=0,
        tiny_model_forwards=192, failed_real_model_requests=0, failed_final_tiny_tests=0,
        harness_setup_failure_attempts=1, harness_setup_failed_before_tiny_model_forward=True,
        value_table_bytes=table['table_bytes'], value_table_prepare_seconds=table['prepare_seconds'],
        static_table_preparation_not_a_full_DiT=True, value_time_contract_checks=48)
    (ROOT / 'data/final_validation_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    manifest = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.name != 'artifact_manifest.json' and '__pycache__' not in p.parts}
    (ROOT / 'data/artifact_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
