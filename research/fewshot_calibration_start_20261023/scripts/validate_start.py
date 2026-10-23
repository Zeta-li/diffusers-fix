import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from moments import load_npz

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prereg=json.loads((ROOT/'data/preregistration_engineering.json').read_text())
    assert all(sha(ROOT/p)==value for p,value in prereg['source_files'].items())
    collector=json.loads((ROOT/'data/collector_tiny_checks.json').read_text())
    fitted=json.loads((ROOT/'data/fitted_tiny_correctness.json').read_text())
    assert collector['status']=='pass' and fitted['status']=='pass'
    raw=[json.loads(line) for line in (ROOT/'data/runs_engineering.jsonl').read_text().splitlines()]
    assert len(raw)==1 and raw[0]['status']=='complete'
    row=raw[0]
    assert row['local_calls']==8 and row['oracle_calls']==6 and row['real_dit_forwards']==14
    assert row['formal_training_image'] is False and row['old_R2_PNG_exact'] is True
    assert sha(ROOT/row['output'])==row['output_sha256']=='615b3db8b859ae49c27bb5c0460533fe624a84adcaec6a9eba9e898a7cb94d56'
    assert sha(ROOT/row['moments'])==row['moments_sha256']
    assert len(row['records'])==8
    for k,step in enumerate(row['records']):
        assert step['step_0based']==k and step['student_only_euler_bitwise_equal'] is True
        if k in range(1,7):
            assert step['shadow'] is True and step['student_input_sha256']==step['oracle_input_sha256']
            assert step['sampled_all_layer_cache_unchanged'] is True and step['velocity_age_rng_unchanged'] is True
        else:assert step['shadow'] is False
    samples=json.loads((ROOT/'data/samples_engineering_blocks_text.json').read_text())
    data=load_npz(ROOT/row['moments'])
    assert len(data['a'])==len(samples)==60 and len(data['b'])==360
    for layer in range(60):
        assert len(samples[str(layer)])==len(set(samples[str(layer)]))==128
        a=data['a'][f'{layer}:0'];assert a['n']==128
        assert len(a['sum_h'])==3072 and np.isfinite(a['sum_h']).all() and np.isfinite(a['sum_h2']).all()
        assert np.all(a['sum_h2']/a['n']-(a['sum_h']/a['n'])**2 >= -1e-9)
        first=data['b'][f'{layer}:1:0']
        for step in range(1,7):
            b=data['b'][f'{layer}:{step}:0'];assert b['n']==128
            for key in ['sum_x','sum_y','sum_x2','sum_y2','sum_xy']:
                assert b[key].shape==(3072,) and np.isfinite(b[key]).all()
            assert np.array_equal(first['sum_x'],b['sum_x']) and np.array_equal(first['sum_x2'],b['sum_x2'])
    result={'status':'pass','validated_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Engineering collection integrity and CPU contracts; not Train8 fit or quality acceptance',
        'engineering_requests':1,'engineering_student_forwards':8,'engineering_dense_shadow_forwards':6,'engineering_real_dit_forwards':14,
        'new_tiny_forwards':collector['tiny_model_forwards']+sum(fitted['tiny_model_forward_counts'].values()),
        'formal_training_images':0,'real_fitted_training_models':0,'rl_trajectories':0,
        'old_R2_PNG_exact':True,'sampled_layers':60,'observed_anchor_records':60,'observed_value_pairs':360,
        'raw_old_value_moments_exactly_constant_across_six_steps':True,
        'diagnostic_seconds':row['diagnostic_seconds'],'peak_allocated_GiB':row['peak_allocated_bytes']/2**30,
        'cumulative_through_engineering_requests':420,'cumulative_through_engineering_real_dit_forwards':4022,
        'new_training_calls_not_inferred_from_plans':True,'validator_sha256':sha(Path(__file__))}
    (ROOT/'data/engineering_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
