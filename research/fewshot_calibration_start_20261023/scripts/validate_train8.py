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
    pre=json.loads((ROOT/'data/preregistration_train8.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in pre['source_files'].items())
    cases=json.loads((ROOT/'cases_train8.json').read_text())['cases']
    parent=json.loads((ROOT/'data/frozen_selection.json').read_text())
    assert [x['img_id'] for x in cases]==[x['img_id'] for x in parent['selections']['train8']]
    for split in ['val16','test64']:
        assert not set(x['img_id'] for x in cases)&set(x['img_id'] for x in parent['selections'][split])
    rows=[json.loads(s) for s in (ROOT/'data/runs_train8.jsonl').read_text().splitlines()]
    assert len(rows)==8 and all(r['status']=='complete' for r in rows)
    assert [r['case_id'] for r in rows]==[c['case_id'] for c in cases]==pre['training_ids']
    images=[]
    for case,row in zip(cases,rows):
        assert sha(ROOT/case['image'])==row['source_image_sha256']==case['source_sha256']
        assert sha(ROOT/row['output'])==row['output_sha256'] and sha(ROOT/row['moments'])==row['moments_sha256']
        assert row['formal_training_image'] and row['local_calls']==8 and row['oracle_calls']==6 and row['real_dit_forwards']==14
        assert len(row['records'])==8
        for k,record in enumerate(row['records']):
            assert record['step_0based']==k and record['student_only_euler_bitwise_equal']
            if k in range(1,7):
                assert record['student_input_sha256']==record['oracle_input_sha256']
                assert record['sampled_all_layer_cache_unchanged'] and record['velocity_age_rng_unchanged']
            else:assert not record['shadow']
        image=load_npz(ROOT/row['moments']);assert image['image_id']==case['case_id']
        assert len(image['a'])==60 and len(image['b'])==360
        for layer in range(60):
            first=image['b'][f'{layer}:1:0']
            for step in range(1,7):
                b=image['b'][f'{layer}:{step}:0'];assert b['n']==128
                assert np.array_equal(first['sum_x'],b['sum_x']) and np.array_equal(first['sum_x2'],b['sum_x2'])
        images.append(image)
    summary=json.loads((ROOT/'data/fit_train8_summary.json').read_text())
    assert summary['source_image_count']==8 and summary['new_model_queries']==summary['gradient_steps']==0
    assert sha(ROOT/summary['fitted_file'])==summary['fitted_sha256']
    fitted=load_npz(ROOT/summary['fitted_file']);assert set(fitted['source_image_ids'])==set(pre['training_ids'])
    eta=summary['selected_eta'];assert eta in [1e-4,1e-2,1.]
    expected_eta=min(summary['candidate_scores'],key=lambda x:(x['score'],x['eta']))['eta'];assert eta==expected_eta
    for candidate in fitted['loo']['candidate_scores']:
        assert len(candidate['folds'])==8
        for fold in candidate['folds']:
            assert set(fold['training_image_ids'])==set(pre['training_ids'])-{fold['heldout_image_id']}
        assert np.isclose(candidate['score'],np.mean([f['normalized_sse'] for f in candidate['folds']]))
    for layer in range(60):
        a=[image['a'][f'{layer}:0'] for image in images]
        mean=np.mean([v['sum_h']/v['n'] for v in a],axis=0)
        second=np.mean([v['sum_h2']/v['n'] for v in a],axis=0)
        np.testing.assert_allclose(fitted['route_a']['mean'][layer,0],mean,rtol=1e-10,atol=1e-9)
        np.testing.assert_allclose(fitted['route_a']['variance'][layer,0],np.maximum(0,second-mean*mean),rtol=1e-10,atol=1e-9)
        for step in range(1,7):
            b=[image['b'][f'{layer}:{step}:0'] for image in images]
            ex,ey,ex2,exy=[np.mean([v[key]/v['n'] for v in b],axis=0) for key in ['sum_x','sum_y','sum_x2','sum_xy']]
            variance=np.maximum(0,ex2-ex*ex);covariance=exy-ex*ey;ridge=eta*variance.mean()+1e-12
            alpha=(covariance+ridge)/(variance+ridge);beta=ey-alpha*ex
            np.testing.assert_allclose(fitted['route_b']['alpha'][layer,step,0],alpha,rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(fitted['route_b']['beta'][layer,step,0],beta,rtol=1e-9,atol=1e-9)
    mask=fitted['route_b']['coverage']
    assert int(mask.sum())==360 and int(fitted['route_a']['coverage'].sum())==60
    assert np.all(fitted['route_b']['alpha'][~mask]==1) and np.all(fitted['route_b']['beta'][~mask]==0)
    result={'status':'pass','validated_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Frozen Train8 acquisition, image-level moments and ridge fit integrity; not independent quality/speed validation',
        'train8_sources':8,'train8_successful_requests':8,'train8_student_forwards':64,'train8_oracle_forwards':48,
        'train8_total_real_dit_forwards':112,'failed_train8_requests':0,'new_model_forwards_for_fit':0,
        'independent_A_anchor_recomputations':60,'independent_B_pair_recomputations':360,
        'source_level_loo_splits_verified':24,'unobserved_and_diagonal_identity_verified':True,
        'val_generated':False,'test_generated':False,'rl_started':False,
        'cumulative_successful_requests':428,'cumulative_real_dit_forwards':4134,'cumulative_oracle_included':126,
        'validator_sha256':sha(Path(__file__))}
    (ROOT/'data/train8_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
