import datetime
import json

import run_s1 as common
from budget_r25_profiles import make_budget_plan

ROOT = common.ROOT


def main():
    path = ROOT / 'data/budget_r25_v1_preregistration.json'
    assert not path.exists()
    old_path = ROOT / 'data/budget_sweep_v1_preregistration.json'
    old = json.loads(old_path.read_text())
    cases = old['cases']; reused = []; equal = []
    assert len(cases) == 48
    for case in cases:
        cid = case['case_id']
        assert common.sha(ROOT / case['image']) == case['source_sha256']
        for arm in ('B','B50','C'):
            assert case['plans'][arm] == make_budget_plan(case, arm)
        plan = make_budget_plan(case, 'R25'); case['plans']['R25'] = plan
        assert plan['full_steps_zero_based'] == [0,1,4,7]
        assert plan['effective_local_fraction'] == case['plans']['B']['effective_local_fraction']
        for step in (0,1,2,3,5,6,7):
            assert plan['selected_indices_per_step'][step] == case['plans']['B']['selected_indices_per_step'][step]
        if plan['effective_local_fraction'] >= 0.5:
            assert plan['selected_indices_per_step'] == case['plans']['C']['selected_indices_per_step']
            equal.append(cid)
        cp = ROOT / old['conditioning_run'] / 'conditioning' / (cid+'.pt')
        assert common.sha(cp) == old['conditioning_sha256'][cid]
        for arm in ('F','B','B50','C'):
            stem = 'runs/budget_sweep_v1/evaluation/'+cid+'/'+arm
            reused.append(dict(case_id=cid,arm=arm,stem=stem,
                sha256={ext:common.sha(ROOT/(stem+'.'+ext)) for ext in ('png','pt','json')}))
    assert len(equal)==15 and len(reused)==192
    names = ['freeze_budget_r25.py','evaluate_budget_r25.py','budget_r25_profiles.py',
        'budget_profiles.py','train_budget_expansion.py','cache_aware_adapters.py',
        'trainable_budget_executor.py','budget_cache_executor.py','run_free_pilot.py',
        'run_s1.py','controlled_mask.py','free_region.py']
    spec = dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='Complete spatial budget x middle refresh factorial with frozen LOCAL128',
        selection='All same frozen Train32+Cal16 from prior sweep; no new source filtering or prompt changes',
        prior_protocol_sha256=common.sha(old_path),cases=cases,reused=reused,
        jobs=[dict(case_id=c['case_id'],arm='R25') for c in cases],
        planned_requests=48,planned_dit=384,evaluation_arms=['F','B','B50','C','R25'],visual_arms=['F','B','R25','C'],
        evaluation_sources=[c['case_id'] for c in cases],
        checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],
        noise=old['noise'],noise_sha256=old['noise_sha256'],conditioning_run=old['conditioning_run'],
        conditioning_sha256=old['conditioning_sha256'],code_sha256={n:common.sha(ROOT/'scripts'/n) for n in names},
        plan_contract=dict(old_B_B50_C_exact=144,R25_B_common_steps_exact=336,R25_C_identical_cases=equal),
        adapter_schedule='LoRA off at full anchors 1/2/5/8, on at local 3/4/6/7; unmerged',
        primary_comparisons=['B_to_R25_refresh_at_25','R25_to_C_spatial_at_4anchors'],
        quality='Reference RGB MAE inside/outside; anonymous edit/preservation review; report failures, no F-as-truth assumption',
        candidate_rule='Retain R25 as lower-cost fidelity candidate if Cal MAE no worse than C by 5%, no new Cal joint failures vs C, and net latency lower; mechanism development only, new independent validation required',
        visual_limits='Prior source outputs have been seen; anonymous relabeling does not make the reviewer source-naive. Report repeated-score disagreements.',
        hard_mask=False,pixel_copy=False,training_updates=0,RL_updates=0,test64_opened=False,independent_validation=False)
    common.write(path,spec)
    print(json.dumps(dict(path=str(path),sha256=common.sha(path),new_requests=48,DiT=384,reused=192,identical_plan_controls=15)))


if __name__=='__main__':main()
