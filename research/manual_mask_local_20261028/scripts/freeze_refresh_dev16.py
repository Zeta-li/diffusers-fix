import datetime
import json

import run_s1 as common
from local_edit_runtime import GeometryRefreshSelector, request_from_case, resolve_request, timing_plan, export_backend_plan

ROOT=common.ROOT


def main():
    output=ROOT/'data/refresh_dev16_v1_preregistration.json';assert not output.exists()
    review_path=ROOT/'data/r25_dev_source_visual_review_v1.json'
    review=json.loads(review_path.read_text());assert review['status']=='SOURCE_ONLY_REVIEW_FROZEN'
    unused=[r['img_id'] for r in review['rows'] if r['eligible'] and r['img_id'] not in review['selected_new15']]
    assert len(unused)==19
    ids=unused[:16]
    pool=ROOT/'inputs/r25_dev_candidates_v1/manifest.json'
    source={c['img_id']:c for c in json.loads(pool.read_text())['records']}
    qc=json.loads((ROOT/'inputs/r25_dev_candidates_v1/review/qc.json').read_text())
    assert all(c['eligible_before_visual'] for c in qc['records'] if c['img_id'] in ids)
    audit=[]
    for p in sorted((ROOT/'runs').glob('*/preregistration.json')):
        old=json.loads(p.read_text())
        groups={str(c.get('source_group_id',c.get('img_id',''))) for c in old.get('cases',[]) if isinstance(c,dict)}
        hits=sorted(set(ids)&groups);assert not hits,(str(p),hits)
        audit.append(dict(path=str(p.relative_to(ROOT)),sha256=common.sha(p),overlap=hits))
    parent=json.loads((ROOT/'data/budget_r25_v1_preregistration.json').read_text())
    head='runs/refresh_policy_v1/checkpoints/supervised.pt'
    assert common.sha(ROOT/head)=='da8a58f34513716471741f181d7fdb9ec32a4992b3eb9b829ff2e836ecd8a699'
    selector=GeometryRefreshSelector(ROOT/head)
    cases=[]
    for sid in ids:
        row=source[sid];assert row['official_split']=='dev' and common.sha(ROOT/row['image'])==row['source_sha256']
        case=dict(case_id='refresh_dev_'+sid,source_group_id=sid,role='independent_dev',official_split='dev',
            image=row['image'],source_sha256=row['source_sha256'],source_wh=row['source_wh'],bbox=row['bbox'],signature=row['signature'],
            original_instruction=row['instruction'],prompt=row['instruction'].rstrip().rstrip('.')+'. Preserve all other objects and the surrounding scene. Keep changes local to the requested object or attribute.',
            prompt_source_only_refinement=False,annotation_mask_used_only_as_bbox_hint=True)
        case['plans']={a:timing_plan(case,a) for a in ('R4','R25','R6')}
        plan,metadata=resolve_request(request_from_case(case),'SUP',selector)
        case['plans']['SUP']=plan;case['SUP_precomputed_source_only']=metadata
        assert len({p['total_target_token_updates'] for p in case['plans'].values()})==1
        cases.append(case)
    names=['local_edit_runtime.py','freeze_refresh_dev16.py','evaluate_refresh_dev16.py','analyze_refresh_dev16.py',
        'budget_r25_profiles.py','budget_profiles.py','refresh_policy.py','trainable_budget_executor.py','budget_cache_executor.py',
        'cache_aware_adapters.py','train_budget_expansion.py','run_free_pilot.py','run_s1.py','controlled_mask.py','free_region.py']
    spec=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,
        evaluation_arms=['F','R25','R6','SUP'],visual_arms=['F','R25','R6','SUP'],planned_requests=64,planned_dit=512,
        seed=2026102201,noise=parent['noise'],noise_sha256=parent['noise_sha256'],checkpoint=parent['checkpoint'],checkpoint_sha256=parent['checkpoint_sha256'],
        head=head,head_sha256=common.sha(ROOT/head),conditioning_mode='capture',
        code_sha256={n:common.sha(ROOT/'scripts'/n) for n in names},source_review_sha256=common.sha(review_path),source_pool_sha256=common.sha(pool),
        selection='first16 of19 remaining source-only-approved officialdev in original parent rank;no output based replacement',reserved3=unused[16:],
        historical_generation_audit=audit,source_dedup='previous conservative aHash/dHash QC against prior sources and all earlier candidates;no Test media read',
        candidate_gate=dict(mean_RGB_no_worse_than_R25=True,mean_surrogate_no_worse_than_R25=True,joint_new_failures_vs_R25_at_most=1,joint_count_no_lower_than_R25=True),
        choice='If multiple candidates pass, choose lower mean surrogate;report paired intervals and comparison with both fixedR25 andR6;no tuning after generation',
        latency='After numeric and visual gate, first8 frozen source order;F/R25/selected,3rotated repeats plus1warmup perarm. If no alternative passes retainR25 and benchmarkF/R25 unified interface only.',
        visual_rule='AnonymousA/B/C/D;single assistant;uncertain is not jointpass;no removal ofFfailures;freeze scores before unblinding',
        independent_validation=True,hard_mask=False,pixel_copy=False,training_updates=0,RL_updates=0,test64_opened=False,target_media_read=False)
    common.write(output,spec)
    print(json.dumps(dict(path=str(output),sha256=common.sha(output),ids=ids,requests=64,DiT=512)))


if __name__=='__main__':main()
