import datetime
import json
from pathlib import Path

import run_s1 as common
from budget_r25_profiles import make_budget_plan

ROOT = common.ROOT


def main():
    output = ROOT / 'data/r25_fresh_dev32_v1_preregistration.json'
    assert not output.exists()
    decision_path = ROOT / 'runs/r25_seed2_v1/analysis/final_decision.json'
    decision = json.loads(decision_path.read_text())
    assert decision['pass_go_fresh_dev32'] is True
    contract_path = ROOT / 'data/r25_source_screening_contract_v1.json'
    contract = json.loads(contract_path.read_text())
    review_path = ROOT / 'data/r25_dev_source_visual_review_v1.json'
    review = json.loads(review_path.read_text())
    assert review['status'] == 'SOURCE_ONLY_REVIEW_FROZEN'
    assert review['generated_outputs_seen'] is False
    ids = contract['reserve17'] + review['selected_new15']
    assert len(ids) == len(set(ids)) == 32
    pools = [ROOT / 'inputs/budget_expansion_v1/manifest.json', ROOT / 'inputs/r25_dev_candidates_v1/manifest.json']
    source = {r['img_id']: r for p in pools for r in json.loads(p.read_text())['records']}
    old_split = json.loads((ROOT / 'data/budget_expansion_split_v1.json').read_text())
    assert not set(ids).intersection(str(c['source_group_id']) for c in old_split['cases'])
    audit = []
    hits = []
    for p in sorted((ROOT / 'runs').glob('*/preregistration.json')):
        data = json.loads(p.read_text())
        cases = data.get('cases', [])
        groups = {str(c.get('source_group_id', c.get('img_id', ''))) for c in cases if isinstance(c, dict)}
        overlap = sorted(set(ids).intersection(groups))
        audit.append(dict(path=str(p.relative_to(ROOT)), sha256=common.sha(p), overlapping_ids=overlap))
        hits.extend(overlap)
    assert not hits, hits
    cases = []
    preservation = ' Preserve all other objects and the surrounding scene. Keep changes local to the requested object or attribute.'
    for sid in ids:
        row = source[sid]
        assert row['official_split'] == 'dev' and row['geometry_eligible']
        assert common.sha(ROOT / row['image']) == row['source_sha256']
        prompt = old_split['all_prompt_overrides'].get(sid, row['instruction']).rstrip().rstrip('.') + '.' + preservation
        case = dict(case_id='r25_fresh_dev_' + sid, source_group_id=sid, role='independent_dev',
            image=row['image'], source_sha256=row['source_sha256'], source_wh=row['source_wh'], bbox=row['bbox'],
            prompt=prompt, original_instruction=row['instruction'], signature=row['signature'], official_split='dev',
            prompt_source_only_refinement=sid in old_split['all_prompt_overrides'], annotation_mask_used_only_as_bbox_hint=True)
        case['plans'] = {a: make_budget_plan(case, a) for a in ('B', 'R25')}
        cases.append(case)
    previous = json.loads((ROOT / 'data/budget_r25_v1_preregistration.json').read_text())
    names = ['freeze_r25_fresh_dev32.py', 'evaluate_r25_validation.py', 'budget_r25_profiles.py',
             'budget_profiles.py', 'train_budget_expansion.py', 'cache_aware_adapters.py',
             'trainable_budget_executor.py', 'budget_cache_executor.py', 'run_free_pilot.py',
             'run_s1.py', 'controlled_mask.py', 'free_region.py']
    spec = dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='Fresh independent official-dev source evaluation of the fixed R25 candidate, no tuning on these outputs',
        cases=cases, evaluation_arms=['F', 'B', 'R25'], visual_arms=['F', 'B', 'R25'],
        seed=2026102201, noise=previous['noise'], noise_sha256=previous['noise_sha256'],
        checkpoint=previous['checkpoint'], checkpoint_sha256=previous['checkpoint_sha256'], conditioning_mode='capture',
        planned_requests=96, planned_dit=768,
        code_sha256={name: common.sha(ROOT / 'scripts' / name) for name in names},
        seed2_decision_sha256=common.sha(decision_path), source_screening_contract_sha256=common.sha(contract_path),
        source_review_sha256=common.sha(review_path), source_pool_sha256={str(p.relative_to(ROOT)): common.sha(p) for p in pools},
        historical_generation_audit=audit, historical_generation_hits=hits,
        gate=dict(Dev_mean_MAE_reduction_vs_B_at_least=0.10, Dev_improved_sources_at_least=20,
                  Dev_new_joint_regressions_vs_B_at_most=1, Dev_new_joint_regressions_vs_F_at_most=1),
        visual_rule='Anonymous A/B/C methods, source and instruction shown; single assistant, not human consensus. Evaluate edit, semantic preservation and uncertainty; uncertain is not joint pass. No removal of dense-reference failures.',
        decision_rule='Report every source. No tuning, prompt changes or source exclusions after generation. Numeric and semantic gates must both pass for candidate validation.',
        hard_mask=False, pixel_copy=False, independent_validation=True, training_updates=0, RL_updates=0,
        test64_opened=False, target_media_read=False, test_perceptual_dedup='not performed, test media sealed',
        new_latency_claim=False)
    common.write(output, spec)
    print(json.dumps(dict(protocol=str(output), sha256=common.sha(output), sources=len(cases), requests=96, DiT=768, historical_generation_hits=hits)))


if __name__ == '__main__':
    main()
