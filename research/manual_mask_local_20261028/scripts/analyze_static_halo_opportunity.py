"""Post-hoc numeric opportunity diagnostic; no model execution or learned policy."""
import collections
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = root / 'runs/free_head_closed_v1/analysis_fixed32/metrics.json'
rows = json.loads(p.read_text())['rows']
result = {
    'status': 'POSTHOC_EXISTING_SEEN_DATA_ONLY',
    'input': str(p.relative_to(root)),
    'input_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
    'objective': 'minimize saved RGB MAE to F subject to per-source target-token update count no greater than FIX32',
    'arms_considered': ['FIX16', 'FIX32', 'FIX48'],
    'limitations': [
        'Oracle reads terminal reference error; not an online policy or achieved learned gain.',
        'Target-token updates are not wall-clock latency or complete FLOPs.',
        'All sources previously seen; internal validation remains historical, not new independent validation.',
        'Numeric objective does not certify edit correctness or perceptual equivalence.',
        'Bound applies only to these saved fixed halos and per-source budget; not dynamic spatial/temporal policies.'
    ],
    'new_dit': 0, 'new_training_updates': 0,
    'groups': {}
}
for role in sorted({r['role'] for r in rows}) + ['all_seen_posthoc']:
    selected = [r for r in rows if r['role'] == role or role == 'all_seen_posthoc']
    by_source = collections.defaultdict(list)
    for r in selected:
        if r['arm'] in result['arms_considered']:
            by_source[r['source_group_id']].append(r)
    base, choices, arm_summary = [], [], {}
    for source, rs in by_source.items():
        assert len(rs) == 3 and {r['arm'] for r in rs} == set(result['arms_considered'])
        ref = next(r for r in rs if r['arm'] == 'FIX32')
        eligible = [r for r in rs if r['target_token_update_fraction'] <= ref['target_token_update_fraction']]
        best = min(eligible, key=lambda r: r['mae_vs_F'])
        base.append(ref)
        choices.append(best)
    for arm in result['arms_considered']:
        a = [r for r in selected if r['arm'] == arm]
        arm_summary[arm] = {k: sum(r[k] for r in a) / len(a) for k in ['mae_vs_F', 'target_token_update_fraction']}
    result['groups'][role] = {
        'n': len(base), 'fixed_arm_means': arm_summary,
        'oracle_relative_mae_reduction': 1 - sum(r['mae_vs_F'] for r in choices) / sum(r['mae_vs_F'] for r in base),
        'oracle_mean_target_update_fraction': sum(r['target_token_update_fraction'] for r in choices) / len(choices),
        'oracle_choices': [{'source_group_id': r['source_group_id'], 'arm': r['arm']} for r in choices]
    }
out = root / 'data/static_halo_opportunity_posthoc_v1.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result['groups']['all_seen_posthoc'], ensure_ascii=False, indent=2))
