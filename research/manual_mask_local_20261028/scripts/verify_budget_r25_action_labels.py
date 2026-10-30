import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / 'data/budget_r25_action_labels_v1.json').read_text())
    run = ROOT / 'runs/budget_r25_v1'
    protocol = json.loads((run / 'preregistration.json').read_text())
    numeric = json.loads((run / 'analysis/results.json').read_text())
    visual = json.loads((run / 'blind_review/unblinded_results.json').read_text())
    num = {(r['case_id'], r['arm']): r for r in numeric['records']}
    vis = {(r['case_id'], r['arm']): r for r in visual['records']}
    cases = {r['case_id']: r for r in protocol['cases']}
    assert data['checkpoint_sha256'] == protocol['checkpoint_sha256']
    train = set(data['allowed_training_sources'])
    cal = set(data['calibration_not_training'])
    assert len(train) == 32 and len(cal) == 16 and not train & cal
    assert train | cal == set(cases)
    assert len(data['records']) == 48
    checked = 0
    for row in data['records']:
        cid = row['case_id']
        assert row['role'] == cases[cid]['role']
        assert (cid in train) == (row['role'] == 'train')
        assert row['base_F_joint'] == vis[cid, 'F']['joint']
        assert row['online_geometry']['bbox'] == cases[cid]['bbox']
        for arm, value in row['actions'].items():
            assert arm in ('B', 'R25', 'C')
            expected = num[cid, arm]
            assert value['reference_MAE'] == expected['RGB_MAE']
            for key in ('inside_MAE', 'outside_MAE', 'target_update_fraction'):
                assert value[key] == expected[key]
            for key in ('edit', 'preservation', 'joint', 'uncertain'):
                assert value[key] == vis[cid, arm][key]
            checked += 1
    assert checked == 144
    receipt = dict(status='PASS', source_records=48, action_records=checked,
        training_sources=32, calibration_sources=16, disjoint_roles=True,
        reference_and_semantic_labels_match=True, trained_policy=False, RL_updates=0)
    (ROOT / 'data/budget_r25_action_labels_check_v1.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
