import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--attempt', required=True)
    args = p.parse_args()
    run = ROOT / 'runs' / args.attempt
    spec = json.loads((run / 'preregistration.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    assert status['phase'] == 'complete_analysis_pending'
    assert status['completed_dit'] == status['invoked_dit'] == spec['planned_dit']
    assert status['successful_requests'] == spec['planned_requests']
    torch.set_num_threads(4)
    rows = []
    for case in spec['cases']:
        cid = case['case_id']
        folder = run / 'evaluation' / cid
        raw = {a: torch.load(folder / (a + '.pt'), map_location='cpu', weights_only=True) for a in spec['evaluation_arms']}
        for step in range(4):
            assert torch.equal(raw['R25']['states'][step], raw['B']['states'][step]), (cid, step)
        for arm in ('B', 'R25'):
            for step in range(2):
                assert torch.equal(raw[arm]['states'][step], raw['F']['states'][step])
        ref = np.asarray(Image.open(folder / 'F.png').convert('RGB'), dtype=np.float32) / 255
        h, w = ref.shape[:2]
        sw, sh = case['source_wh']
        x0, y0, x1, y1 = case['bbox']
        box = [max(0, int((x0 - 32) * w / sw)), max(0, int((y0 - 32) * h / sh)),
               min(w, int(np.ceil((x1 + 32) * w / sw))), min(h, int(np.ceil((y1 + 32) * h / sh)))]
        inside = np.zeros((h, w), dtype=bool)
        inside[box[1]:box[3], box[0]:box[2]] = True
        for arm in spec['evaluation_arms']:
            im = Image.open(folder / (arm + '.png')).convert('RGB')
            meta = json.loads((folder / (arm + '.json')).read_text())
            assert hashlib.sha256(im.tobytes()).hexdigest() == meta['RGB_sha256']
            assert len(raw[arm]['states']) == 8 and torch.equal(raw[arm]['final'], raw[arm]['states'][-1])
            assert all(torch.isfinite(z).all() for z in raw[arm]['states'])
            error = np.abs(np.asarray(im, dtype=np.float32) / 255 - ref)
            plan = None if arm == 'F' else case['plans'][arm]
            rows.append(dict(case_id=cid, role=case['role'], arm=arm, seed=spec['seed'], RGB_MAE=float(error.mean()),
                inside_MAE=float(error[inside].mean()), outside_MAE=float(error[~inside].mean()) if (~inside).any() else None,
                latent_MAE=float((raw[arm]['final'].float() - raw['F']['final'].float()).abs().mean()),
                target_update_fraction=1 if plan is None else plan['total_target_token_updates'] / 32768))
    rng = np.random.default_rng(2026103104)
    summary = {}
    roles = list(dict.fromkeys(c['role'] for c in spec['cases'])) + ['all']
    for role in roles:
        source = rows if role == 'all' else [r for r in rows if r['role'] == role]
        by = {arm: {r['case_id']: r for r in source if r['arm'] == arm} for arm in spec['evaluation_arms']}
        stats = {}
        for arm, group in by.items():
            stats[arm] = dict(n=len(group), **{k: float(np.mean([r[k] for r in group.values() if r[k] is not None]))
                for k in ('RGB_MAE', 'inside_MAE', 'outside_MAE', 'latent_MAE', 'target_update_fraction')})
        ids = list(by['B'])
        b = np.array([by['B'][c]['RGB_MAE'] for c in ids])
        a = np.array([by['R25'][c]['RGB_MAE'] for c in ids])
        ix = rng.integers(0, len(ids), (10000, len(ids)))
        stats['B_to_R25'] = dict(relative_change=float(a.mean() / b.mean() - 1),
            improved=int((a < b - 1e-10).sum()), tied=int((np.abs(a - b) < 1e-10).sum()),
            bootstrap95_relative_change=np.percentile(a[ix].mean(1) / b[ix].mean(1) - 1, [2.5, 97.5]).tolist())
        summary[role] = stats
    if 'calibration' in summary:
        cal = summary['calibration']['B_to_R25']
        gate = cal['relative_change'] <= -spec['gate']['Cal_mean_MAE_reduction_vs_B_at_least']
        gate &= cal['improved'] >= spec['gate']['Cal_improved_sources_at_least']
        gate &= summary['train']['B_to_R25']['relative_change'] < 0
    else:
        dev = summary['independent_dev']['B_to_R25']
        gate = dev['relative_change'] <= -spec['gate']['Dev_mean_MAE_reduction_vs_B_at_least']
        gate &= dev['improved'] >= spec['gate']['Dev_improved_sources_at_least']
    result = dict(status='NUMERIC_COMPLETE_VISUAL_PENDING', counts=status, records=rows, summary=summary,
        numeric_gate=bool(gate), R25_B_first4_state_checks=4 * len(spec['cases']),
        local_F_first2_state_checks=4 * len(spec['cases']), contracts_overlap=True,
        independent_validation=spec['independent_validation'], training_updates=0, RL_updates=0, new_latency_claim=False)
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    with (out / 'source_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(numeric_gate=gate, summary=summary)))


if __name__ == '__main__':
    main()
