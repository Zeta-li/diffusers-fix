import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = ROOT / 'runs' / args.attempt
    spec = json.loads((run / 'preregistration.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    assert status['completed_dit'] == status['invoked_dit'] == spec['planned_dit']
    rows = []
    checks = 0
    for case in spec['cases']:
        cid = case['case_id']
        folder = run / 'evaluation' / cid
        raw = {a: torch.load(folder / (a + '.pt'), weights_only=True, map_location='cpu') for a in spec['evaluation_arms']}
        for arm in ('R4', 'R6'):
            for step in range(2):
                assert torch.equal(raw[arm]['states'][step], raw['F']['states'][step])
                checks += 1
            for step in range(3 if arm == 'R4' else 4):
                assert torch.equal(raw[arm]['states'][step], raw['R25']['states'][step])
                checks += 1
        ref = np.asarray(Image.open(folder / 'F.png').convert('RGB'), dtype=np.float32) / 255
        h, w = ref.shape[:2]
        sw, sh = case['source_wh']
        x0, y0, x1, y1 = case['bbox']
        inside = np.zeros((h, w), bool)
        inside[max(0,int((y0-32)*h/sh)):min(h,int(np.ceil((y1+32)*h/sh))), max(0,int((x0-32)*w/sw)):min(w,int(np.ceil((x1+32)*w/sw)))] = True
        for arm in spec['evaluation_arms']:
            im = Image.open(folder / (arm + '.png')).convert('RGB')
            meta = json.loads((folder / (arm + '.json')).read_text())
            assert hashlib.sha256(im.tobytes()).hexdigest() == meta['RGB_sha256']
            assert len(raw[arm]['states']) == 8 and torch.equal(raw[arm]['final'], raw[arm]['states'][-1])
            assert all(torch.isfinite(t).all() for t in raw[arm]['states'])
            err = np.abs(np.asarray(im, dtype=np.float32) / 255 - ref)
            rows.append(dict(case_id=cid, role=case['role'], arm=arm, RGB_MAE=float(err.mean()),
                inside_MAE=float(err[inside].mean()), outside_MAE=float(err[~inside].mean()) if (~inside).any() else None,
                latent_MAE=float((raw[arm]['final'].float()-raw['F']['final'].float()).abs().mean()),
                target_update_fraction=1 if arm == 'F' else case['plans'][arm]['total_target_token_updates']/32768))
    summary = {}
    for role in ('train', 'calibration', 'all'):
        ids = [c['case_id'] for c in spec['cases'] if role == 'all' or c['role'] == role]
        if not ids:
            continue
        data = {a: np.array([next(r['RGB_MAE'] for r in rows if r['case_id'] == cid and r['arm'] == a) for cid in ids]) for a in ('B','R4','R25','R6')}
        means = {a: float(v.mean()) for a,v in data.items()}
        fixed = min(('R4','R25','R6'), key=lambda a: means[a])
        mat = np.stack([data[a] for a in ('R4','R25','R6')], 1)
        oracle = mat.min(1)
        summary[role] = dict(n=len(ids), mean_MAE=means, best_fixed=fixed,
            relative_best_vs_R25=means[fixed]/means['R25']-1,
            oracle_MAE=float(oracle.mean()), oracle_relative_vs_best=float(oracle.mean()/means[fixed]-1),
            oracle_winners=dict(collections.Counter([('R4','R25','R6')[i] for i in mat.argmin(1)])),
            improved_vs_R25={a:int((v < data['R25']-1e-10).sum()) for a,v in data.items()},
            oracle_is_deployable=False)
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    result = dict(counts=status, summary=summary, records=rows, shared_prefix_checks=checks,
        semantic_review='pending', net_latency_claim=False, training_updates=0, RL_updates=0)
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    with (out / 'source_metrics.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}, indent=2))


if __name__ == '__main__':
    main()
