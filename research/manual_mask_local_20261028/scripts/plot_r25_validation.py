import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'runs/r25_fresh_dev32_v1/analysis'
    old = json.loads((ROOT / 'runs/budget_r25_v1/analysis/results.json').read_text())
    old_sem = json.loads((ROOT / 'runs/budget_r25_v1/blind_review/unblinded_results.json').read_text())
    seed = json.loads((ROOT / 'runs/r25_seed2_v1/analysis/results.json').read_text())
    seed_sem = json.loads((ROOT / 'runs/r25_seed2_v1/blind_review/unblinded_results.json').read_text())
    dev = json.loads((out / 'results.json').read_text())
    dev_sem = json.loads((ROOT / 'runs/r25_fresh_dev32_v1/blind_review/unblinded_results.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    groups = [('Cal16 / seed 1', old, old_sem, 'calibration'), ('Cal16 / seed 2', seed, seed_sem, 'calibration'),
              ('Fresh Dev32', dev, dev_sem, 'independent_dev')]
    colors = {'F': '#6b7280', 'B': '#e69f00', 'R25': '#0072b2'}
    for i, (label, obj, sem, role) in enumerate(groups):
        for j, arm in enumerate(('B', 'R25')):
            rows = [r for r in obj['records'] if r['role'] == role and r['arm'] == arm]
            axes[0].bar(i + (j - .5) * .28, np.mean([r['RGB_MAE'] for r in rows]), .26,
                color=colors[arm], label=arm if i == 0 else None)
        for j, arm in enumerate(('F', 'B', 'R25')):
            st = sem['summary'][role][arm]
            h = st['joint'] / st['n']
            x = i + (j - 1) * .25
            axes[1].bar(x, h, .23, color=colors[arm], label=arm if i == 0 else None)
            axes[1].text(x, h + .015, f"{st['joint']}/{st['n']}", ha='center', fontsize=8)
    for ax in axes[:2]:
        ax.set_xticks(range(3), [g[0] for g in groups], rotation=12)
        ax.legend(frameon=False)
        ax.spines[['right', 'top']].set_visible(False)
    axes[0].set(title='Dense-reference RGB error (lower is better)', ylabel='Mean absolute error [0, 1]')
    axes[1].set(title='Edit + preservation success', ylabel='Single-assistant joint pass fraction', ylim=(0, 1))
    table = {a: {r['case_id']: r for r in dev['records'] if r['arm'] == a} for a in ('B', 'R25')}
    x = np.array([table['B'][cid]['RGB_MAE'] for cid in table['B']])
    y = np.array([table['R25'][cid]['RGB_MAE'] for cid in table['B']])
    axes[2].scatter(x, y, s=32, color=colors['R25'], alpha=.8)
    lim = max(x.max(), y.max()) * 1.05
    axes[2].plot([0, lim], [0, lim], '--', color='#777777', linewidth=1)
    axes[2].set(xlim=(0, lim), ylim=(0, lim), xlabel='B reference MAE', ylabel='R25 reference MAE',
                title='Fresh Dev32 / paired sources')
    axes[2].set_aspect('equal')
    fig.suptitle('Fixed LOCAL128 weights; no hard mask; 8-step Qwen-Image-Edit + Lightning', fontsize=13)
    fig.savefig(out / 'validation_summary.png', dpi=180)
    fig.savefig(out / 'validation_summary.pdf')
    print('saved validation_summary.png and .pdf')


if __name__ == '__main__':
    main()
