import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'runs/budget_r25_latency_v1/analysis'
    data = json.loads((folder / 'results.json').read_text())
    arms = ['F', 'B', 'R25', 'C']
    colors = ['#607d8b', '#4c78a8', '#2e8b57', '#b279a2']
    times = [data['summary'][a]['mean_seconds'] for a in arms]
    errors = [data['same8_quality'][a]['RGB_MAE'] for a in arms]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    ax[0].bar(arms, times, color=colors)
    for i, (a, t) in enumerate(zip(arms, times)):
        ax[0].text(i, t + .06, f'{t:.3f}s\n{data["summary"][a]["speedup_vs_F"]:.3f}x', ha='center')
    ax[0].set(ylabel='Complete request seconds', ylim=(0, max(times) * 1.22), title='H800: 8 Cal sources, 3 paired repeats')
    ax[1].scatter(times, errors, color=colors, s=65)
    offsets = {'F': (8, 8), 'B': (8, 8), 'R25': (-28, -23), 'C': (12, 12)}
    for a, x, y in zip(arms, times, errors):
        ax[1].annotate(a, (x, y), xytext=offsets[a], textcoords='offset points')
    ax[1].set(xlabel='Complete request seconds', ylabel='RGB MAE versus F on the same 8 sources',
              title='Reference fidelity and latency (development data)')
    ax[1].margins(x=.18, y=.2)
    fig.savefig(folder / 'latency_and_tradeoff_readable.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
