import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "runs/budget_preflight_v1/analysis/results.json").read_text())
out = ROOT / "figures/budget_preflight_v1"
out.mkdir(parents=True, exist_ok=True)
names = ["F", "FIX32", "SOURCE", "ALL"]
values = [d["aggregate"][name]["mean_seconds"] for name in names]
fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
bars = ax.barh(names, values, color=["#64748b", "#94a3b8", "#0f766e", "#2563eb"], height=0.6)
ax.invert_yaxis()
ax.set_xlim(0, 5.2)
ax.set_xlabel("Complete request latency (seconds)")
ax.set_title("H800: 8-source pilot, common untiled VAE", loc="left", fontweight="bold")
for bar, name, value in zip(bars, names, values):
    ax.text(value + .07, bar.get_y() + bar.get_height()/2,
            f'{value:.3f}s  ({d["aggregate"][name]["speedup_vs_F"]:.2f}x)', va="center")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.grid(axis="x", alpha=.2)
ax.set_axisbelow(True)
fig.text(.01, -.025, "Mean of one main timing per source/arm. SOURCE and ALL use the same B target schedule.",
         fontsize=9)
fig.savefig(out / "latency_main8.png", dpi=180, bbox_inches="tight")
plt.close(fig)
