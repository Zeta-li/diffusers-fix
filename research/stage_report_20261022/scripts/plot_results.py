from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
rows = json.loads((ROOT/'data/evidence_snapshot.json').read_text())['quality_timing_rows']
methods = [('rotation','R2'),('rotation','ROT'),('value','SHIFT'),('value','AFFINE'),('channel','CHANNEL')]
labels = ['R2 (P1)','ROT (P1)','SHIFT (P2)','AFFINE (P2)','CHANNEL (P3)']
colors = ['#94a3b8','#0f766e','#2563eb','#7c3aed','#c2410c']
cases = [('blocks_text','Text replacement'),('shirt_color','Shirt recoloring'),('remove_hair','Hair removal')]
fig, axes = plt.subplots(2,3,figsize=(16,8.4),layout='constrained')
for col,(cid,title) in enumerate(cases):
 subset = [next(r for r in rows if r['experiment']==p and r['config']==m and r['case_id']==cid) for p,m in methods]
 values = [r['lpips_mapmean_to_F'] for r in subset]
 bars = axes[0,col].bar(np.arange(5),values,color=colors,width=.7)
 axes[0,col].axhline(.02,color='#dc2626',linestyle='--',linewidth=1.3,label='Pilot threshold = 0.02')
 axes[0,col].set_title(title,fontweight='bold',pad=12)
 axes[0,col].set_ylim(0,.058)
 axes[0,col].set_ylabel('LPIPS to dense 8-step reference F')
 axes[0,col].set_xticks(np.arange(5),labels,rotation=25,ha='right')
 axes[0,col].bar_label(bars,labels=[f'{v:.4f}' for v in values],fontsize=9,padding=3)
 reductions = [100*(1-r['paired_geometric_latency_ratio_to_F']) for r in subset]
 bars = axes[1,col].bar(np.arange(5),reductions,color=colors,width=.7)
 axes[1,col].axhline(10,color='#dc2626',linestyle='--',linewidth=1.3)
 axes[1,col].set_ylim(0,17)
 axes[1,col].set_ylabel('Paired net latency reduction to F (%)')
 axes[1,col].set_xticks(np.arange(5),labels,rotation=25,ha='right')
 axes[1,col].bar_label(bars,labels=[f'{v:.2f}%' for v in reductions],fontsize=9,padding=3)
 for ax in axes[:,col]:
  ax.spines[['top','right']].set_visible(False)
  ax.grid(axis='y',alpha=.18)
  ax.set_axisbelow(True)
axes[0,0].legend(loc='upper right',fontsize=8,frameon=False)
fig.suptitle('Local editing acceleration: measured quality and latency',fontsize=17,fontweight='bold')
fig.supxlabel('P1: rotation phase; P2: scalar Value phase; P3: channel Value phase. Each ratio uses its own contemporaneous F.\nThree development examples; three timing repeats per configuration; one fixed-seed output per quality evaluation.',fontsize=10)
fig.savefig(ROOT/'assets/quality_latency_overview.png',dpi=180,bbox_inches='tight')
fig.savefig(ROOT/'assets/quality_latency_overview.svg',bbox_inches='tight')
print('saved quality_latency_overview.png/.svg')
