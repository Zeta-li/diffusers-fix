from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
O=Path(__file__).resolve().parent;F=O/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
d=json.loads((O/'最新实验已返回汇总.json').read_text())['results']; names=['F','R25','SUP']; vals=[r['latency_s'] for r in d[:3]]
fig,ax=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
colors=['#6B7280','#167D8D','#D99426']
ax[0].bar(names,vals,color=colors);ax[0].set_ylabel('Resident request latency (s)');ax[0].set_ylim(0,5)
for i,v in enumerate(vals):ax[0].text(i,v+.08,f'{v:.3f}',ha='center')
ax[1].bar(['R25','R6','SUP'],[d[1]['rgb_mae'],d[3]['rgb_mae'],d[2]['rgb_mae']],color=[colors[1],'#B77E80',colors[2]])
ax[1].set_ylabel('RGB MAE to dense F');ax[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax[2].bar(['F','R25','R6','SUP'],[7/16]*4,color=[colors[0],colors[1],'#B77E80',colors[2]])
ax[2].set_ylim(0,1);ax[2].set_ylabel('Joint semantic pass rate')
fig.suptitle('New Dev16: reference fidelity, semantics and measured latency')
fig.savefig(F/'latest_results.png',dpi=200);fig.savefig(F/'latest_results.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(11,4.3));ax.axis('off')
# Illustrative ratios, deliberately not proportional to full 4096/4096/text token counts.
for x0,label,local in [(0,'Dense attention: all queries',False),(6,'Local attention: selected queries',True)]:
 ax.text(x0+2.3,4.35,label,ha='center',weight='bold')
 widths=[1.7,1.4,1.5]; labels=['Target: selected','Target: outside','Source + text']
 for j,(w,lab) in enumerate(zip(widths,labels)):
  x=x0+sum(widths[:j]);ax.text(x+w/2,4.02,lab,ha='center',fontsize=8)
  for i,(y,h) in enumerate([(2.8,.9),(1.7,1.1),(.5,1.2)]):
   present=(not local) or i==0
   ax.add_patch(Rectangle((x,y),w,h,facecolor=('#C6E4E7' if j==0 or not local else '#F9DCA8') if present else '#EEEEEE',edgecolor='white'))
   if present:ax.text(x+w/2,y+h/2,'fresh K,V' if not local or j==0 else 'cached K,V',ha='center',fontsize=8)
   else:ax.text(x+w/2,y+h/2,'rows skipped',ha='center',fontsize=8,color='#888888')
 ax.text(x0+2.3,.15,'QK^T / softmax / AV',ha='center')
ax.text(5.3,2.2,'→',ha='center',fontsize=24);ax.set_xlim(-.2,10.9);ax.set_ylim(-.2,4.6)
fig.savefig(F/'attention_matrix.png',dpi=200,bbox_inches='tight');fig.savefig(F/'attention_matrix.svg',bbox_inches='tight');plt.close(fig)
# Training-time chronology from actual recorded tables, not a common test-set comparison.
fig,ax=plt.subplots(figsize=(9,2.5),layout='constrained')
steps=np.arange(1,9);full={1,2,5,8}
for t in steps:
 ax.barh(0,1,left=t-.5,color='#167D8D' if t in full else '#F2C66D',edgecolor='white',height=.5)
 ax.text(t,0,'FULL' if t in full else 'LOCAL',ha='center',va='center',fontsize=9)
ax.set_xticks(steps);ax.set_xlabel('One-based denoising step');ax.set_yticks([]);ax.set_xlim(.5,8.5)
ax.set_title('R25: full refresh at 1, 2, 5, 8; student enabled only on local steps')
fig.savefig(F/'r25_schedule.png',dpi=200);fig.savefig(F/'r25_schedule.svg');plt.close(fig)
print('Created 3 figures in PNG and SVG')

for svg in F.glob("*.svg"):
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
