import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'runs/refresh_bandit_v1'
result=json.loads((run/'analysis/results.json').read_text())
train=json.loads((ROOT/'runs/refresh_policy_v1/supervised_results.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(10.5,4.0))
names=['R25','R4','R6','SUP','RL'];vals=[result['summary']['calibration'][a]['RGB_MAE'] for a in names]
axs[0].bar(names,vals,color=['#8995a4','#769e98','#769e98','#348ac7','#bd8b4e'])
axs[0].set_ylabel('RGB MAE against full F (lower is better)')
axs[0].set_title('Cal16: actual closed-loop generation')
axs[0].set_ylim(0,max(vals)*1.2)
for i,v in enumerate(vals):axs[0].text(i,v+0.00008,f'{v:.6f}',ha='center',fontsize=9)
base=train['fixed_train_loss']['R25']
names2=['R25','Fold fixed','Geometry','Early-state MLP']
vals2=[1,train['cv']['geometry']['fold_best_fixed_mean_loss']/base,train['cv']['geometry']['mean_loss']/base,train['cv']['mlp']['mean_loss']/base]
axs[1].bar(names2,vals2,color=['#8995a4','#769e98','#348ac7','#a8bad0'])
axs[1].set_ylabel('Out-of-fold surrogate loss / R25')
axs[1].set_title('Train32: source-group cross-validation')
axs[1].set_ylim(0,1.15);axs[1].tick_params(axis='x',labelsize=9)
for i,v in enumerate(vals2):axs[1].text(i,v+0.015,f'{v:.3f}',ha='center',fontsize=9)
fig.suptitle('Timing policy pilot: 18-parameter supervised head, 128 real bandit episodes',fontsize=12)
fig.text(0.5,0.01,'Development data; semantic joint pass remains 10/16; no new net-latency measurement.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,0.06,1,0.93])
for ext in ('png','pdf'):fig.savefig(run/'analysis'/('summary.'+ext),dpi=180)
