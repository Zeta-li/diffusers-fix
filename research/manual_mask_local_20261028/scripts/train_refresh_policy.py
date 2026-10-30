import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from refresh_policy import ACTIONS, FEATURE_NAMES, RefreshHead, features
import run_s1 as common

ROOT = common.ROOT


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    out = ROOT / 'runs/refresh_policy_v1'
    assert not out.exists()
    rule_path = ROOT / 'data/refresh_policy_v1_preregistration.json'
    rules = json.loads(rule_path.read_text())
    for name, digest in rules['code_sha256'].items():
        assert common.sha(ROOT/'scripts'/name) == digest
    out.mkdir()
    (out/'checkpoints').mkdir()
    (out/'preregistration.json').write_text(rule_path.read_text())
    cases, rows, folder_by_id = [], [], {}
    for name in ('refresh_timing_pilot8_v1','refresh_timing_remaining40_v1'):
        run = ROOT/'runs'/name
        spec = json.loads((run/'preregistration.json').read_text())
        cases.extend(c for c in spec['cases'] if c['role']=='train')
        rows.extend(r for r in json.loads((run/'analysis/results.json').read_text())['records'] if r['role']=='train')
        folder_by_id.update({c['case_id']:run/'evaluation'/c['case_id'] for c in spec['cases'] if c['role']=='train'})
    assert len(cases)==32 and len(set(c['source_group_id'] for c in cases))==32
    noise = torch.load(ROOT/spec['noise'], weights_only=True,map_location='cpu')
    xs, losses = [], []
    for case in cases:
        cid = case['case_id']
        item = torch.load(ROOT/spec['conditioning_run']/'conditioning'/(cid+'.pt'),weights_only=True,map_location='cpu')
        states = torch.load(folder_by_id[cid]/'F.pt',weights_only=True,map_location='cpu')['states'][:2]
        xs.append(features(case,noise,states,item['hidden_states'][:,4096:],item['encoder_hidden_states'],item.get('encoder_hidden_states_mask'),item['sigmas']))
        selected = [next(r for r in rows if r['case_id']==cid and r['arm']==a) for a in ACTIONS]
        losses.append([r['RGB_MAE']+0.25*(r['outside_MAE'] or 0) for r in selected])
    x = torch.stack(xs)
    cost = torch.tensor(losses)
    y = torch.log((cost+1e-8)/(cost[:,1:2]+1e-8)).clamp(-2,2)
    order = sorted(range(32),key=lambda i:hashlib.sha256(('refresh_cv_v1|'+cases[i]['source_group_id']).encode()).hexdigest())
    folds = torch.empty(32,dtype=torch.long)
    for rank,i in enumerate(order): folds[i]=rank%4
    updates = 0
    trace = []

    def fit(ids,kind,label):
        nonlocal updates
        torch.manual_seed(2026103007)
        model = RefreshHead(kind)
        mean=x[ids].mean(0); std=x[ids].std(0).clamp_min(1e-4)
        inp=((x[ids]-mean)/std).clamp(-5,5)
        opt=torch.optim.AdamW(model.parameters(),lr=0.01,weight_decay=0.01)
        start=None
        for step in range(500):
            pred=model(inp)
            loss=F.smooth_l1_loss(pred[:,[0,2]],y[ids][:,[0,2]],beta=0.2)
            if start is None:start=float(loss.detach())
            opt.zero_grad();loss.backward();opt.step();updates+=1
        end=float(F.smooth_l1_loss(model(inp)[:,[0,2]],y[ids][:,[0,2]],beta=0.2).detach())
        trace.append(dict(label=label,kind=kind,updates=500,initial_loss=start,final_loss=end))
        return model,mean,std

    cv={}
    for kind in ('geometry','mlp'):
        predictions=torch.empty_like(y)
        fixed_choice=torch.empty(32,dtype=torch.long)
        for fold in range(4):
            tr=torch.where(folds!=fold)[0]; va=torch.where(folds==fold)[0]
            model,mean,std=fit(tr,kind,kind+'_fold'+str(fold))
            with torch.no_grad():predictions[va]=model(((x[va]-mean)/std).clamp(-5,5))
            fixed_choice[va]=cost[tr].mean(0).argmin()
        choices=predictions.argmin(1)
        vals=cost[torch.arange(32),choices]
        cv[kind]=dict(mean_loss=float(vals.mean()),relative_vs_R25=float(vals.mean()/cost[:,1].mean()-1),
            choices=[ACTIONS[i] for i in choices.tolist()],predicted_log_relative=predictions.tolist(),
            fold_best_fixed_mean_loss=float(cost[torch.arange(32),fixed_choice].mean()),
            fold_best_fixed_choices=[ACTIONS[i] for i in fixed_choice.tolist()])
    chosen=min(cv,key=lambda k:cv[k]['mean_loss'])
    start=time.perf_counter()
    model,mean,std=fit(torch.arange(32),chosen,'final')
    payload=dict(kind=chosen,state_dict=model.state_dict(),mean=mean,std=std,feature_names=FEATURE_NAMES,actions=list(ACTIONS),
        seed=2026103007,fit_sources=[c['case_id'] for c in cases],training_updates=500)
    torch.save(payload,out/'checkpoints/supervised.pt')
    torch.save(dict(x=x,y=y,cost=cost,folds=folds,case_ids=[c['case_id'] for c in cases]),out/'train_data.pt')
    result=dict(status='supervised_complete_calibration_not_read',parameters=sum(p.numel() for p in model.parameters()),
        fit_sources=32,features=len(FEATURE_NAMES),chosen_kind=chosen,cv=cv,trace=trace,total_optimizer_updates=updates,
        final_optimizer_updates=500,final_fit_seconds=time.perf_counter()-start,
        fixed_train_loss={a:float(cost[:,i].mean()) for i,a in enumerate(ACTIONS)},
        best_train_fixed=ACTIONS[int(cost.mean(0).argmin())],head_is_promoted=False,semantic_risk_trained=False,
        risk_note='Outside-reference MAE is only a preservation surrogate, not semantic failure probability.',
        checkpoint_sha256=common.sha(out/'checkpoints/supervised.pt'),RL_updates=0,DiT=0)
    common.write(out/'supervised_results.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
