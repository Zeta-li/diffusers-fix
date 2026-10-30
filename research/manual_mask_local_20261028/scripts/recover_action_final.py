import json
import shutil
import time

import torch

import run_s1 as common
from action_value_head import ActionValueHead, FEATURE_NAMES, ACTIONS
from train_action_value_head import load_trees, tree_rollout

ROOT=common.ROOT


def main():
    original=ROOT/'runs/action_fit_v1';collection=ROOT/'runs/action_tree_v1';out=ROOT/'runs/action_weight_recovery_v1'
    if out.exists():raise RuntimeError('new recovery attempt required')
    spec=json.loads((collection/'preregistration.json').read_text());cfg=spec['training']
    freeze=json.loads((ROOT/'data/action_head_training_code_freeze_v1.json').read_text())
    for name,h in freeze['code_sha256'].items():assert common.sha(ROOT/'scripts'/name)==h
    history={r['update']:r for r in map(json.loads,(original/'history.jsonl').read_text().splitlines())}
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    trees,rows,_=load_trees(collection,spec);fit=[r for r in rows if r['role']=='fit']
    x=torch.tensor([r['features'] for r in fit],device='cuda');y=torch.tensor([r['q'] for r in fit],device='cuda')
    mean=x.mean(0);std=x.std(0).clamp_min(1e-4);z=((x-mean)/std).clamp(-5,5)
    model=ActionValueHead().cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    out.mkdir();shutil.copy2(__file__,out/'recover_action_final.py');checks=[];started=time.perf_counter()
    for update in range(cfg['updates']+1):
        if update:
            model.train();loss=(model(z)-y).square().mean();opt.zero_grad();loss.backward();opt.step()
        if update%cfg['eval_interval']==0:
            model.eval()
            with torch.no_grad():value=float((model(z)-y).square().mean())
            assert value==history[update]['train_mse'],(update,value,history[update]['train_mse'])
            for role,key in [('fit','fit_policy'),('calibration','calibration_policy')]:
                actual=tree_rollout(model,mean,std,trees,role)
                assert actual==history[update][key],(update,role)
            checks.append(update)
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},mean=mean.cpu(),std=std.cpu(),
        feature_names=FEATURE_NAMES,actions=list(ACTIONS),selected_update=1000,training_updates=1000,
        archival_replay_updates=1000,diagnostic_only_not_calibration_selected=True),out/'final_step1000.pt')
    result=dict(status='PASS',reason='Original fitter archived only calibration-selected checkpoint; final-step parameters recovered by exact deterministic replay.',
        effective_training_steps=1000,additional_optimizer_updates=1000,new_DiT=0,changed_hyperparameters=False,changed_checkpoint_selection=False,
        exact_history_and_predictions_checks=checks,seconds=seconds,checkpoint_sha256=common.sha(out/'final_step1000.pt'))
    common.write(out/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
