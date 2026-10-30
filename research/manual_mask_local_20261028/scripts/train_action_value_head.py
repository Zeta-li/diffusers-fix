import argparse
import copy
import itertools
import json
import shutil
import time

import torch

import run_s1 as common
from action_value_head import ActionValueHead, ACTIONS, FEATURE_NAMES

ROOT=common.ROOT
SEQUENCES=[''.join(v) for v in itertools.product(ACTIONS,repeat=2)]


def load_trees(folder,spec):
    trees={}; rows=[]; provenance={}
    for case in spec['cases']:
        if case['role'] not in ('fit','calibration'):raise ValueError('unexpected data split')
        cid=case['case_id'];tree={a:json.loads((folder/cid/(a+'.json')).read_text()) for a in SEQUENCES}
        base=tree['NN']['quality']['mae_vs_F']
        if base<=0:raise ValueError('nonpositive reference error')
        losses={a:tree[a]['quality']['mae_vs_F'] for a in SEQUENCES}
        root=tree['NN']['execution']['decisions'][0]
        q=[100*(1-min(losses[a+b] for b in ACTIONS)/base) for a in ACTIONS]
        rows.append(dict(case_id=cid,source_group_id=case['source_group_id'],role=case['role'],step=3,prefix='',features=root['features'],q=q))
        for a in ACTIONS:
            decision=tree[a+'N']['execution']['decisions'][1]
            q=[100*(1-losses[a+b]/base) for b in ACTIONS]
            rows.append(dict(case_id=cid,source_group_id=case['source_group_id'],role=case['role'],step=5,prefix=a,features=decision['features'],q=q))
        trees[cid]=dict(case=case,records=tree,losses=losses,base=base)
        provenance[cid]={a:common.sha(folder/cid/(a+'.json')) for a in SEQUENCES}
    return trees,rows,provenance


@torch.no_grad()
def tree_rollout(model,mean,std,trees,role):
    results=[]
    for cid,item in trees.items():
        if item['case']['role']!=role:continue
        records=item['records'];sequence='';scores=[]
        for step in (3,5):
            decision=records['NN' if step==3 else sequence+'N']['execution']['decisions'][0 if step==3 else 1]
            x=torch.tensor(decision['features'],device=mean.device)
            q=model(((x-mean)/std).clamp(-5,5))
            sequence+=ACTIONS[int(q.argmax())];scores.append(q.cpu().tolist())
        loss=item['losses'][sequence]
        results.append(dict(case_id=cid,sequence=sequence,mae=loss,relative_mae=loss/item['base'],scores=scores))
    return dict(mean_relative_mae=sum(r['relative_mae'] for r in results)/len(results),
                mean_mae=sum(r['mae'] for r in results)/len(results),predictions=results)


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',required=True);p.add_argument('--attempt',required=True)
    args=p.parse_args();folder=ROOT/'runs'/args.collection;out=ROOT/'runs'/args.attempt
    if out.exists() or '/' in args.attempt:raise RuntimeError('new attempt required')
    spec=json.loads((folder/'preregistration.json').read_text());status=json.loads((folder/'status.json').read_text())
    if status['phase']!='complete_quality_pending' or status['completed_dit']!=spec['planned_dit']:raise RuntimeError('collection incomplete')
    for name,h in spec['code_sha256'].items():
        if common.sha(ROOT/'scripts'/name)!=h:raise RuntimeError('frozen implementation changed')
    cfg=spec['training'];torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    trees,rows,provenance=load_trees(folder,spec)
    fit=[r for r in rows if r['role']=='fit'];cal=[r for r in rows if r['role']=='calibration']
    fit_sources=sorted({r['source_group_id'] for r in fit});cal_sources=sorted({r['source_group_id'] for r in cal})
    if len(fit)!=48 or len(cal)!=12 or set(fit_sources)&set(cal_sources):raise RuntimeError('split mismatch')
    out.mkdir();common.write(out/'labels.json',dict(rows=rows,provenance=provenance))
    shutil.copy2(folder/'preregistration.json',out/'collection_protocol.json')
    for name in ('action_value_head.py','train_action_value_head.py'):shutil.copy2(ROOT/'scripts'/name,out/name)
    x=torch.tensor([r['features'] for r in fit],device='cuda');y=torch.tensor([r['q'] for r in fit],device='cuda')
    mean=x.mean(0);std=x.std(0).clamp_min(1e-4);z=((x-mean)/std).clamp(-5,5)
    model=ActionValueHead().cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    constants={a:sum(t['losses'][a]/t['base'] for t in trees.values() if t['case']['role']=='fit')/len(fit_sources) for a in SEQUENCES}
    constant=min(SEQUENCES,key=lambda a:(constants[a],a))
    best=None;history=[];started=time.perf_counter()
    for update in range(cfg['updates']+1):
        if update:
            model.train();loss=(model(z)-y).square().mean()
            opt.zero_grad();loss.backward();opt.step()
        if update%cfg['eval_interval']==0:
            model.eval()
            with torch.no_grad():
                train_mse=float((model(z)-y).square().mean());fit_policy=tree_rollout(model,mean,std,trees,'fit');cal_policy=tree_rollout(model,mean,std,trees,'calibration')
            row=dict(update=update,train_mse=train_mse,fit_policy=fit_policy,calibration_policy=cal_policy)
            history.append(row);common.append(out/'history.jsonl',row)
            if best is None or cal_policy['mean_relative_mae']<best['calibration_policy']['mean_relative_mae']-1e-12:
                best=copy.deepcopy(row);best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    final_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state);model.eval()
    bundle=dict(state_dict=best_state,mean=mean.cpu(),std=std.cpu(),feature_names=FEATURE_NAMES,actions=list(ACTIONS),
        selected_update=best['update'],training_updates=cfg['updates'],fit_sources=fit_sources,calibration_sources=cal_sources,
        seed=cfg['seed'],collection_protocol_sha256=common.sha(folder/'preregistration.json'))
    torch.save(bundle,out/'head.pt')
    torch.save(dict(bundle,state_dict=final_state,selected_update=cfg['updates'],
        diagnostic_only_not_calibration_selected=best['update']!=cfg['updates']),out/'final_step1000.pt')
    final_predictions={role:tree_rollout(model,mean,std,trees,role) for role in ('fit','calibration')}
    oracle={role:sum(min(t['losses'].values())/t['base'] for t in trees.values() if t['case']['role']==role)/len(fit_sources if role=='fit' else cal_sources) for role in ('fit','calibration')}
    result=dict(status='TRAINING_COMPLETE_CLOSED_LOOP_PENDING',parameter_count=sum(p.numel() for p in model.parameters()),
        training_updates=cfg['updates'],selected_update=best['update'],training_seconds=seconds,device='cuda',
        fit_sources=len(fit_sources),calibration_sources=len(cal_sources),fit_states=len(fit),calibration_states=len(cal),
        probe_sources_read=0,independent_Dev16_sources_read=0,teacher_calls=0,gpu_DiT=0,RL_updates=0,
        initial_train_mse=history[0]['train_mse'],last_train_mse=history[-1]['train_mse'],selected_train_mse=best['train_mse'],
        checkpoint_sha256=common.sha(out/'head.pt'),final_checkpoint_sha256=common.sha(out/'final_step1000.pt'),
        labels_sha256=common.sha(out/'labels.json'),
        selected_constant=constant,constant_fit_mean_relative_mae=constants,terminal_oracle_mean_relative_mae=oracle,
        selected_policy=final_predictions,selection='calibration exact recorded tree rollout, earliest tie; not independent validation',
        labels='finite horizon fitted optimal action values from all9 complete terminal outcomes, no same-state dense teacher')
    common.write(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('selected_policy','constant_fit_mean_relative_mae')},indent=2))


if __name__=='__main__':main()
