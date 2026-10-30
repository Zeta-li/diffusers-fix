import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

import run_s1 as common
from run_free_pilot import load_pipeline
from train_budget_expansion import euler
from cache_aware_adapters import CacheAdapters
from trainable_budget_executor import TrainableBudgetExecutor
from refresh_policy import ACTIONS, features, load_head, normalize, switch_plan

ROOT = common.ROOT


def main():
    torch.set_num_threads(4)
    run = ROOT/'runs/refresh_bandit_v1'
    assert not run.exists()
    rule = ROOT/'data/refresh_policy_v1_preregistration.json'
    rules = json.loads(rule.read_text())
    for name,digest in rules['code_sha256'].items():
        assert common.sha(ROOT/'scripts'/name)==digest
    run.mkdir(); (run/'episodes').mkdir(); (run/'calibration').mkdir(); (run/'checkpoints').mkdir()
    (run/'preregistration.json').write_text(rule.read_text())
    common.write(run/'code_snapshot_hashes.json',{n:common.sha(ROOT/'scripts'/n) for n in ['run_refresh_bandit.py','refresh_policy.py']})
    cases=[]; folders={}
    for name in ('refresh_timing_pilot8_v1','refresh_timing_remaining40_v1'):
        old=ROOT/'runs'/name
        spec=json.loads((old/'preregistration.json').read_text())
        cases.extend(spec['cases'])
        folders.update({c['case_id']:old/'evaluation'/c['case_id'] for c in spec['cases']})
    training=[c for c in cases if c['role']=='train']; calibration=[c for c in cases if c['role']=='calibration']
    assert len(training)==32 and len(calibration)==16
    initial, data=load_head(ROOT/'runs/refresh_policy_v1/checkpoints/supervised.pt')
    initial.eval(); initial.requires_grad_(False)
    policy=copy.deepcopy(initial); policy.requires_grad_(True)
    opt=torch.optim.Adam(policy.parameters(),lr=0.0005)
    rng=np.random.default_rng(2026103011)
    generator=torch.Generator(device='cpu').manual_seed(2026103011)
    counts=dict(invoked_dit=0,completed_dit=0,successful_requests=0,RL_episodes=0,RL_updates=0,calibration_requests=0)
    active={}; samples=[]; history=[]
    def status(phase,**kw): common.write(run/'status.json',dict(phase=phase,active=active,**counts,**kw))
    try:
        status('loading_model')
        pipe=load_pipeline(); pipe.vae.disable_tiling(); pipe.text_encoder.to('cpu')
        gc.collect(); torch.cuda.empty_cache()
        model=pipe.transformer; model.requires_grad_(False)
        adapters=CacheAdapters(model)
        adapters.load_state_dict(torch.load(ROOT/spec['checkpoint'],weights_only=True,map_location='cpu')['adapters'])
        for p in adapters.parameters():p.requires_grad_(False)
        noise_cpu=torch.load(ROOT/spec['noise'],weights_only=True,map_location='cpu')
        noise=noise_cpu.to('cuda')
        condition_cache={}
        for c in cases:
            cid=c['case_id'];cp=ROOT/spec['conditioning_run']/'conditioning'/(cid+'.pt')
            item=torch.load(cp,weights_only=True,map_location='cpu')
            assert torch.equal(item['hidden_states'][:,:4096],noise_cpu)
            condition_cache[cid]=item

        def generate(case,head,stochastic,folder):
            cid=case['case_id']; item=condition_cache[cid]
            fields={k:v.to('cuda') if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in item.items()
                    if k not in ('hidden_states','sigmas','timesteps','timestep')}
            cond=item['hidden_states'][:,4096:].to('cuda'); sigmas=item['sigmas'].to('cuda'); times=item['timesteps'].to('cuda')
            exe=TrainableBudgetExecutor(model,case['plans']['R25'])
            z=noise.clone(); states=[]; selected=None
            with torch.no_grad():
                for step in range(8):
                    assert exe.step==step
                    if step==2:
                        feat=features(case,noise_cpu,states[:2],item['hidden_states'][:,4096:],item['encoder_hidden_states'],item.get('encoder_hidden_states_mask'),item['sigmas'])
                        norm=normalize(feat,data)
                        scores=head(norm)
                        probs=torch.softmax(-scores/0.25,dim=-1)
                        chosen=int(torch.multinomial(probs,1,generator=generator)) if stochastic else int(scores.argmin())
                        selected=ACTIONS[chosen]
                        switch_plan(exe,case,selected)
                    counts['invoked_dit']+=1
                    common.append(run/'calls.jsonl',dict(**active,step=step,arm=selected,invoked=counts['invoked_dit']))
                    with adapters.mode(step not in exe.full_steps):
                        velocity=exe.forward(hidden_states=torch.cat([z,cond],1),timestep=times[step].reshape(1).to(z.dtype)/1000,**fields)[0][:,:4096]
                    assert torch.isfinite(velocity).all()
                    counts['completed_dit']+=1
                    z=euler(z,velocity,sigmas,step);states.append(z.cpu().clone())
                exe.clear(); image=common.decode(pipe,z)
            actual_hash=hashlib.sha256(image.tobytes()).hexdigest()
            expected=json.loads((folders[cid]/(selected+'.json')).read_text())
            expected_raw=torch.load(folders[cid]/(selected+'.pt'),weights_only=True,map_location='cpu')
            assert actual_hash==expected['RGB_sha256']
            assert all(torch.equal(a,b) for a,b in zip(states,expected_raw['states']))
            ref=np.asarray(Image.open(folders[cid]/'F.png').convert('RGB'),np.float32)/255
            err=np.abs(np.asarray(image,np.float32)/255-ref)
            h,w=err.shape[:2];sw,sh=case['source_wh'];x0,y0,x1,y1=case['bbox']
            inside=np.zeros((h,w),bool)
            inside[max(0,int((y0-32)*h/sh)):min(h,int(np.ceil((y1+32)*h/sh))),max(0,int((x0-32)*w/sw)):min(w,int(np.ceil((x1+32)*w/sw)))]=True
            outer=float(err[~inside].mean()) if (~inside).any() else 0.
            loss=float(err.mean())+0.25*outer
            base=np.abs(np.asarray(Image.open(folders[cid]/'R25.png').convert('RGB'),np.float32)/255-ref)
            base_loss=float(base.mean())+0.25*(float(base[~inside].mean()) if (~inside).any() else 0.)
            reward=float(np.clip(-np.log((loss+1e-8)/(base_loss+1e-8)),-2,2))
            image.save(folder.with_suffix('.png'))
            torch.save(dict(final=z.cpu(),states=states,features=feat),folder.with_suffix('.pt'))
            counts['successful_requests']+=1
            row=dict(**active,action=selected,predicted_scores=scores.tolist(),probabilities=probs.tolist(),
                RGB_MAE=float(err.mean()),outside_MAE=outer,surrogate_loss=loss,R25_surrogate_loss=base_loss,reward=reward,
                RGB_sha256=actual_hash,matched_frozen_action=True,all8states_match=True,features=feat.tolist(),net_latency_claim=False)
            common.write(folder.with_suffix('.json'),row)
            return norm,chosen,reward,row

        for round_id in range(4):
            for idx in rng.permutation(32):
                case=training[int(idx)]
                episode=counts['RL_episodes']
                active=dict(phase='bandit',episode=episode,round=round_id,case_id=case['case_id'])
                status('bandit_training')
                norm,chosen,reward,row=generate(case,policy,True,run/'episodes'/f'{episode:03d}')
                counts['RL_episodes']+=1
                samples.append((norm,chosen,reward));history.append(row)
                common.append(run/'episodes.jsonl',row)
                if len(samples)==8:
                    inputs=torch.stack([s[0] for s in samples]);actions=torch.tensor([s[1] for s in samples]);rewards=torch.tensor([s[2] for s in samples])
                    logits=-policy(inputs)/0.25
                    dist=torch.distributions.Categorical(logits=logits)
                    with torch.no_grad():old=torch.distributions.Categorical(logits=-initial(inputs)/0.25)
                    kl=torch.distributions.kl_divergence(dist,old).mean()
                    loss=-(dist.log_prob(actions)*rewards).mean()+0.05*kl-0.02*dist.entropy().mean()
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);opt.step()
                    counts['RL_updates']+=1
                    common.append(run/'optimizer.jsonl',dict(update=counts['RL_updates'],loss=float(loss.detach()),mean_reward=float(rewards.mean()),KL=float(kl.detach())))
                    samples=[]
                print(json.dumps(dict(episode=episode,action=row['action'],reward=reward,**counts)),flush=True)
        assert not samples and counts['RL_episodes']==128 and counts['RL_updates']==16
        data['state_dict']=policy.state_dict();data['RL_episodes']=128;data['RL_updates']=16
        torch.save(data,run/'checkpoints/bandit.pt')
        common.write(run/'training_complete.json',dict(**counts,checkpoint_sha256=common.sha(run/'checkpoints/bandit.pt'),calibration_started=False))
        for case in calibration:
            for name,head in [('SUP',initial),('RL',policy)]:
                active=dict(phase='calibration',case_id=case['case_id'],policy=name)
                status('calibration_frozen_heads')
                _,_,_,row=generate(case,head,False,run/'calibration'/(case['case_id']+'_'+name))
                counts['calibration_requests']+=1
                common.append(run/'calibration.jsonl',row)
        assert counts['invoked_dit']==counts['completed_dit']==1280
        assert counts['successful_requests']==160
        status('complete_analysis_pending',supervised_checkpoint_sha256=common.sha(ROOT/'runs/refresh_policy_v1/checkpoints/supervised.pt'),
            RL_checkpoint_sha256=common.sha(run/'checkpoints/bandit.pt'),calibration_tuned=False,test64_opened=False)
    except BaseException as exc:
        status('failed',exception=repr(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
