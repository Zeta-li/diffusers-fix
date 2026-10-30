import json
import time

import torch

import run_s1 as common

ROOT=common.ROOT


def main():
    folder=ROOT/'runs/context_closed_v1';spec=json.loads((folder/'preregistration.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    assert status['phase']=='generation_complete_quality_pending'
    calls=[json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()]
    assert sum(v['event']=='invoked' for v in calls)==sum(v['event']=='completed' for v in calls)==192
    assert status['successful_requests']==24
    noise=torch.load(folder/'explicit_noise.pt',map_location='cpu',weights_only=True)
    steps,gates=0,0
    for case in spec['cases']:
        cid=case['case_id'];paired=[]
        old=torch.load(ROOT/'runs'/case['reference_FIX32_run']/cid/'FIX32.pt',map_location='cpu',weights_only=True)
        for arm in ['FRESH64','CONTEXT64']:
            d=torch.load(folder/cid/(arm+'_raw.pt'),map_location='cpu',weights_only=True)
            m=json.loads((folder/cid/(arm+'.json')).read_text());execution=m['execution'];plan=execution['plan']
            assert all(torch.equal(a,b) for a,b in zip(d['states'][:3],old['states'][:3]))
            assert plan['total_target_token_updates']==plan['base_plan']['total_target_token_updates']+128
            paired.append(plan['target_tokens_per_step'])
            previous=noise;sigmas=torch.tensor(m['record']['sigmas'])
            for k,state in enumerate(d['states']):
                expected=(previous.float()+(sigmas[k+1]-sigmas[k])*d['velocities'][k]).to(state.dtype)
                assert torch.equal(state,expected),(cid,arm,k)
                previous=state;steps+=1
                extra=plan['extra_indices'][k]
                assert not set(extra).intersection(plan['base_plan']['selected_indices_per_step'][k])
                if extra:
                    assert len(extra)==64
                    assert set(plan['base_plan']['core_indices']).issubset(plan['selected_indices'][k])
                    if arm=='CONTEXT64':
                        assert torch.equal(d['velocities'][k][:,extra],d['velocities'][k-1][:,extra])
                        assert all(execution['velocity_age_trace'][k]['stamps'][i]<k for i in extra)
                    gates+=1
            for row in execution['trace']:
                k=row['step']-1
                assert row['selected_target']==plan['target_tokens_per_step'][k]
                assert row['condition']==4096
                assert row['image_qkv_tokens']==row['selected_target']+4096
                assert row['image_mlp_tokens']==row['selected_target']+4096
            assert all(g['extra_kv_fresh'] for g in execution['gates'])
        assert paired[0]==paired[1]
    result=dict(status='PASS',time_unix=time.time(),successful_requests=24,invoked_dit=192,completed_dit=192,
        offline_Euler_steps=steps,paired_equal_token_budget_sources=12,gate_events=gates,
        context_velocity_retention_events=24,teacher_calls=0,training_updates=0,RL_updates=0,
        net_latency_requests=0,script_sha256=common.sha(__file__))
    assert steps==192 and gates==48
    common.write(ROOT/'data/context_closed_audit_v1.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
