import argparse
import json

import torch

import run_s1 as common

ROOT=common.ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);args=p.parse_args()
    folder=ROOT/'runs'/args.attempt;spec=json.loads((folder/'preregistration.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    assert status['phase']=='complete_quality_pending'
    calls=[json.loads(x) for x in (folder/'calls.jsonl').read_text().splitlines()]
    for event in ('invoked','completed'):assert sum(c['event']==event for c in calls)==spec['planned_dit']
    assert status['successful_requests']==spec['planned_requests']
    noise=torch.load(folder/'explicit_noise.pt',map_location='cpu',weights_only=True)
    steps=0;gates=0;same_state_groups=0;controls=0
    for case in spec['cases']:
        cid=case['case_id'];records={};roots=[];branches={}
        for arm in case['arms']:
            name=arm['name'];d=torch.load(folder/cid/(name+'_raw.pt'),map_location='cpu',weights_only=True)
            m=json.loads((folder/cid/(name+'.json')).read_text());ex=m['execution'];plan=ex['base_plan']
            sigmas=torch.tensor(m['record']['sigmas']);previous=noise
            for k,state in enumerate(d['states']):
                expected=(previous.float()+(sigmas[k+1]-sigmas[k])*d['velocities'][k]).to(state.dtype)
                assert torch.equal(state,expected),(cid,name,k)
                previous=state;steps+=1
            assert m['added_target_updates']==sum(ex['selected_target_per_step'])-plan['total_target_token_updates']
            assert 0<=m['added_target_updates']<=128
            decisions=ex['decisions'];assert [v['step'] for v in decisions]==[3,5]
            for v in decisions:
                k=v['step'];candidates=v['candidates'];assert len(candidates)==64
                assert not set(candidates).intersection(plan['selected_indices_per_step'][k])
                if v['action']=='C':
                    assert torch.equal(d['velocities'][k][:,candidates],d['velocities'][k-1][:,candidates])
                    assert all(s<k for s in v['extra_velocity_stamps']);gates+=1
                elif v['action']=='F':assert all(s==k for s in v['extra_velocity_stamps'])
                if v['action']!='N':assert v['extra_kv_fresh'] and v['added_tokens']==64
                else:assert v['added_tokens']==0
            for row in ex['trace']:
                assert row['selected_target']==ex['selected_target_per_step'][row['step']-1]
                assert row['condition']==4096
                assert row['image_qkv_tokens']==row['image_mlp_tokens']==4096+row['selected_target']
            controls+=len(m['checks']['matched_archived_controls'])
            if spec['mode']=='tree_collection':
                records[name]=d;roots.append(decisions[0]);branches.setdefault(name[0],[]).append(decisions[1])
        if spec['mode']=='tree_collection':
            for name,d in records.items():
                assert all(torch.equal(a,b) for a,b in zip(d['states'][:3],records['NN']['states'][:3]))
                assert all(torch.equal(a,b) for a,b in zip(d['states'][:5],records[name[0]+'N']['states'][:5]))
                assert all(torch.equal(a,b) for a,b in zip(d['velocities'][:5],records[name[0]+'N']['velocities'][:5]))
            for group in [roots]+list(branches.values()):
                for row in group:assert row['features']==group[0]['features'] and row['state_signature']==group[0]['state_signature']
                same_state_groups+=1
    result=dict(status='PASS',attempt=args.attempt,offline_Euler_steps=steps,requests=status['successful_requests'],
        invoked_DiT=status['invoked_dit'],completed_DiT=status['completed_dit'],same_state_groups=same_state_groups,
        velocity_retention_events=gates,archived_control_matches=controls,teacher_calls=0,script_sha256=common.sha(__file__))
    common.write(ROOT/'data'/(args.attempt+'_audit.json'),result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
