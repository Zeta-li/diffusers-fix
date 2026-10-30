import json
import torch

import run_s1 as common

ROOT = common.ROOT


def main():
    folder=ROOT/'runs/localdev16_v1'
    spec=json.loads((folder/'preregistration.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    assert status['phase']=='generation_complete_quality_pending'
    assert status['successful_requests']==64 and status['invoked_dit']==status['completed_dit']==512
    calls=[json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()]
    assert sum(r['event']=='invoked' for r in calls)==sum(r['event']=='completed' for r in calls)==512
    for name,digest in spec['code_sha256'].items():
        assert common.sha(folder/'code_snapshot'/name)==digest
    noise=torch.load(folder/'explicit_noise.pt',map_location='cpu',weights_only=True)
    steps=retentions=0
    for case in spec['cases']:
        assert common.sha(ROOT/case['image'])==case['source_sha256']
        d=folder/case['case_id'];raws={};metas={}
        for arm in ('F','FIX32','FRESH64','CONTEXT64'):
            raw=torch.load(d/(arm+'_raw.pt'),map_location='cpu',weights_only=True)
            meta=json.loads((d/(arm+'.json')).read_text())
            raws[arm]=raw;metas[arm]=meta
            sigmas=torch.tensor(meta['record']['sigmas']);previous=noise
            for k,state in enumerate(raw['states']):
                expected=(previous.float()+(sigmas[k+1]-sigmas[k])*raw['velocities'][k]).to(state.dtype)
                assert torch.equal(expected,state),(case['case_id'],arm,k)
                previous=state;steps+=1
            assert torch.equal(raw['final'],raw['states'][-1])
            assert meta['record']['dit_calls']==8
            if arm!='F':
                for key in ('initial_noise_sha256','reference_condition_sha256','text_sha256','text_mask_sha256'):
                    assert meta['record'][key]==metas['F']['record'][key]
                assert all(torch.equal(a,b) for a,b in zip(raw['states'][:2],raws['F']['states'][:2]))
            if arm in ('FRESH64','CONTEXT64'):
                ex=meta['execution'];plan=ex['plan']
                assert all(torch.equal(a,b) for a,b in zip(raw['states'][:3],raws['FIX32']['states'][:3]))
                assert sum(meta['target_tokens_per_step'])==sum(metas['FIX32']['target_tokens_per_step'])+128
                for row in ex['trace']:
                    k=row['step']-1
                    assert row['selected_target']==meta['target_tokens_per_step'][k]
                    assert row['condition']==4096
                    assert row['image_qkv_tokens']==row['image_mlp_tokens']==row['selected_target']+4096
                for k,extra in enumerate(plan['extra_indices']):
                    if not extra:continue
                    assert len(extra)==64 and not set(extra).intersection(plan['base_plan']['selected_indices_per_step'][k])
                    if arm=='CONTEXT64':
                        assert torch.equal(raw['velocities'][k][:,extra],raw['velocities'][k-1][:,extra])
                        retentions+=1
                assert all(g['extra_kv_fresh'] for g in ex['gates'])
        assert metas['FRESH64']['target_tokens_per_step']==metas['CONTEXT64']['target_tokens_per_step']
    assert steps==512 and retentions==32
    result=dict(status='PASS',requests=64,invoked_dit=512,completed_dit=512,
                offline_Euler_steps=steps,retained_velocity_events=retentions,
                same_budget_pairs=16,training_updates=0,teacher_dit=0,RL_updates=0,net_latency_requests=0)
    common.write(ROOT/'data/localdev16_audit_v1.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
