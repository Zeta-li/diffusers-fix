import json
import math
import time

import torch

import run_s1 as common
from controlled_mask import tensor_digest
from refresh_supervision import candidate_regions

ROOT = common.ROOT


def main():
    runs = ['object_selectivity_v1', 'object_selectivity_fixed32_v1',
            'object_selectivity_seed2_v1', 'refresh_supervision_v1', 'refresh_controls_v1']
    total, requests, euler, same_states, actions = 0, 0, 0, 0, 0
    for run in runs:
        folder = ROOT/'runs'/run
        spec = json.loads((folder/'preregistration.json').read_text())
        status = json.loads((folder/'status.json').read_text())
        calls = [json.loads(v) for v in (folder/'calls.jsonl').read_text().splitlines()]
        ni = sum(c['event']=='invoked' for c in calls)
        nc = sum(c['event']=='completed' for c in calls)
        assert ni == nc == spec['planned_dit'] == status['completed_dit']
        assert status['phase'] in ('generation_complete_quality_pending', 'collection_complete_analysis_pending')
        total += nc; requests += status['successful_requests']
        noise = torch.load(folder/'explicit_noise.pt', map_location='cpu', weights_only=True)
        for case in spec['cases']:
            cid = case['case_id']; dst=folder/cid
            assert common.sha(ROOT/case['image']) == case['source_sha256']
            if run.startswith('refresh_'):
                arms = ['FIX32']
                record = json.loads((ROOT/'runs/free_fixed32_v1'/cid/'FIX32.json').read_text())['record']
            else:
                arms = [a['name'] for a in case['arms']]
            for arm in arms:
                p = dst/(arm+'_raw.pt' if run.startswith('refresh_') else arm+'_raw_before_checks.pt')
                data = torch.load(p, map_location='cpu', weights_only=True)
                if not run.startswith('refresh_'):
                    record = json.loads((dst/(arm+'.json')).read_text())['record']
                previous=noise
                sigma=torch.tensor(record['sigmas'],dtype=torch.float32)
                for k,state in enumerate(data['states']):
                    expected=(previous.float()+(sigma[k+1]-sigma[k])*data['velocities'][k]).to(state.dtype)
                    assert torch.equal(expected,state), (run,cid,arm,k)
                    previous=state;euler+=1
                if run.startswith('refresh_'):
                    old=torch.load(ROOT/'runs/free_fixed32_v1'/cid/'FIX32.pt',map_location='cpu',weights_only=True)
                    assert all(torch.equal(a,b) for a,b in zip(data['states'],old['states']))
                    plan=json.loads((dst/'FIX32_plan.json').read_text())
                    provenance=[]
                    for k in spec['teacher_steps']:
                        d=torch.load(dst/f'step{k}.pt',map_location='cpu',weights_only=True)
                        assert torch.equal(d['current'], data['states'][k-1])
                        assert torch.equal(d['condition'], old['condition'])
                        assert torch.equal(d['student_velocity'],data['velocities'][k])
                        err=(d['student_velocity'].float()-d['teacher_velocity'].float()).square().mean(-1)[0]
                        torch.testing.assert_close(err,d['error_map'],rtol=2e-6,atol=1e-10)
                        dt=sigma[k+1]-sigma[k]
                        a=(d['current'].float()+dt*d['student_velocity']).to(torch.bfloat16)
                        b=(d['current'].float()+dt*d['teacher_velocity']).to(torch.bfloat16)
                        ue=(a.float()-b.float()).square().mean(-1)[0]
                        torch.testing.assert_close(ue,d['update_error_map'],rtol=2e-6,atol=1e-10)
                        assert math.isclose(float(ue.mean()),d['update_mse'],rel_tol=2e-6,abs_tol=1e-10)
                        if d['actions']:
                            expected_regions=({'REPLAY':[], 'FULL':sorted(set(range(4096))-set(d['selected']))}
                                if spec.get('candidate_mode')=='controls' else
                                candidate_regions(d['error_map'],d['selected'],plan['core_indices'],[64,64],64,spec['seed']+k))
                            for action in d['actions']:
                                assert action['extra']==expected_regions[action['action']]
                                assert action['additional_tokens']==len(expected_regions[action['action']])
                                assert not set(action['extra']).intersection(d['selected'])
                                assert math.isclose(float(action['update_gain_map'].mean()),action['update_gain'],rel_tol=1e-4,abs_tol=1e-9)
                                if action['action']=='REPLAY':
                                    assert action['update_gain']==0 and torch.count_nonzero(action['update_gain_map'])==0
                                if action['action']=='FULL':
                                    assert action['velocity_mse']==0 and action['update_mse']==0
                                actions+=1
                        provenance.append(dict(step=k, sigma_current=float(sigma[k]),sigma_next=float(sigma[k+1]),
                            current_sha256=tensor_digest(d['current']),reference_condition_sha256=record['reference_condition_sha256'],
                            text_sha256=record['text_sha256'], teacher_input_source='exact native dispatch fields of this student step',
                            state_matches_archived_FIX32=True))
                        same_states+=1
                    common.write(dst/'label_provenance_audit.json',dict(derived_after_collection=True,states=provenance))
    assert (total,requests,euler,same_states,actions)==(260,23,184,28,48)
    result=dict(status='PASS',time_unix=time.time(),invoked_dit=total,completed_dit=total,
        successful_requests=requests,offline_Euler_steps=euler,same_state_labels_verified=same_states,
        counterfactuals_verified=actions,equal_budget_counterfactuals_verified=32,control_counterfactuals_verified=16,
        teacher_dit=28,counterfactual_dit=48,
        prior_FIX32_student_eight_state_equivalence_requests=8,new_training_updates=0,new_RL_updates=0,
        new_net_latency_requests=0,script_sha256=common.sha(__file__),
        CPU_GPU_float_reduction_tolerance=dict(rtol=2e-6,atol=1e-10),
        first_audit_failure='CPU/GPU FP32 reductions were incorrectly required bitwise; failure log retained; no new DiT')
    common.write(ROOT/'data/fix32_supervision_audit_v2.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
