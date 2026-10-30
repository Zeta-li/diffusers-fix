import datetime
import hashlib
import json
from pathlib import Path
import random

from budget_profiles import make_budget_plan

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['spatial_velocity_student.py', 'cache_aware_adapters.py', 'trainable_budget_executor.py',
         'test_budget_student.py', 'freeze_budget_student.py', 'train_budget_student.py',
         'budget_profiles.py', 'budget_cache_executor.py', 'run_free_pilot.py', 'run_s1.py']


def main():
    dest = ROOT / 'data/budget_student_pilot_v1_preregistration.json'
    if dest.exists():
        raise RuntimeError('Protocol already frozen')
    old = json.loads((ROOT / 'data/action_tree_v1_preregistration.json').read_text())
    preflight = json.loads((ROOT / 'data/budget_preflight_v1_preregistration.json').read_text())
    internal = json.loads((ROOT / 'data/free_head_closed_v1_preregistration.json').read_text())
    train_ids = [c['case_id'] for c in old['cases'] if c['role'] == 'fit']
    cal_ids = [c['case_id'] for c in old['cases'] if c['role'] == 'calibration']
    diagnostic_ids = [c['case_id'] for c in preflight['cases']]
    internal_ids = [c['case_id'] for c in internal['cases'] if c['role'] == 'internal_validation']
    all_cases = {c['case_id']: c for c in internal['cases'] + preflight['cases'] + old['cases']}
    eval_ids = list(dict.fromkeys(diagnostic_ids + cal_ids + internal_ids))
    case_ids = list(dict.fromkeys(train_ids + eval_ids))
    cases = []
    for cid in case_ids:
        row = {k: all_cases[cid][k] for k in ('case_id', 'source_group_id', 'image', 'source_sha256', 'source_wh', 'bbox', 'prompt')}
        if hashlib.sha256((ROOT / row['image']).read_bytes()).hexdigest() != row['source_sha256']:
            raise ValueError('Source hash differs')
        row['plan_B'] = make_budget_plan(row, 'B')
        cases.append(row)
    assert len(train_ids) == 12 and len(cal_ids) == 3 and len(internal_ids) == 4
    assert not set(train_ids) & set(cal_ids + internal_ids)
    rng = random.Random(2026103002)
    jobs = []
    order = train_ids[:]
    for i in range(100):
        if i % len(order) == 0: rng.shuffle(order)
        two = i % 4 == 3
        anchor = i % 8 == 0
        start = (1 if (i // 8) % 2 == 0 else 7) if anchor else rng.choice([2, 3, 4, 5] if two else [2, 3, 4, 5, 6])
        jobs.append(dict(micro=i, case_id=order[i % len(order)], start=start, window=2 if two else 1,
                         full_regularization=anchor, seed=2026102201))
    proto = dict(phase='FROZEN_BEFORE_TRAINING', created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='100-microbatch real-weight training preflight before128/256-source expansion',
        training_sources=train_ids, calibration_sources=cal_ids, diagnostic_sources=diagnostic_ids,
        internal_validation_sources=internal_ids, evaluation_sources=eval_ids, cases=cases, jobs=jobs,
        training_sources_count=12, optimizer_updates=25, microbatches=100, accumulation=4,
        lora_rank=8, lora_lr=1e-4, spatial_lr=3e-4, grad_clip=1.0, weight_decay=0.01,
        spatial_width=128, spatial_blocks=4, group_weights=[1,2,1], roll_weight=.5, edge_weight=.1,
        kv_gradient='detached across steps; differentiable target latent within two-step window',
        scale_sources=diagnostic_ids[:2], scale_from='previous native untiled F velocities, training sources only',
        model='original Qwen-Image-Edit Lightning8 fused, BF16, native attention, no quantization',
        mode='all', profile='B', noise='runs/s1_v2/explicit_noise.pt',
        noise_sha256=hashlib.sha256((ROOT / 'runs/s1_v2/explicit_noise.pt').read_bytes()).hexdigest(),
        initialization_seed=2026103001, no_new_time_budget_embedding_in_pilot=True,
        evaluation_arms=['F','ALL','FRESH_B','JOINT','VELOCITY_ONLY','LORA_ONLY','FULL_STUDENT'],
        full_student_spatial_disabled=True, final_checkpoint='update25_fixed_not_selected_on_evaluation',
        planned_prefix_dit=sum(j['start'] for j in jobs),
        planned_teacher_dit=sum(j['window'] for j in jobs),
        planned_gradient_student_dit=sum(j['window'] for j in jobs),
        planned_contract_dit=64, planned_evaluation_dit=len(eval_ids)*7*8,
        timing_claim=False, new_RL_updates=0, test64_opened=False,
        code_sha256={n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in NAMES})
    dest.write_text(json.dumps(proto,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:proto[k] for k in ('training_sources_count','optimizer_updates','planned_prefix_dit','planned_teacher_dit','planned_gradient_student_dit','planned_contract_dit','planned_evaluation_dit')}))


if __name__ == '__main__': main()
