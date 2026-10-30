import datetime
import hashlib
import json
from pathlib import Path
import random

from budget_profiles import make_budget_plan

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'data/budget_expansion_train_v1_preregistration.json'
    if path.exists():
        raise RuntimeError('Already frozen')
    selected = json.loads((ROOT / 'data/budget_expansion_split_v1.json').read_text())
    old = json.loads((ROOT / 'data/budget_student_pilot_v1_preregistration.json').read_text())
    cases = [c for c in selected['cases'] if c['role'] in ('train', 'calibration')]
    train_ids = [c['case_id'] for c in cases if c['role'] == 'train']
    cal_ids = [c['case_id'] for c in cases if c['role'] == 'calibration']
    historical = old['evaluation_sources']
    prior = {c['case_id']:c for c in old['cases']}
    for cid in historical:
        cases.append({k:v for k,v in prior[cid].items() if k != 'plan_B'})
    assert len(train_ids) == 128 and len(cal_ids) == 32
    for c in cases:
        if hashlib.sha256((ROOT/c['image']).read_bytes()).hexdigest() != c['source_sha256']:
            raise RuntimeError('Source changed')
        c['plan_B'] = make_budget_plan(c, 'B')
    rng = random.Random(2026103004); jobs = []; order = train_ids[:]
    for i in range(512):
        if i % len(order) == 0: rng.shuffle(order)
        two = i % 4 == 3; anchor = i % 8 == 0
        start = (1 if (i // 8) % 2 == 0 else 7) if anchor else rng.choice([2,3,4,5] if two else [2,3,4,5,6])
        jobs.append(dict(micro=i,case_id=order[i % len(order)],start=start,window=2 if two else 1,
                         full_regularization=anchor,seed=2026103100+i))
    names = ['train_budget_expansion.py','freeze_budget_expansion_training.py','cache_aware_adapters.py',
             'trainable_budget_executor.py','spatial_velocity_student.py','budget_profiles.py',
             'budget_cache_executor.py','run_free_pilot.py','run_s1.py','controlled_mask.py','free_region.py']
    checkpoint = 'runs/budget_student_pilot_v1/checkpoints/student_inference025.pt'
    noise = 'runs/s1_v2/explicit_noise.pt'
    spec = dict(phase='FROZEN_BEFORE_EXPANDED_TRAINING',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cases=cases,training_sources=train_ids,calibration_sources=cal_ids,historical_sources=historical,
        evaluation_sources=cal_ids+historical,evaluation_arms=['F','ALL','PILOT','EXPAND'],
        jobs=jobs,optimizer_updates=128,microbatches=512,accumulation=4,lora_rank=8,lora_lr=5e-5,
        grad_clip=1.0,weight_decay=.01,save_updates=[32,64,96,128],spatial_disabled=True,
        initialization_seed=2026103003,initial_checkpoint=checkpoint,
        initial_checkpoint_sha256=hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest(),
        noise=noise,noise_sha256=hashlib.sha256((ROOT/noise).read_bytes()).hexdigest(),
        training_noise='512 distinct frozen CUDA generator seeds; four visits per new source; eval seed absent',
        scale_sources='inherited from two historical training sources; no calibration or Dev fitting',
        kv_gradient='detached across steps; target latent differentiable within two-step window',
        teacher='frozen original Lightning8; added LoRA disabled; student state held unchanged',
        planned_prefix_dit=sum(j['start'] for j in jobs),planned_teacher_dit=sum(j['window'] for j in jobs),
        planned_gradient_student_dit=sum(j['window'] for j in jobs),planned_contract_dit=0,
        planned_evaluation_dit=(len(cal_ids)+len(historical))*4*8,
        inherited_contract='prior CPU3/77calls and H80064DiT; executor/core unchanged; unmerged merge-run F/PILOT bitwise checks',
        final_checkpoint='fixed_update128_no_selection_on_evaluation',
        Dev_gate='Cal32 mean RGBMAE vsF at least5% lower than PILOT, >=24/32 source wins, no net increase in severe task/preservation failures vsPILOT; then frozen newDev32',
        code_sha256={n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in names},
        split_sha256=hashlib.sha256((ROOT/'data/budget_expansion_split_v1.json').read_bytes()).hexdigest(),
        test64_opened=False,new_RL_updates=0,timing_claim=False)
    path.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in spec.items() if k.startswith('planned_')},indent=2))


if __name__ == '__main__':
    main()
