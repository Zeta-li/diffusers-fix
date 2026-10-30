import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'data/budget_student_latency_v2_preregistration.json'
    if out.exists():raise RuntimeError('Already frozen')
    run=ROOT/'runs/budget_student_pilot_v1'
    results=json.loads((run/'analysis/results.json').read_text())
    original=json.loads((run/'preregistration.json').read_text())
    options=['JOINT','VELOCITY_ONLY','LORA_ONLY']
    calibration=results['summary']['calibration']
    candidate=min(options,key=lambda arm:(calibration[arm]['RGB_MAE'],options.index(arm)))
    cases_by_id={c['case_id']:c for c in original['cases']}
    cases=[{k:v for k,v in cases_by_id[cid].items() if k!='plan_B'} for cid in original['diagnostic_sources']]
    jobs=[]
    for arm in ('F','ALL','STUDENT','FRESH_B'):jobs.append(dict(phase='warmup',case_id=cases[0]['case_id'],arm=arm,repeat=0))
    arms=['F','ALL','STUDENT','FRESH_B']
    for repeat in range(3):
        for index,case in enumerate(cases):
            shift=(repeat+index)%4
            for arm in arms[shift:]+arms[:shift]:jobs.append(dict(phase='paired',case_id=case['case_id'],arm=arm,repeat=repeat))
    names=['benchmark_budget_student.py','freeze_budget_student_latency_v2.py','cache_aware_adapters.py','spatial_velocity_student.py',
        'trainable_budget_executor.py','budget_cache_executor.py','budget_profiles.py','run_free_pilot.py','run_s1.py']
    checkpoint='runs/budget_student_pilot_v1/checkpoints/update025.pt'
    spec=dict(phase='FROZEN_BEFORE_TIMING',supersedes_unexecuted='budget_student_latency_v1_preregistration.json',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        candidate=candidate,selection='posthoc choose min calibration3 RGB_MAE among three inference ablations; no internal4 score used',
        calibration_values={a:calibration[a]['RGB_MAE'] for a in options},
        source_scope='same8seen diagnostics; no newTest/Dev',cases=cases,jobs=jobs,planned_DiT=len(jobs)*8,planned_requests=len(jobs),
        repeats=3,checkpoint=checkpoint,checkpoint_sha256=hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest(),
        code_sha256={n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in names},
        execution='all components GPU resident; native prompt/source encode; untiledVAE; unmergedFP32LoRA;onlineplan andcachelifecycle included',
        timing_scope='synchronized full pipeline to PIL; excludes weights/inputI/O,serialization,hashchecks',
        quality_warning='timing does not establish task quality; paired outputs must equal pilot candidate; preserve all quality findings',
        RL_updates=0,test64_opened=False)
    out.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:spec[k] for k in ('candidate','calibration_values','planned_DiT','planned_requests')}))


if __name__=='__main__':main()
