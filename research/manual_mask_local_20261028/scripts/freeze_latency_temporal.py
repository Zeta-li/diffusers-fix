import collections
import datetime
import hashlib
import json
from pathlib import Path

from diagnostic_schedule import make_diagnostic_plan

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = ROOT / "data/latency_temporal_v1_preregistration.json"
    if destination.exists():
        raise RuntimeError("Never overwrite a frozen protocol")
    master = json.loads((ROOT / "data/free_head24_master_v2.json").read_text())
    reviews = {c["case_id"]: c for c in json.loads((ROOT / "data/free_full24_visual_review_v1.json").read_text())["cases"]}
    receipt_path = ROOT / "data/cpu_latency_temporal_v1.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["exit_code"] != 0:
        raise RuntimeError("CPU contract tests must pass")
    cases = []
    for original in master["cases"][:8]:
        case = {k: v for k, v in original.items() if k != "arms"}
        assert case["role"] == "fit" and case["source_origin"] == "existing_S1_seen_development"
        references = list((ROOT / "runs").glob("*/" + case["case_id"] + "/F.pt"))
        fixes = list((ROOT / "runs").glob("*/" + case["case_id"] + "/FIX32.pt"))
        assert len(references) == 1 and len(fixes) <= 1
        case["old_reference"] = str(references[0].relative_to(ROOT))
        case["old_reference_sha256"] = sha(references[0])
        if fixes:
            case["old_fix32"] = str(fixes[0].relative_to(ROOT))
            case["old_fix32_sha256"] = sha(fixes[0])
        case["evaluation_bbox"] = reviews[case["case_id"]]["influence_bbox"]
        case["historical_F_semantic_pass"] = reviews[case["case_id"]]["full_task_pass"]
        case["historical_F_review_note"] = reviews[case["case_id"]]["note"]
        case["plans"] = {name: make_diagnostic_plan(case, name) for name in ("EARLY", "MIDDLE", "LATE")}
        plans = list(case["plans"].values())
        assert len({p["total_target_token_updates"] for p in plans}) == 1
        multisets = [collections.Counter(i for step in p["selected_indices_per_step"] for i in step) for p in plans]
        assert multisets[0] == multisets[1] == multisets[2]
        assert sha(ROOT / case["image"]) == case["source_sha256"]
        cases.append(case)
    jobs = [dict(phase="warmup", case_id=cases[0]["case_id"], arm=arm, repeat=0) for arm in ("F", "FIX32")]
    for repeat in range(3):
        order = list(range(8)) if repeat != 1 else list(reversed(range(8)))
        for index in order:
            arms = ("F", "FIX32") if (index + repeat) % 2 == 0 else ("FIX32", "F")
            for arm in arms:
                jobs.append(dict(phase="latency", case_id=cases[index]["case_id"], arm=arm, repeat=repeat))
    for index, case in enumerate(cases):
        for arm in ("F", "EARLY", "MIDDLE", "LATE"):
            jobs.append(dict(phase="temporal", case_id=case["case_id"], arm=arm, repeat=0,
                             profile=index == 0 and arm in ("F", "EARLY")))
    files = ["diagnostic_schedule.py", "test_diagnostic_schedule.py", "run_latency_temporal.py",
             "freeze_latency_temporal.py", "run_s1.py", "run_free_pilot.py", "free_region.py",
             "controlled_mask.py", "local_mask_bridge.py", "legacy_local_executor.py",
             "legacy_refresh_policy.py"]
    spec = dict(phase="FROZEN_BEFORE_GENERATION", created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                purpose="user-authorized diagnostic latency and equal-budget timing experiment; no promotion of failed LocalDev16",
                source_selection="first eight cases in pre-existing free_head24_master_v2, all original S1 fit sources, no filtering on outcome",
                cases=cases, jobs=jobs, code_sha256={name: sha(ROOT / "scripts" / name) for name in files},
                CPU_test_receipt_sha256=sha(receipt_path), noise_file_sha256=sha(ROOT / "runs/s1_v2/explicit_noise.pt"),
                seed=2026102201, steps=8, dtype="bfloat16", CFG=1, target_hw=[1024,1024],
                hardware="user-confirmed H800, driver label L20Y", model_and_lora="same original Qwen Image Edit and Lightning8 as archived",
                planned_requests=82, planned_dit=656, timing_requests=48, timing_dit=384,
                warmup_requests=2, warmup_dit=16, temporal_requests=32, temporal_dit=256,
                timing_scope="resident model, PIL source already loaded; includes online plan construction, preprocess, text encoding, source VAE encoding, 8 DiT/scheduler steps, decode, CPU postprocess, bridge/cache allocation and cleanup; excludes model loading, source disk/network IO, output serialization, hashes/audits",
                timing_instrumentation="CUDA event pairs around each DiT and encode/prepare/decode; no per-step CPU sync or tensor downloads; one final device sync; no torch profiler during eligible timing",
                profile_scope="separate first temporal F/EARLY requests, counted in temporal 256 DiT, never included in net-latency samples",
                latency_analysis="per-source median of three paired repetitions, aggregate median and mean, pairwise speed ratios, individual repetitions and fluctuation; eight sources are statistical units, no quality deployment claim",
                temporal_control="identical FIX32 core, partition, endpoints and first two full states; exact equal total target updates and each-token update multiplicity; shift groups 0+1 from step index2 to4 or6",
                temporal_primary="paired terminal full-image MAE to F, fixed old influence-box outside MAE, edit correctness and extra failures, cost held equal; report all8 and historical F-success subset separately",
                temporal_opportunity_rule=dict(description="exploratory training opportunity only, not independent success or deployment gate",
                    mean_oracle_gain_over_best_constant_min=0.02, unique_constant_winner_fraction_max=0.75,
                    severe_new_failures_max=0,
                    subset_rule="all8 primary retained; repeat analysis on prespecified historical F-success subset, no post-output exclusion"),
                numeric_reference_thresholds=dict(whole_MAE_max=0.015,outside_MAE_max=0.01,
                    meaning="descriptive inherited thresholds; cannot certify semantics or overturn old decision"),
                new_training_updates=0,new_RL_updates=0,teacher_extra_dit=0,test64_opened=False,
                failure_policy="stop on first runtime, equality, budget or input mismatch; retain all completed/partial cost; no silent retries or candidate changes",
                immutable_old_decision="data/localdev16_decision_v1.json remains NO_GO")
    destination.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(protocol=str(destination),sha256=sha(destination),requests=len(jobs),dit=8*len(jobs),
                         historical_F_failures=sum(not c['historical_F_semantic_pass'] for c in cases)),ensure_ascii=False))


if __name__ == "__main__":
    main()
