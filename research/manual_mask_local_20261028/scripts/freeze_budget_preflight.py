import datetime
import hashlib
import json
from pathlib import Path

from budget_profiles import make_budget_plan
from diagnostic_schedule import make_diagnostic_plan


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = ROOT / "data/budget_preflight_v1_preregistration.json"
    if destination.exists():
        raise RuntimeError("Frozen protocol already exists")
    old = json.loads((ROOT / "data/latency_temporal_v1_preregistration.json").read_text())
    cases = []
    for item in old["cases"]:
        case = {k: v for k, v in item.items() if k != "plans"}
        case["plans"] = {p: make_budget_plan(case, p) for p in ("C", "B", "S", "FULL")}
        case["plans"]["FIX32"] = make_diagnostic_plan(case)
        if sha(ROOT / case["image"]) != case["source_sha256"]:
            raise RuntimeError("Source changed")
        cases.append(case)
    jobs = []
    def add(phase, case, arm, repeat=0, engine="chosen"):
        jobs.append(dict(phase=phase, case_id=case["case_id"], arm=arm,
                         repeat=repeat, engine=engine))
    for case in cases[:2]:
        for arm in ("F", "FULL"):
            add("contract", case, arm, engine="tiled")
    for case in cases[:2]:
        add("vae", case, "F", engine="tiled")
        add("vae", case, "F", engine="untiled")
    for index, arm in enumerate(("F", "FIX32", "SOURCE", "ALL", "F", "ALL")):
        add("warmup", cases[0], arm, repeat=index)
    arms = ["F", "FIX32", "SOURCE", "ALL"]
    for i, case in enumerate(cases):
        order = arms[i % 4:] + arms[:i % 4]
        for arm in order:
            add("main", case, arm)
    for case in reversed(cases[:2]):
        for arm in ("ALL", "SOURCE"):
            add("repeat", case, arm, repeat=1)
    assert len(jobs) == 50
    names = ["budget_profiles.py", "budget_cache_executor.py", "test_budget_cache_executor.py",
             "freeze_budget_preflight.py", "run_budget_preflight.py", "diagnostic_schedule.py",
             "run_latency_temporal.py", "run_free_pilot.py", "run_s1.py", "free_region.py",
             "local_mask_bridge.py", "legacy_local_executor.py", "legacy_refresh_policy.py",
             "controlled_mask.py"]
    receipt = ROOT / "data/cpu_budget_cache_v1.json"
    if json.loads(receipt.read_text())["exit_code"] != 0:
        raise RuntimeError("CPU contracts not passed")
    spec = dict(phase="FROZEN_BEFORE_GENERATION",
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), cases=cases, jobs=jobs,
        code_sha256={name: sha(ROOT / "scripts" / name) for name in names},
        CPU_receipt_sha256=sha(receipt), noise_sha256=sha(ROOT / "runs/s1_v2/explicit_noise.pt"),
        planned_requests=50, planned_dit=400,
        source_scope="same first8 seen fit sources as latency_temporal_v1, none removed",
        model="original Qwen-Image-Edit plus correct Lightning8, fused, CFG1 BF16 native attention",
        steps=8, seed=2026102201, target_hw=[1024, 1024],
        timing_scope="resident model; preloaded PIL source/noise; includes plan, prompt, source VAE, 8 DiT, decode, PIL postprocess and cache lifecycle",
        first_stage="rule budget only, no learned adapters or velocity corrections",
        source_arm="B target plan; condition stream cached at nonanchors, text fresh",
        all_arm="same B target plan; condition and text cached at nonanchors",
        engine_rule=dict(use_untiled_if="both VAE-check outputs finite, each RGB MAE versus tiled <=0.01, peak <=76GiB, and summed encode+decode CUDA time lower",
                         selection_sources=2, selection_is_development_not_test=True,
                         same_engine_for_all_main_arms=True),
        timing_note="main8 one sample per arm, first2 source/all one extra paired sample; preliminary not formal3repeat benchmark",
        quality_rule="report all outputs and incremental semantic failures; no inherited pixel-MAE stop gate",
        fatal_rule="stop on input/contract/nonfinite/runtime failure; save raw outputs and exact partial counts; no silent retries",
        planned_trace_steps=112, new_training_updates=0, new_RL_updates=0,
        teacher_extra_calls=0, test64_opened=False)
    destination.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(protocol=str(destination), sha256=sha(destination), planned_dit=400)))


if __name__ == "__main__":
    main()
