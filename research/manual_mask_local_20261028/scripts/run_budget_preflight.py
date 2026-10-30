import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch
from PIL import Image

import run_s1 as common
from run_free_pilot import load_pipeline
from run_latency_temporal import Observer
from controlled_mask import tensor_digest
from diagnostic_schedule import native_request, make_diagnostic_plan
from budget_profiles import make_budget_plan
from budget_cache_executor import budget_request

ROOT = common.ROOT


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    protocol = ROOT / args.protocol
    spec = json.loads(protocol.read_text())
    out = ROOT / "runs" / args.attempt
    if out.exists() or Path(args.attempt).name != args.attempt:
        raise RuntimeError("Fresh attempt required")
    for name, digest in spec["code_sha256"].items():
        if common.sha(ROOT / "scripts" / name) != digest:
            raise RuntimeError("Frozen code changed: " + name)
    for case in spec["cases"]:
        if common.sha(ROOT / case["image"]) != case["source_sha256"]:
            raise RuntimeError("Frozen image changed")
        for p in ("C", "B", "S", "FULL"):
            if make_budget_plan(case, p) != case["plans"][p]:
                raise RuntimeError("Frozen plan changed")
    if common.sha(ROOT / "runs/s1_v2/explicit_noise.pt") != spec["noise_sha256"]:
        raise RuntimeError("Frozen noise changed")
    out.mkdir(parents=True)
    shutil.copy2(protocol, out / "preregistration.json")
    snap = out / "code_snapshot"
    snap.mkdir()
    for name in spec["code_sha256"]:
        shutil.copy2(ROOT / "scripts" / name, snap / name)
    counts = dict(invoked_dit=0, completed_dit=0, successful_requests=0)
    observer, active = None, None
    def status(phase, **extra):
        common.write(out / "status.json", dict(phase=phase, updated_unix=time.time(), **counts, **extra))
    try:
        status("loading_model")
        pipe = load_pipeline()
        noise = torch.load(ROOT / "runs/s1_v2/explicit_noise.pt", weights_only=True, map_location="cuda")
        noise_copy = noise.clone()
        images = {c["case_id"]: Image.open(ROOT / c["image"]).convert("RGB") for c in spec["cases"]}
        cases = {c["case_id"]: c for c in spec["cases"]}
        observer = Observer(pipe, counts)
        common.write(out / "environment.json", dict(torch=torch.__version__, cuda=torch.version.cuda,
            driver_name=torch.cuda.get_device_name(), user_confirmed_gpu="H800",
            capability=list(torch.cuda.get_device_capability()), git_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            pipeline_sha256=common.sha(inspect.getfile(type(pipe))), scheduler=dict(pipe.scheduler.config),
            weights_receipt_sha256=common.sha("/workspace/model-weights/deployment_20261028.json"),
            compile=False, dtype="bfloat16", cfg=1, source_preloaded=True))
        references, seen, vae_pairs, rows = {}, {}, {}, []
        engine = None
        for job_index, job in enumerate(spec["jobs"]):
            active = dict(job_index=job_index, **job)
            status("running_" + job["phase"], active_request=active)
            if job["engine"] == "chosen" and engine is None:
                checks = []
                for cid, pair in vae_pairs.items():
                    a, b = pair["tiled"], pair["untiled"]
                    error = float(np.abs(a["rgb"].astype(np.float32) - b["rgb"].astype(np.float32)).mean() / 255)
                    checks.append(dict(case_id=cid, RGB_MAE=error, peak_GiB=b["peak"] / 2**30,
                                       tiled_vae_seconds=a["vae"], untiled_vae_seconds=b["vae"]))
                use = len(checks) == 2 and all(c["RGB_MAE"] <= 0.01 and c["peak_GiB"] <= 76 for c in checks)
                use = use and sum(c["untiled_vae_seconds"] for c in checks) < sum(c["tiled_vae_seconds"] for c in checks)
                engine = "untiled" if use else "tiled"
                common.write(out / "engine_choice.json", dict(chosen=engine, development_checks=checks,
                    rule=spec["engine_rule"], semantic_review_pending=True))
            current_engine = engine if job["engine"] == "chosen" else job["engine"]
            if current_engine == "tiled":
                pipe.vae.enable_tiling()
            else:
                pipe.vae.disable_tiling()
            case = cases[job["case_id"]]
            arm, phase = job["arm"], job["phase"]
            path = out / phase / case["case_id"]
            path.mkdir(parents=True, exist_ok=True)
            stem = arm + "_" + current_engine + "_r" + str(job["repeat"])
            trace = phase in ("contract", "vae", "warmup")
            observer.reset(trace)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            plan_started = time.perf_counter()
            if arm == "FIX32":
                plan = make_diagnostic_plan(case)
            elif arm in ("SOURCE", "ALL", "FULL"):
                plan = make_budget_plan(case, "FULL" if arm == "FULL" else "B")
            else:
                plan = None
            plan_seconds = time.perf_counter() - plan_started
            if arm in ("SOURCE", "ALL", "FULL"):
                result, bridge = budget_request(pipe, images[case["case_id"]], case["prompt"], noise, plan,
                    "source" if arm == "SOURCE" else "all", observer.callback)
            else:
                result, bridge = native_request(pipe, images[case["case_id"]], case["prompt"], noise,
                                                plan, observer.callback)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated()
            raw, events, hashes = observer.finish(noise)
            image = result.images[0]
            torch.save(raw, path / (stem + ".pt"))
            image.save(path / (stem + ".png"))
            raw_sha = tensor_digest(raw["final"])
            rgb_sha = hashlib.sha256(image.tobytes()).hexdigest()
            checks = dict(euler_steps=8 if trace else 0, finite_final=bool(torch.isfinite(raw["final"]).all()),
                          caller_noise_unchanged=torch.equal(noise, noise_copy))
            if not checks["finite_final"] or not checks["caller_noise_unchanged"]:
                raise RuntimeError("Invalid final or caller state")
            if phase == "contract":
                if arm == "F":
                    references[case["case_id"]] = raw
                else:
                    ref = references[case["case_id"]]
                    checks["all8states_bitwise_native"] = all(torch.equal(a, b) for a, b in zip(raw["states"], ref["states"]))
                    checks["all8velocities_bitwise_native"] = all(torch.equal(a, b) for a, b in zip(raw["velocities"], ref["velocities"]))
                    if not checks["all8states_bitwise_native"] or not checks["all8velocities_bitwise_native"]:
                        raise RuntimeError("Full executor not equivalent to native")
            if phase == "vae":
                vae_pairs.setdefault(case["case_id"], {})[current_engine] = dict(rgb=np.asarray(image),
                    peak=peak, vae=sum(sum(events.get(k, [])) for k in ("prepare_latents_vae_encode", "vae_decode")) / 1000)
            key = (case["case_id"], arm, current_engine)
            if key in seen:
                checks["repeat_final_bitwise"] = seen[key] == (raw_sha, rgb_sha)
                if not checks["repeat_final_bitwise"]:
                    raise RuntimeError("Repeated output differs")
            seen[key] = (raw_sha, rgb_sha)
            counts["successful_requests"] += 1
            row = dict(**job, job_index=job_index, resolved_engine=current_engine, total_wall_seconds=elapsed,
                component_cuda_ms=events, plan_seconds=plan_seconds, peak_allocated_bytes=peak,
                checks=checks, input_hashes=hashes, final_tensor_sha256=raw_sha, raw_RGB_sha256=rgb_sha,
                bridge_record=bridge, plan=plan, output_prefix=str((path / stem).relative_to(ROOT)),
                timing_eligible=phase in ("main", "repeat"), **counts)
            common.write(path / (stem + ".json"), row)
            common.append(out / "runs.jsonl", {k: v for k, v in row.items() if k not in ("plan", "bridge_record")})
            rows.append(row)
            status("running_" + phase, last_completed=active)
            print(json.dumps(dict(job=job_index + 1, total=50, phase=phase, arm=arm,
                                 case=case["case_id"], engine=current_engine, seconds=round(elapsed, 3), **counts)), flush=True)
            observer.active = None
            del raw, result
        if counts["completed_dit"] != 400 or counts["successful_requests"] != 50:
            raise RuntimeError("Budget accounting mismatch")
        status("generation_complete_analysis_pending", chosen_engine=engine, new_training_updates=0,
               new_RL_updates=0, test64_opened=False)
    except Exception as exc:
        status("failed", active_request=active, exception=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if observer is not None:
            observer.close()


if __name__ == "__main__":
    main()
