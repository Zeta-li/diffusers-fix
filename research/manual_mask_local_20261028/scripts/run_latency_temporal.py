import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import torch
from PIL import Image

import run_s1 as common
from run_free_pilot import load_pipeline
from diagnostic_schedule import make_diagnostic_plan, native_request
from controlled_mask import tensor_digest

ROOT = common.ROOT


class Observer:
    def __init__(self, pipe, counts):
        self.pipe, self.counts = pipe, counts
        self.active = None
        self.originals = {}
        self.handles = [pipe.transformer.register_forward_pre_hook(self.before, with_kwargs=True),
                        pipe.transformer.register_forward_hook(self.after)]
        for owner, method, label in [(pipe, "encode_prompt", "prompt_encode"),
                                     (pipe, "prepare_latents", "prepare_latents_vae_encode"),
                                     (pipe.vae, "decode", "vae_decode")]:
            original = getattr(owner, method)
            self.originals[(owner, method)] = original
            def wrapped(*args, _fn=original, _label=label, **kwargs):
                start = self.event()
                value = _fn(*args, **kwargs)
                end = self.event()
                self.active["events"].setdefault(_label, []).append((start, end))
                return value
            setattr(owner, method, wrapped)

    @staticmethod
    def event():
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def reset(self, trace):
        self.active = dict(trace=trace, calls=0, events={}, velocities=[], states=[], input0=None,
                           condition=None, text=None, text_mask=None, final=None)

    def before(self, module, args, fields):
        a = self.active
        if args or a["calls"] >= 8 or fields["hidden_states"].shape != (1, 8192, 64):
            raise RuntimeError("Unexpected real DiT input")
        a["dit_start"] = self.event()
        if a["trace"] and a["calls"] == 0:
            a["input0"] = fields["hidden_states"][:, :4096].detach().clone()
            a["condition"] = fields["hidden_states"][:, 4096:].detach().clone()
            a["text"] = fields["encoder_hidden_states"].detach().clone()
            mask = fields.get("encoder_hidden_states_mask")
            a["text_mask"] = None if mask is None else mask.detach().clone()
        a["calls"] += 1
        self.counts["invoked_dit"] += 1

    def after(self, module, args, result):
        a = self.active
        end = self.event()
        a["events"].setdefault("dit", []).append((a["dit_start"], end))
        self.counts["completed_dit"] += 1
        if a["trace"]:
            a["velocities"].append(result[0][:, :4096].detach().clone())

    def callback(self, pipe, index, timestep, values):
        a = self.active
        if a["calls"] != index + 1:
            raise RuntimeError("DiT/scheduler callback order mismatch")
        if a["trace"]:
            a["states"].append(values["latents"].detach().clone())
        if index == 7:
            a["final"] = values["latents"].detach()
        return {}

    def finish(self, noise):
        a = self.active
        if a["calls"] != 8 or a["final"] is None:
            raise RuntimeError("Incomplete request")
        events = {name: [start.elapsed_time(end) for start, end in pairs] for name, pairs in a["events"].items()}
        data = dict(final=a["final"].cpu(), states=[], velocities=[])
        hashes = {}
        if a["trace"]:
            if not torch.equal(a["input0"], noise):
                raise RuntimeError("Initial noise differs from frozen input")
            previous = noise
            for i, (state, velocity) in enumerate(zip(a["states"], a["velocities"])):
                dt = self.pipe.scheduler.sigmas[i + 1] - self.pipe.scheduler.sigmas[i]
                expected = (previous.float() + dt * velocity).to(state.dtype)
                if not torch.equal(expected, state) or not bool(torch.isfinite(state).all()):
                    raise RuntimeError("Unmodified all-target Euler contract failed")
                previous = state
            if len(a["states"]) != 8:
                raise RuntimeError("Missing saved steps")
            data.update(states=[v.cpu() for v in a["states"]], velocities=[v.cpu() for v in a["velocities"]],
                        input0=a["input0"].cpu(), condition=a["condition"].cpu(),
                        sigmas=self.pipe.scheduler.sigmas.cpu())
            for name in ("input0", "condition", "text", "text_mask"):
                hashes[name] = None if a[name] is None else tensor_digest(a[name])
        return data, events, hashes

    def close(self):
        for handle in self.handles:
            handle.remove()
        for (owner, method), original in self.originals.items():
            setattr(owner, method, original)


def profile_summary(profiler, output):
    keys = []
    for event in profiler.key_averages():
        keys.append(dict(name=event.key, count=event.count,
                         self_cpu_us=event.self_cpu_time_total, cpu_total_us=event.cpu_time_total,
                         self_device_us=event.self_device_time_total, device_total_us=event.device_time_total))
    common.write(output, sorted(keys, key=lambda row: row["self_device_us"], reverse=True))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    protocol = ROOT / args.protocol
    spec = json.loads(protocol.read_text())
    out = ROOT / "runs" / args.attempt
    if out.exists() or Path(args.attempt).name != args.attempt or spec["phase"] != "FROZEN_BEFORE_GENERATION":
        raise RuntimeError("Fresh attempt and frozen protocol required")
    for filename, digest in spec["code_sha256"].items():
        if common.sha(ROOT / "scripts" / filename) != digest:
            raise RuntimeError("Code changed after freeze: " + filename)
    for case in spec["cases"]:
        if common.sha(ROOT / case["image"]) != case["source_sha256"]:
            raise RuntimeError("Source changed")
        for name in ("EARLY", "MIDDLE", "LATE"):
            if make_diagnostic_plan(case, name) != case["plans"][name]:
                raise RuntimeError("Plan changed after freeze")
    if common.sha(ROOT / "runs/s1_v2/explicit_noise.pt") != spec["noise_file_sha256"]:
        raise RuntimeError("Initial noise file changed")
    out.mkdir(parents=True)
    shutil.copy2(protocol, out / "preregistration.json")
    (out / "code_snapshot").mkdir()
    for filename in spec["code_sha256"]:
        shutil.copy2(ROOT / "scripts" / filename, out / "code_snapshot" / filename)
    counts = dict(invoked_dit=0, completed_dit=0, successful_requests=0,
                  warmup_requests=0, latency_requests=0, temporal_requests=0)
    def status(phase, **extra):
        common.write(out / "status.json", dict(phase=phase, updated_unix=time.time(), **counts, **extra))
    observer = None
    active_request = None
    try:
        status("loading_model")
        start_load = time.perf_counter()
        pipe = load_pipeline()
        torch.cuda.synchronize()
        model_load = time.perf_counter() - start_load
        noise = torch.load(ROOT / "runs/s1_v2/explicit_noise.pt", weights_only=True, map_location="cuda")
        noise_original = noise.clone()
        images = {c["case_id"]: Image.open(ROOT / c["image"]).convert("RGB") for c in spec["cases"]}
        observer = Observer(pipe, counts)
        common.write(out / "environment.json", dict(torch=torch.__version__, cuda=torch.version.cuda,
            driver_name=torch.cuda.get_device_name(), user_confirmed_gpu="H800",
            capability=list(torch.cuda.get_device_capability()), model_load_seconds=model_load,
            git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            native_pipeline_sha256=common.sha(inspect.getfile(type(pipe))), scheduler=dict(pipe.scheduler.config),
            torch_threads=torch.get_num_threads(), native_attention=True, dtype="bfloat16", cfg=1,
            full_condition_count=4096, source_preloaded=True, output="PIL", hardware_clocks_not_locked=True))
        cases = {c["case_id"]: c for c in spec["cases"]}
        first_results = {}
        temporal_prefix = {}
        temporal_inputs = {}
        latency_checks = []
        jobs = spec["jobs"]
        assert len(jobs) == 82 and sum(j["phase"] == "latency" for j in jobs) == 48
        for job_index, job in enumerate(jobs):
            case = cases[job["case_id"]]
            arm, phase = job["arm"], job["phase"]
            active_request = dict(job_index=job_index, **job)
            status("running_" + phase, active_request=active_request)
            request_dir = out / phase / case["case_id"]
            request_dir.mkdir(parents=True, exist_ok=True)
            stem = arm + "_r" + str(job["repeat"])
            trace = phase != "latency"
            observer.reset(trace)
            torch.cuda.reset_peak_memory_stats()
            profiling = job.get("profile", False)
            ctx = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False,
                with_stack=False) if profiling else nullcontext(None)
            with ctx as profiler:
                torch.cuda.synchronize()
                start = time.perf_counter()
                plan_start = time.perf_counter()
                plan = None if arm == "F" else make_diagnostic_plan(case, "EARLY" if arm == "FIX32" else arm)
                plan_seconds = time.perf_counter() - plan_start
                result, bridge = native_request(pipe, images[case["case_id"]], case["prompt"], noise,
                                               plan, observer.callback)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated()
            raw, events, hashes = observer.finish(noise)
            if not torch.equal(noise, noise_original):
                raise RuntimeError("Caller noise changed")
            image = result.images[0]
            image_sha = hashlib.sha256(image.tobytes()).hexdigest()
            raw_sha = tensor_digest(raw["final"])
            checks = dict(euler_checked_steps=8 if trace else 0, noise_unchanged=True)
            key = (case["case_id"], "FIX32" if arm == "EARLY" else arm)
            if key in first_results:
                checks["repeat_final_bitwise"] = raw_sha == first_results[key][0]
                checks["repeat_RGB_bitwise"] = image_sha == first_results[key][1]
                if not checks["repeat_final_bitwise"] or not checks["repeat_RGB_bitwise"]:
                    raise RuntimeError("Repeated frozen inference changed output")
            else:
                first_results[key] = (raw_sha, image_sha)
            old_path = case.get("old_reference") if arm == "F" else case.get("old_fix32") if arm in ("FIX32", "EARLY") else None
            if old_path:
                old = torch.load(ROOT / old_path, weights_only=True, map_location="cpu")
                checks["archived_final_bitwise"] = torch.equal(raw["final"], old["final"])
                if not checks["archived_final_bitwise"]:
                    raise RuntimeError("Lean path differs from archived final state")
            if phase == "temporal":
                if arm == "F":
                    temporal_prefix[case["case_id"]] = raw["states"][:2]
                    temporal_inputs[case["case_id"]] = hashes
                else:
                    checks["first_two_states_bitwise_F"] = all(torch.equal(x, y) for x, y in zip(raw["states"][:2], temporal_prefix[case["case_id"]]))
                    checks["same_input_hashes"] = hashes == temporal_inputs[case["case_id"]]
                    if not checks["first_two_states_bitwise_F"] or not checks["same_input_hashes"]:
                        raise RuntimeError("Temporal paired inputs or anchor states changed")
            torch.save(raw, request_dir / (stem + ".pt"))
            image.save(request_dir / (stem + ".png"))
            if profiling:
                profile_summary(profiler, request_dir / (stem + "_operator_profile.json"))
            counts["successful_requests"] += 1
            counts[phase + "_requests"] += 1
            row = dict(**job, job_index=job_index, total_wall_seconds=elapsed,
                       plan_seconds=plan_seconds, component_cuda_ms=events,
                       peak_allocated_bytes=peak, checks=checks,
                       hashes=hashes, final_tensor_sha256=raw_sha, raw_RGB_sha256=image_sha,
                       bridge_record=bridge, plan=plan,
                       timing_eligible=(phase == "latency"), quality_approved=False, **counts)
            common.write(request_dir / (stem + ".json"), row)
            common.append(out / "runs.jsonl", {k: v for k, v in row.items() if k not in ("bridge_record", "plan")})
            latency_checks.append(checks)
            status("running_" + phase, last_completed=active_request)
            print(json.dumps(dict(job=job_index + 1, total=len(jobs), phase=phase, case=case["case_id"],
                                  arm=arm, seconds=round(elapsed, 3), **counts)), flush=True)
            observer.active = None
            del result, raw
        if counts["completed_dit"] != 656 or counts["successful_requests"] != 82:
            raise RuntimeError("Final budget mismatch")
        status("generation_complete_analysis_pending", test64_opened=False, new_training_updates=0, new_RL_updates=0)
    except BaseException as error:
        common.write(out / "failure.json", dict(error=repr(error), traceback=traceback.format_exc(),
                                               active_request=active_request, **counts))
        status("failed", error=repr(error), active_request=active_request)
        raise
    finally:
        if observer:
            observer.close()


if __name__ == "__main__":
    main()
