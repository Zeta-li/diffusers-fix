"""Audited native/free-region requests from a frozen experiment manifest."""

import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import QwenImageEditPipeline, FlowMatchEulerDiscreteScheduler

import run_s1 as common
from controlled_mask import tensor_digest
from free_region import build_region_plan, run_free_region

ROOT = common.ROOT


def load_pipeline():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pipe = QwenImageEditPipeline.from_pretrained(
        common.BASE, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(common.SHIFT3)
    pipe.vae.enable_tiling()
    pipe.transformer.set_attention_backend("native")
    pipe.set_progress_bar_config(disable=True)
    for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
        model.eval().requires_grad_(False)
    pipe.load_lora_weights(common.LORA, weight_name=common.LORA_FILE,
                           adapter_name="lightning8", local_files_only=True)
    pipe.fuse_lora(adapter_names=["lightning8"], safe_fusing=True)
    pipe.unload_lora_weights()
    return pipe


def make_plan(case, arm):
    if arm["kind"] in ("native", "learned", "matched"):
        return None
    box = case["bbox"]
    envelope = arm.get("envelope")
    if "margin" in arm:
        d = arm["margin"]
        w, h = case["source_wh"]
        envelope = [max(0, box[0]-d), max(0, box[1]-d),
                    min(w, box[2]+d), min(h, box[3]+d)]
    return build_region_plan(
        box, case["source_wh"], [64,64], case["case_id"], halo=arm.get("halo", 0),
        target_envelope_xyxy=envelope,
        kind="all_selected" if arm["kind"] == "all_selected" else "front",
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    spec = json.loads(protocol_path.read_text())
    out = ROOT / "runs" / args.attempt
    if out.exists() or Path(args.attempt).name != args.attempt:
        raise RuntimeError("new attempt name required")
    if spec["phase"] != "FROZEN_BEFORE_GENERATION":
        raise RuntimeError("protocol not frozen")
    receipt = json.loads((ROOT / "data/cpu_free_region_v1.json").read_text())
    if receipt["exit_code"] != 0:
        raise RuntimeError("free-region CPU contract failed")
    for name, digest in receipt["code_sha256"].items():
        if common.sha(ROOT / "scripts" / name) != digest:
            raise RuntimeError("free-region implementation changed after tests: " + name)
    has_head=any(a["kind"] in ("learned","matched") for c in spec["cases"] for a in c["arms"])
    if has_head:
        online_receipt=json.loads((ROOT/"data/cpu_online_expansion_v2.json").read_text())
        if online_receipt["exit_code"]!=0:
            raise RuntimeError("online selector tests failed")
        for name,digest in online_receipt["code_sha256"].items():
            if common.sha(ROOT/"scripts"/name)!=digest:
                raise RuntimeError("online code changed after tests")
        if common.sha(ROOT/spec["checkpoint"])!=spec["checkpoint_sha256"]:
            raise RuntimeError("checkpoint changed after freeze")
    for case in spec["cases"]:
        if common.sha(ROOT / case["image"]) != case["source_sha256"]:
            raise RuntimeError("source image hash changed")
    planned = sum(len(case["arms"]) for case in spec["cases"])
    if planned != spec["planned_requests"] or planned*8 != spec["planned_dit"]:
        raise RuntimeError("protocol budget mismatch")
    out.mkdir(parents=True)
    shutil.copy2(protocol_path, out / "preregistration.json")
    snap = out / "code_snapshot"
    snap.mkdir()
    code = {}
    for name in ("run_object_selectivity.py", "run_s1.py", "free_region.py", "local_mask_bridge.py",
                 "legacy_local_executor.py", "legacy_refresh_policy.py"):
        src = ROOT / "scripts" / name
        shutil.copy2(src, snap / name)
        code[name] = common.sha(src)
    if has_head:
        for name in ("online_expansion.py","expansion_head.py"):
            shutil.copy2(ROOT/"scripts"/name,snap/name)
            code[name]=common.sha(ROOT/"scripts"/name)
    common.write(out / "code_sha256.json", code)
    noise_path = ROOT / spec["noise_path"]
    if common.sha(noise_path) != spec["noise_sha256"]:
        raise RuntimeError("noise changed after freeze")
    noise_cpu = torch.load(noise_path, weights_only=True, map_location="cpu")
    torch.save(noise_cpu, out / "explicit_noise.pt")
    counts = dict(invoked_dit=0, completed_dit=0, successful_requests=0)
    active = {}
    def status(phase, **extra):
        common.write(out / "status.json", dict(phase=phase, updated_unix=time.time(), **counts, **extra))
    status("loading_model")
    try:
        pipe = load_pipeline()
        if has_head:
            from expansion_head import ExpansionPredictor, CenteredBudgetPredictor
            from online_expansion import run_online_region
            predictor=ExpansionPredictor(ROOT/spec["checkpoint"])
            matched_predictor=CenteredBudgetPredictor(predictor)
        common.write(out / "environment.json", dict(
            torch=torch.__version__, cuda=torch.version.cuda,
            user_confirmed_gpu="H800", driver_name=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()),
            git_head=subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip(),
            scheduler=dict(pipe.scheduler.config), code_sha256=code,
            native_pipeline_sha256=common.sha(inspect.getfile(QwenImageEditPipeline)),
            weights_receipt_sha256=common.sha("/workspace/model-weights/deployment_20261028.json"),
        ))
        noise = noise_cpu.to("cuda")
        def before(module, positional, fields):
            step = active["calls"]
            if positional or step >= 8:
                raise RuntimeError("unexpected transformer call")
            hidden = fields["hidden_states"]
            if hidden.shape != (1,8192,64):
                raise RuntimeError("unexpected full target/reference geometry")
            if step == 0:
                if not torch.equal(hidden[:,:4096], noise):
                    raise RuntimeError("initial noise mismatch")
                active["condition"] = hidden[:,4096:].detach().cpu().clone()
                active["text_pool"] = F.adaptive_avg_pool1d(
                    fields["encoder_hidden_states"].float().mean(1).unsqueeze(1), 64
                ).squeeze(1).cpu()
            active["input"] = hidden[:,:4096].detach().float().clone()
            active["sigma"] = float(pipe.scheduler.sigmas[step].item())
            counts["invoked_dit"] += 1
            active["calls"] += 1
            common.append(out / "calls.jsonl", dict(event="invoked", case=active["case_id"],
                arm=active["arm"], step=step, **counts))
        def after(module, positional, result):
            counts["completed_dit"] += 1
            step = active["calls"] - 1
            v = result[0][:,:4096]
            if not bool(torch.isfinite(v).all()):
                raise RuntimeError("nonfinite velocity")
            active["velocities"].append(v.detach().cpu().clone())
            if step == 1:
                active["warmup_clean"] = (active["input"]-active["sigma"]*v.float()).cpu()
            common.append(out / "calls.jsonl", dict(event="completed", case=active["case_id"],
                arm=active["arm"], step=step, **counts))
        pre = pipe.transformer.register_forward_pre_hook(before, with_kwargs=True)
        post = pipe.transformer.register_forward_hook(after)
        for case in spec["cases"]:
            case_dir = out / case["case_id"]
            case_dir.mkdir()
            image = Image.open(ROOT / case["image"]).convert("RGB")
            if list(image.size) != case["source_wh"]:
                raise RuntimeError("source size mismatch")
            native_states = None
            native_record = None
            learned_counts = None
            if case.get("reference_run"):
                reference=ROOT/"runs"/case["reference_run"]/case["case_id"]
                native_states=torch.load(reference/"F.pt",map_location="cpu",weights_only=True)["states"]
                native_record=SimpleNamespace(**json.loads((reference/"F.json").read_text())["record"])
            for arm in case["arms"]:
                name = arm["name"]
                active.clear()
                active.update(case_id=case["case_id"], arm=name, calls=0, velocities=[])
                status("running", case_id=case["case_id"], arm=name)
                plan = make_plan(case, arm)
                started = time.perf_counter()
                if arm["kind"] in ("learned","matched"):
                    result=run_online_region(pipe,image,case["prompt"],noise,case,
                        predictor if arm["kind"]=="learned" else matched_predictor)
                else:
                    result = run_free_region(pipe, image, case["prompt"], noise, plan=plan,
                                             trace_latents=True, output_type="latent")
                record = pipe.last_free_region_run
                plan=record.plan
                torch.save(dict(final=result.images.cpu(), states=record.step_latents,
                    condition=active["condition"], warmup_clean=active["warmup_clean"],
                    text_pool=active["text_pool"], velocities=active["velocities"]),
                    case_dir / (name+"_raw_before_checks.pt"))
                checks = {}
                previous = noise_cpu
                outside = sorted(set(range(4096))-set(plan["source_bbox_indices"])) if plan else list(range(4096))
                changed = []
                for k, state in enumerate(record.step_latents):
                    dt = pipe.scheduler.sigmas[k+1]-pipe.scheduler.sigmas[k]
                    # Match the scheduler's BF16 velocity multiplication before
                    # adding to its FP32 sample, on the same device.
                    expected = (previous.to("cuda").float()+dt*active["velocities"][k].to("cuda")).to(state.dtype).cpu()
                    if not torch.equal(state, expected):
                        raise RuntimeError("native Euler update modified")
                    changed.append(not torch.equal(previous[:,outside], state[:,outside]))
                    previous = state
                checks["all_tokens_unmodified_euler"] = True
                checks["outside_hint_changed_each_step"] = changed
                if arm["kind"]=="learned":
                    learned_counts=plan["target_tokens_per_step"]
                if arm["kind"]=="matched":
                    checks["matched_each_step_token_budget"]=(learned_counts==plan["target_tokens_per_step"])
                    if not checks["matched_each_step_token_budget"]:
                        raise RuntimeError("matched geometry differs in token budget")
                if arm["kind"] == "native":
                    native_states = [x.clone() for x in record.step_latents]
                    native_record = record
                    common.decode(pipe, active["condition"].to("cuda")).save(case_dir / "SOURCE_RECON.png")
                elif native_record is not None:
                    for key in ("initial_noise_sha256", "reference_condition_sha256", "text_sha256", "text_mask_sha256"):
                        if getattr(record,key) != getattr(native_record,key):
                            raise RuntimeError("paired input mismatch: "+key)
                    checks["first_two_states_bitwise_native"] = all(
                        torch.equal(a,b) for a,b in zip(record.step_latents[:2], native_states[:2]))
                    if not checks["first_two_states_bitwise_native"]:
                        raise RuntimeError("online warmup differs from frozen full reference")
                    if arm["kind"] == "all_selected":
                        errors = [(a.float()-b.float()).abs().max().item() for a,b in zip(record.step_latents,native_states)]
                        checks["all_selected_stepwise_max_abs"] = errors
                        checks["all_selected_bitwise_native"] = all(e == 0 for e in errors)
                        if not checks["all_selected_bitwise_native"]:
                            common.write(case_dir / (name+"_equivalence_failure.json"), checks)
                            raise RuntimeError("real-weight all-selected not bitwise native")
                common.decode(pipe, result.images).save(case_dir / (name+".png"))
                torch.save(dict(final=result.images.cpu(), states=record.step_latents,
                    condition=active["condition"], warmup_clean=active["warmup_clean"],
                    text_pool=active["text_pool"]), case_dir / (name+".pt"))
                counts["successful_requests"] += 1
                row = dict(case_id=case["case_id"], arm=name, checks=checks, **counts,
                    diagnostic_seconds=time.perf_counter()-started,
                    net_latency_claim=False, record=record.manifest())
                common.write(case_dir / (name+".json"), row)
                common.append(out / "runs.jsonl", dict(case_id=case["case_id"], arm=name, checks=checks,
                    **counts, net_latency_claim=False))
                status("running", case_id=case["case_id"], arm=name)
        pre.remove()
        post.remove()
        if counts != dict(invoked_dit=planned*8,completed_dit=planned*8,successful_requests=planned):
            raise RuntimeError("final budget mismatch")
        status("generation_complete_quality_pending")
    except BaseException as exc:
        status("failed", error=type(exc).__name__+": "+str(exc), active_case=active.get("case_id"),
               active_arm=active.get("arm"))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
