"""Quality-gated S2 local-mask generation; no retries, teachers, or timing claim.

This script does not choose the S1 gate. A separately saved, hash-bound GO_S2
decision and passing CPU bridge receipt must exist before model loading.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline

from controlled_mask import ControlledMaskQwenImageEditPipeline, pack_binary_mask, tensor_digest
from local_mask_bridge import build_plan, install_local_executor
from run_s1 import BASE, LORA, LORA_FILE, ROOT, SHIFT3, append, decode, sha, write


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text())


def resolve_root(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def check_hash_bindings(mapping, *, default_dir=None):
    require(isinstance(mapping, dict) and mapping, "missing hash bindings")
    actual = {}
    for value, expected in mapping.items():
        path = Path(value)
        if not path.is_absolute():
            path = (default_dir if default_dir is not None and len(path.parts) == 1 else ROOT) / path
        require(path.is_file() and sha(path) == expected, f"frozen file changed: {path}")
        actual[str(path.resolve())] = expected
    return actual


def check_reference(reference):
    """Read all gate bindings before creating an attempt or touching the GPU."""
    decision_path = ROOT / "data/s1_decision.json"
    decision = load(decision_path)
    require(decision.get("decision") == "GO_S2", "S1 quality decision is not GO_S2")
    require(resolve_root(decision["reference_attempt"]).resolve() == reference.resolve(), "decision points to another S1 attempt")
    prereg_path = reference / "preregistration.json"
    require(sha(prereg_path) == decision["preregistration_sha256"], "S1 preregistration binding changed")
    manual_path = resolve_root(decision["manual_review_path"])
    require(manual_path.is_file() and sha(manual_path) == decision["manual_review_sha256"], "manual review binding changed")
    prereg = load(prereg_path)
    require(prereg["steps"] == 8 and prereg["cfg"] == 1 and prereg["target_hw"] == [1024, 1024], "S1 configuration differs")
    check_hash_bindings(prereg["code_sha256"])
    cases_path = reference / "cases_s1.json"
    require(sha(cases_path) == prereg["cases_sha256"], "reference cases changed")
    spec = load(cases_path)
    require(spec["n_sources"] == len(spec["cases"]) == 8, "exactly eight reference sources required")
    require([x["case_id"] for x in spec["cases"]] == prereg["case_ids"], "reference source order changed")
    status = load(reference / "status.json")
    require(status["phase"] == "generation_complete_quality_pending" and status["completed_dit"] == status["invoked_dit"] == 144
            and status["successful_requests"] == 18, "S1 generation was not complete")
    receipt_path = ROOT / "data/cpu_local_bridge_v2.json"
    receipt = load(receipt_path)
    require(receipt["exit_code"] == 0 and receipt["result"]["errors"] == receipt["result"]["failures"] == 0,
            "CPU local bridge v2 did not pass")
    receipt_files = check_hash_bindings(receipt["code_sha256"], default_dir=ROOT / "scripts")
    required = {str((ROOT / "scripts" / name).resolve()) for name in (
        "controlled_mask.py", "local_mask_bridge.py", "legacy_local_executor.py",
        "legacy_refresh_policy.py", "test_local_mask_bridge.py")}
    require(required.issubset(receipt_files), "CPU bridge receipt lacks required runtime/test bindings")
    references = {}
    for case in spec["cases"]:
        require(case["official_split"] == "train" and case["role"] == "nonholdout_training_pilot", "not a training-pilot source")
        for key, hashkey in (("image", "source_sha256"), ("user_mask", "user_mask_sha256"), ("allow_mask", "allow_mask_sha256")):
            require(sha(ROOT / case[key]) == case[hashkey], f"source asset changed: {case['case_id']} {key}")
        folder = reference / case["case_id"]
        files = {name: folder / name for name in ("MASK_FULL.json", "MASK_FULL.pt", "MASK_FULL_control.pt", "MASK_FULL.png")}
        require(all(path.is_file() for path in files.values()), "missing S1 MASK_FULL artifact")
        row = load(files["MASK_FULL.json"])
        require(row["case_id"] == case["case_id"] and row["arm"] == "MASK_FULL" and row["calls"] == 8
                and row["mask_record"]["completed"] and row["mask_record"]["dit_calls"] == 8, "incomplete S1 mask reference")
        payload = torch.load(files["MASK_FULL.pt"], map_location="cpu", weights_only=True)
        control = torch.load(files["MASK_FULL_control.pt"], map_location="cpu", weights_only=True)
        require(len(payload["states"]) == 8 and payload["final"].shape == (1, 4096, 64), "bad reference state geometry")
        require(tensor_digest(payload["final"]) == row["final_sha256"], "S1 final tensor hash changed")
        require(torch.equal(payload["final"], payload["states"][-1]), "S1 final/state mismatch")
        require(torch.equal(control["source"], payload["ref_cond"]), "S1 background source/reference differ")
        require(tensor_digest(payload["ref_cond"]) == row["condition_sha256"], "S1 reference hash changed")
        for field in ("source", "packed_mask", "pixel_mask", "latent_mask"):
            require(tensor_digest(control[field]) == row["mask_record"]["tensor_sha256"][field], f"S1 {field} binding changed")
        require(row["mask_record"]["active_target_ids"] == control["active_ids"].tolist(), "S1 mask support differs")
        require(all(tensor_digest(state) == step["state_sha256"] for state, step in zip(payload["states"], row["mask_record"]["steps"])),
                "S1 poststep tensor binding changed")
        references[case["case_id"]] = dict(row=row, payload=payload, control=control,
                                            files={str(path): sha(path) for path in files.values()})
    noise_path = reference / "explicit_noise.pt"
    noise = torch.load(noise_path, map_location="cpu", weights_only=True)
    require(noise.shape == (1, 4096, 64) and noise.dtype == torch.bfloat16 and torch.isfinite(noise).all().item(), "bad explicit S1 noise")
    require(all(tensor_digest(noise) == item["row"]["mask_record"]["tensor_sha256"]["noise"] for item in references.values()),
            "S1 references used different initial noise")
    return dict(decision=decision, decision_path=decision_path, prereg=prereg, manual_path=manual_path,
                receipt=receipt, receipt_path=receipt_path, spec=spec, references=references, noise=noise,
                noise_path=noise_path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-attempt", default="runs/s1_v1")
    parser.add_argument("--attempt", default="s2_v1")
    args = parser.parse_args()
    reference = resolve_root(args.reference_attempt)
    out = ROOT / "runs" / args.attempt
    require(not out.exists(), "preserve existing attempt; select a new explicit attempt name")
    checked = check_reference(reference)
    spec, references = checked["spec"], checked["references"]
    weights_path = Path("/workspace/model-weights/deployment_20261028.json")
    weights = load(weights_path)
    require(weights["phase"] == "complete" and sha(weights_path) == checked["prereg"]["weights_receipt_sha256"], "weight deployment differs from S1")
    require(all(f.get("verified") for model in weights["models"] for f in model["files"]), "weights not verified")
    require(sha(Path(LORA) / LORA_FILE) == "36ee4ff860856559b9c581eaa69414900488f3de8be2f2448586daec77e201b9", "Lightning LoRA hash differs")
    diagnostic = [case for case in spec["cases"] if str(case["img_id"]) == "15953"]
    require(len(diagnostic) == 1, "unique diagnostic source 15953 missing")
    source_files = [Path(__file__), ROOT / "scripts/run_s1.py", ROOT / "scripts/controlled_mask.py",
                    ROOT / "scripts/local_mask_bridge.py", ROOT / "scripts/legacy_local_executor.py",
                    ROOT / "scripts/legacy_refresh_policy.py", Path(inspect.getfile(QwenImageEditPipeline)),
                    Path(inspect.getfile(FlowMatchEulerDiscreteScheduler))]
    frozen_inputs = {str(checked["noise_path"]): sha(checked["noise_path"])}
    plans = {}
    for case in spec["cases"]:
        item = references[case["case_id"]]
        frozen_inputs.update(item["files"])
        plans[case["case_id"]] = build_plan(item["control"]["packed_mask"], [64, 64], case["case_id"], kind="front", halo=2)
    all_selected_plan = build_plan(references[diagnostic[0]["case_id"]]["control"]["packed_mask"], [64, 64], diagnostic[0]["case_id"], kind="all_selected", halo=2)
    out.mkdir(parents=True)
    protocol = dict(phase="FROZEN_BEFORE_GENERATION", created_unix=time.time(),
                    reference_attempt=str(reference), reference_preregistration_sha256=sha(reference / "preregistration.json"),
                    s1_decision_sha256=sha(checked["decision_path"]), manual_review_path=str(checked["manual_path"]),
                    manual_review_sha256=sha(checked["manual_path"]), cpu_bridge_receipt_sha256=sha(checked["receipt_path"]),
                    code_sha256={str(path): sha(path) for path in source_files}, reference_artifact_sha256=frozen_inputs,
                    cases_sha256=sha(reference / "cases_s1.json"), weights_receipt_sha256=sha(weights_path),
                    target_hw=[1024, 1024], target_tokens=4096, condition_tokens=4096, dtype="bfloat16",
                    cfg=1, steps=8, scheduler_config=SHIFT3, shared_noise_tensor_sha256=tensor_digest(checked["noise"]),
                    planned_requests=9, planned_total_dit=72, diagnostic_requests=1, diagnostic_dit=8,
                    front_requests=8, front_dit=64, extra_teachers=0, warmups=0, fitting_updates=0, rl_updates=0,
                    full_selected_diagnostic_case=diagnostic[0]["case_id"], front_case_order=[case["case_id"] for case in spec["cases"]],
                    plans=plans, all_selected_plan=all_selected_plan, velocity_correction="none", rgb_copy=False,
                    timing_claim=False, quality_gate="post-generation independent semantic/numeric evaluation; no automatic GO",
                    failure_policy="stop first error or full-selected bitwise mismatch; preserve partial ledger; no automatic retry")
    write(out / "preregistration.json", protocol)
    write(out / "cases_s2.json", spec)
    torch.save(checked["noise"], out / "explicit_noise.pt")
    counts = dict(invoked_dit=0, completed_dit=0, successful_requests=0)
    active = {}

    def status(phase, **extra):
        write(out / "status.json", dict(phase=phase, updated_unix=time.time(), **counts, **extra))

    status("loading_model")
    try:
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        native = QwenImageEditPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
        native.scheduler = FlowMatchEulerDiscreteScheduler.from_config(SHIFT3)
        native.vae.enable_tiling()
        native.transformer.set_attention_backend("native")
        native.set_progress_bar_config(disable=True)
        for model in (native.transformer, native.text_encoder, native.vae):
            model.eval().requires_grad_(False)
        native.load_lora_weights(LORA, weight_name=LORA_FILE, adapter_name="lightning8", local_files_only=True)
        native.fuse_lora(adapter_names=["lightning8"], safe_fusing=True)
        native.unload_lora_weights()
        pipe = ControlledMaskQwenImageEditPipeline(**native.components)
        pipe.set_progress_bar_config(disable=True)
        require(pipe.vae_scale_factor == 8 and pipe.transformer.config.in_channels == 64 and len(pipe.transformer.transformer_blocks) == 60,
                "loaded model geometry changed")
        s1_env = load(reference / "environment.json")
        require(sha(inspect.getfile(type(pipe.transformer))) == s1_env["transformer_sha256"], "transformer implementation differs from S1")
        write(out / "environment.json", dict(torch=torch.__version__, cuda=torch.version.cuda,
              driver_gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
              packages={name: importlib.metadata.version(name) for name in ("diffusers", "transformers", "huggingface_hub", "peft", "accelerate", "safetensors", "numpy", "Pillow")},
              git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              git_status=subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True),
              scheduler_config=dict(pipe.scheduler.config), transformer_sha256=sha(inspect.getfile(type(pipe.transformer)))))
        noise = checked["noise"].to(device="cuda")
        order = [(diagnostic[0], "MASK_LOCAL_ALL_SELECTED", all_selected_plan)]
        order.extend((case, "MASK_LOCAL_FRONT", plans[case["case_id"]]) for case in spec["cases"])
        for case, arm, plan in order:
            case_id = case["case_id"]
            folder = out / case_id
            folder.mkdir(exist_ok=True)
            item = references[case_id]
            image = Image.open(ROOT / case["image"]).convert("RGB")
            mask = Image.open(ROOT / case["allow_mask"]).convert("L")
            actual_packed = pack_binary_mask(mask, image.size, 1024, 1024, 8, 16, device="cpu")[2]
            require(torch.equal(actual_packed, item["control"]["packed_mask"]), "actual allow mask differs from S1")
            require(plan["support_indices"] == item["control"]["active_ids"].tolist(), "plan mask support differs from S1")
            write(folder / (arm + "_plan.json"), plan)
            active.clear()
            active.update(case_id=case_id, arm=arm, calls=0, per_step=[], previous_velocity_sha256=None)
            append(out / "requests.jsonl", dict(event="started", case_id=case_id, arm=arm,
                   initial_noise_sha256=tensor_digest(noise), reference_result_sha256=sha(reference / case_id / "MASK_FULL.json"),
                   plan_sha256=sha(folder / (arm + "_plan.json")), **counts))
            status("running", case_id=case_id, arm=arm)
            diagnostic_started = time.perf_counter()
            with install_local_executor(pipe, plan, condition_count=4096) as base:
                dispatch = pipe.transformer.forward

                def counted_forward(*positional, **fields):
                    # Count actual model.forward body entry, not a module pre-hook.
                    step = active["calls"]
                    counts["invoked_dit"] += 1
                    active["calls"] += 1
                    append(out / "calls.jsonl", dict(event="invoked", case_id=case_id, arm=arm, step=step, **counts))
                    require(not positional and step < 8 and counts["invoked_dit"] <= 72, "extra/positional model call")
                    if active["previous_velocity_sha256"] is not None:
                        require(tensor_digest(base.velocity) == active["previous_velocity_sha256"], "scheduler/mask fusion modified raw velocity cache")
                    hidden = fields["hidden_states"]
                    require(hidden.shape == (1, 8192, 64) and hidden.dtype == torch.bfloat16 and torch.isfinite(hidden).all().item(), "unexpected local model input")
                    require(fields.get("guidance") is None and not fields.get("attention_kwargs"), "CFG/options changed")
                    require([list(shape) for shape in fields["img_shapes"][0]] == [[1, 64, 64], [1, 64, 64]], "actual image shapes changed")
                    require(pipe.scheduler.timesteps.cpu().tolist() == item["row"]["timesteps"] and pipe.scheduler.sigmas.cpu().tolist() == item["row"]["sigmas"], "actual schedule differs from S1")
                    expected_t = (torch.tensor(item["row"]["timesteps"][step], device=hidden.device, dtype=hidden.dtype)/1000).reshape(1)
                    require(torch.equal(fields["timestep"], expected_t), "actual model time differs from S1")
                    require(torch.equal(hidden[:, 4096:].cpu(), item["payload"]["ref_cond"]), "actual reference condition differs from S1")
                    text_sha = tensor_digest(fields["encoder_hidden_states"])
                    text_mask_sha = tensor_digest(fields["encoder_hidden_states_mask"]) if fields.get("encoder_hidden_states_mask") is not None else None
                    require(text_sha == item["row"]["text_sha256"] and text_mask_sha == item["row"]["text_mask_sha256"], "actual prompt encoding differs from S1")
                    record = pipe.last_mask_run
                    require(torch.equal(record.source.cpu(), item["control"]["source"]) and torch.equal(record.packed_mask.cpu(), item["control"]["packed_mask"]), "actual source/mask differs from S1")
                    if step == 0:
                        require(torch.equal(hidden[:, :4096], noise), "initial target differs from saved S1 noise")
                    result = dispatch(*positional, **fields)
                    counts["completed_dit"] += 1
                    append(out / "calls.jsonl", dict(event="completed", case_id=case_id, arm=arm, step=step, **counts))
                    raw = result.sample if hasattr(result, "sample") else result[0]
                    require(raw.shape == (1, 8192, 64) and torch.isfinite(raw).all().item(), "bad local full output")
                    require(raw.data_ptr() == base.velocity.data_ptr(), "legacy raw velocity return contract changed")
                    active["previous_velocity_sha256"] = tensor_digest(base.velocity)
                    row = dict(step=step, selected_target=plan["target_tokens_per_step"][step],
                               target_input_sha256=tensor_digest(hidden[:, :4096]), raw_velocity_sha256=active["previous_velocity_sha256"],
                               condition_sha256=tensor_digest(hidden[:, 4096:]), text_sha256=text_sha, text_mask_sha256=text_mask_sha,
                               raw_cache_bytes=base.cache_bytes(), age_sha256=tensor_digest(base.age),
                               age_trace=copy.deepcopy(base.age_trace[-1]))
                    active["per_step"].append(row)
                    write(folder / (arm + "_partial.json"), dict(completed=False, steps=active["per_step"], **counts))
                    return result

                pipe.transformer.forward = counted_forward
                try:
                    result = pipe(image=image, prompt=case["prompt"], mask_image=mask, initial_noise=noise,
                                  true_cfg_scale=1, num_inference_steps=8, output_type="latent", trace_latents=True)
                    require(tensor_digest(base.velocity) == active["previous_velocity_sha256"], "last mask fusion modified raw cache")
                finally:
                    # The bridge context then restores the real native forward.
                    pipe.transformer.forward = dispatch
            record = pipe.last_mask_run
            require(active["calls"] == 8 and record.completed and len(record.step_latents) == 8, "incomplete local request")
            final = result.images
            checks = dict(source_same_as_s1=torch.equal(record.source.cpu(), item["control"]["source"]),
                          ref_same_as_s1=torch.equal(record.ref_cond.cpu(), item["payload"]["ref_cond"]),
                          actual_mask_same_as_s1=torch.equal(record.packed_mask.cpu(), item["control"]["packed_mask"]))
            if arm == "MASK_LOCAL_ALL_SELECTED":
                checks["stepwise_bitwise_equal_to_s1_MASK_FULL"] = all(torch.equal(a, b) for a, b in zip(record.step_latents, item["payload"]["states"]))
                checks["final_bitwise_equal_to_s1_MASK_FULL"] = torch.equal(final.cpu(), item["payload"]["final"])
            torch.save(dict(final=final.cpu(), states=record.step_latents, ref_cond=record.ref_cond.cpu()), folder / (arm + ".pt"))
            torch.save(dict(source=record.source.cpu(), noise=record.noise.cpu(), packed_mask=record.packed_mask.cpu(),
                            pixel_mask=record.pixel_mask.cpu(), latent_mask=record.latent_mask.cpu(), active_ids=record.active_target_ids.cpu()), folder / (arm + "_control.pt"))
            decode(native, final).save(folder / (arm + ".png"))
            row = dict(case_id=case_id, arm=arm, calls=active["calls"], completed=True, diagnostic_checks=checks,
                       final_sha256=tensor_digest(final), diagnostic_seconds=time.perf_counter()-diagnostic_started,
                       timing_claim=False, mask_record=record.manifest(), bridge_record=base.bridge_record,
                       per_step=active["per_step"], reference_result_sha256=sha(reference / case_id / "MASK_FULL.json"),
                       artifacts_sha256={str(folder / (arm + suffix)): sha(folder / (arm + suffix)) for suffix in (".pt", ".png", "_control.pt", "_plan.json")}, **counts)
            write(folder / (arm + ".json"), row)
            require(all(checks.values()), "full-selected or input equivalence check failed; no FRONT requests will follow")
            counts["successful_requests"] += 1
            row.update(counts)
            write(folder / (arm + ".json"), row)
            append(out / "runs.jsonl", row)
            append(out / "requests.jsonl", dict(event="completed", case_id=case_id, arm=arm, **counts))
            status("running", case_id=case_id, arm=arm)
        require(counts == dict(invoked_dit=72, completed_dit=72, successful_requests=9), "S2 accounting mismatch")
        status("generation_complete_quality_pending", teachers=0, fitting_updates=0, rl_updates=0, timing_claim=False)
    except BaseException as exc:
        status("failed", error=type(exc).__name__ + ": " + str(exc), active_case=active.get("case_id"), active_arm=active.get("arm"))
        if active:
            append(out / "requests.jsonl", dict(event="failed", case_id=active.get("case_id"), arm=active.get("arm"),
                                                error=type(exc).__name__ + ": " + str(exc), **counts))
            record = getattr(locals().get("pipe"), "last_mask_run", None)
            if record is not None and getattr(record, "step_latents", None):
                folder = out / active["case_id"]
                torch.save(dict(states=record.step_latents, completed=False), folder / (active["arm"] + "_failed_partial.pt"))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
