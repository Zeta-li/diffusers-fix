"""Frozen six-request S1 failure diagnosis; no local execution or training.

Reads data/s1_repair_preregistration.json and reuses the original S1 noise
and source controls. This creates a separate attempt and never overwrites
the original failed cases. Run with the same environment as run_s1.py.
There are exactly six permitted requests / at most 48 real DiT forwards;
on any failure the partial ledger is retained and execution stops.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import run_s1 as s1

import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline

from controlled_mask import ControlledMaskQwenImageEditPipeline, pack_binary_mask, tensor_digest


ROOT = s1.ROOT
EXPECTED_TIMESTEPS = [1000., 954.5454711914062, 900., 833.3333740234375, 750., 642.8570556640625, 500., 300.]
EXPECTED_ARMS = ["MASK_FULL", "MASK_FULL", "NATIVE_FULL", "MASK_FULL", "NATIVE_FULL", "MASK_FULL"]
EXPECTED_VARIANTS = ["goat_ALLOW32", "goat_ALLOW48", "dog_specific_NATIVE", "dog_specific_MASK", "remote_specific_NATIVE", "remote_specific_MASK"]


def require(condition, message):
    # These guards must survive python -O; they are execution contracts.
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_tensor_file(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def stage_path(path):
    result = (ROOT / path).resolve()
    require(result.is_relative_to(ROOT.resolve()), "input path escapes the research stage")
    require(result.is_file(), "missing input: " + str(result))
    return result


def finite(tensor, name):
    require(bool(torch.isfinite(tensor).all().item()), name + " contains non-finite values")


def preflight(protocol_path):
    spec = read_json(protocol_path)
    require(spec["phase"] == "FROZEN_BEFORE_GENERATION", "repair protocol is not frozen")
    require(spec["reference_attempt"] == "runs/s1_v2", "unexpected S1 reference attempt")
    require(spec["planned_requests"] == 6 and spec["planned_dit"] == 48, "repair budget must be 6 requests / 48 DiT")
    variants = spec["variants"]
    require(len(variants) == 6, "expected exactly six frozen repair variants")
    require([v["arm"] for v in variants] == EXPECTED_ARMS, "repair execution order/arms changed")
    require([v["variant_id"] for v in variants] == EXPECTED_VARIANTS, "repair variant order changed")
    require(len({v["variant_id"] for v in variants}) == 6, "variant ids are not unique")
    for variant in variants:
        require(Path(variant["variant_id"]).name == variant["variant_id"], "variant id must be a directory name")
        require(isinstance(variant["prompt"], str) and bool(variant["prompt"].strip()), "empty prompt")
        require(bool(variant["intervention"]), "missing intervention description")
        for path_field, sha_field in (("image", "source_sha256"), ("mask_image", "mask_sha256")):
            require(s1.sha(stage_path(variant[path_field])) == variant[sha_field], "input SHA mismatch: " + variant["variant_id"] + ":" + path_field)
    require(variants[0]["case_id"] == variants[1]["case_id"], "first two variants must share the goat source")
    require(variants[2]["case_id"] == variants[3]["case_id"], "dog pair source mismatch")
    require(variants[4]["case_id"] == variants[5]["case_id"], "remote pair source mismatch")
    require(len({v["case_id"] for v in variants}) == 3, "expected three diagnostic source groups")
    require(variants[0]["prompt"] == variants[1]["prompt"], "goat expansion must hold prompt fixed")
    for first, second in ((variants[2], variants[3]), (variants[4], variants[5])):
        for field in ("prompt", "source_sha256", "mask_sha256"):
            require(first[field] == second[field], "native/masked repair pair differs in " + field)

    reference = ROOT / spec["reference_attempt"]
    require(s1.sha(reference / "preregistration.json") == spec["reference_preregistration_sha256"], "S1 reference protocol hash mismatch")
    require(s1.sha(ROOT / "data/s1_decision.json") == spec["s1_decision_sha256"], "frozen S1 quality decision hash mismatch")
    prior_status = read_json(reference / "status.json")
    require(prior_status["invoked_dit"] == prior_status["completed_dit"] == 144, "S1 reference does not contain 144 completed DiT")
    require(prior_status["successful_requests"] == 18, "S1 reference request count differs")
    prior_cases = read_json(reference / "cases_s1.json")
    prior_cases = {case["case_id"]: case for case in prior_cases["cases"]}
    noise_path = reference / "explicit_noise.pt"
    noise = load_tensor_file(noise_path)
    require(isinstance(noise, torch.Tensor) and tuple(noise.shape) == (1, 4096, 64), "unexpected original explicit noise shape")
    require(noise.dtype == torch.bfloat16, "original explicit noise must remain BF16")
    finite(noise, "original explicit noise")
    references = {}
    input_hashes = {"explicit_noise.pt": s1.sha(noise_path)}
    for variant in variants:
        case_id = variant["case_id"]
        prior = prior_cases[case_id]
        require(prior["official_split"] == "train" and prior["role"] == "nonholdout_training_pilot", "source is not a training-only diagnostic")
        require(prior["source_sha256"] == variant["source_sha256"], "repair changed the original source image")
        if case_id not in references:
            control_path = reference / case_id / "MASK_FULL_control.pt"
            prior_row_path = reference / case_id / "MASK_FULL.json"
            source = load_tensor_file(control_path)["source"]
            require(tuple(source.shape) == tuple(noise.shape) and source.dtype == noise.dtype, "S1 source/noise grid or dtype differs")
            finite(source, "original S1 source")
            references[case_id] = {
                "source": source.clone(), "source_tensor_sha256": tensor_digest(source),
                "control_path": str(control_path.relative_to(ROOT)),
                "control_file_sha256": s1.sha(control_path),
                "prior_row": read_json(prior_row_path),
                "prior_row_sha256": s1.sha(prior_row_path),
            }
            input_hashes[str(control_path.relative_to(ROOT))] = s1.sha(control_path)
            input_hashes[str(prior_row_path.relative_to(ROOT))] = s1.sha(prior_row_path)
        with Image.open(stage_path(variant["image"])) as opened:
            image = opened.convert("RGB")
        require(image.size == (500, 500), "this frozen diagnosis expects the original 500-square sources")
        with Image.open(stage_path(variant["mask_image"])) as opened:
            mask = opened.copy()
        pixels, latent, packed, ids = pack_binary_mask(mask, image.size, 1024, 1024, 8, 16, device="cpu")
        require(0 < ids.numel() < 4096, "repair task mask must be nonempty and genuinely local")
        # Freeze actual processed mask tensors rather than geometric estimates.
        variant["_mask_control"] = {"pixel_mask": pixels, "latent_mask": latent, "packed_mask": packed, "active_ids": ids}
        variant["_image"] = image
        variant["_mask"] = mask
    goat_case = prior_cases[variants[0]["case_id"]]
    require(variants[0]["prompt"] == goat_case["prompt"], "goat mask-only intervention changed its original prompt")

    adapter_path = ROOT / "data/adapter_environment_check.json"
    adapter = read_json(adapter_path)
    require(adapter["status"] == "PASS", "adapter environment check did not pass")
    require(all(importlib.metadata.version(name) == version for name, version in adapter["packages"].items()), "package environment drifted after S1")
    cpu_path = ROOT / "data/cpu_contract_v2.json"
    cpu = read_json(cpu_path)
    require(cpu["exit_code"] == 0 and cpu["result"]["errors"] == cpu["result"]["failures"] == 0, "mask CPU contracts did not pass")
    require(all(s1.sha(ROOT / "scripts" / name) == digest for name, digest in cpu["code_sha256"].items()), "mask code changed after CPU contract tests")
    weights_path = Path("/workspace/model-weights/deployment_20261028.json")
    weights = read_json(weights_path)
    require(weights["phase"] == "complete", "model verification is incomplete")
    require(weights["models"][0]["revision"] == "ac7f9318f633fc4b5778c59367c8128225f1e3de", "base revision mismatch")
    require(weights["models"][1]["revision"] == "d468c534b360e0754c6bf9e38ed49e1e09cd2e2a", "adapter revision mismatch")
    require(all(f.get("verified") for model in weights["models"] for f in model["files"]), "unverified model files")
    require(s1.sha(Path(s1.LORA) / s1.LORA_FILE) == "36ee4ff860856559b9c581eaa69414900488f3de8be2f2448586daec77e201b9", "adapter SHA mismatch")
    input_hashes.update({str(p): s1.sha(p) for p in (protocol_path, reference / "preregistration.json", ROOT / "data/s1_decision.json", reference / "cases_s1.json", reference / "status.json", adapter_path, cpu_path, weights_path)})
    return spec, variants, reference, noise, references, input_hashes, weights


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="s1_repair_v1")
    args = parser.parse_args()
    require(bool(args.attempt) and Path(args.attempt).name == args.attempt, "attempt must be one directory name")
    out = ROOT / "runs" / args.attempt
    require(not out.exists(), "preserve existing attempt; select a new explicit name")
    protocol_path = ROOT / "data/s1_repair_preregistration.json"
    # Freeze an unmodified JSON snapshot before attaching in-memory tensors.
    protocol_json = read_json(protocol_path)
    spec, variants, reference, noise_cpu, references, input_hashes, weights = preflight(protocol_path)
    out.mkdir(parents=True)
    s1.write(out / "s1_repair_preregistration.json", protocol_json)
    shutil.copy2(reference / "explicit_noise.pt", out / "explicit_noise.pt")
    snapshots = out / "code_snapshot"
    snapshots.mkdir()
    code_paths = {
        "run_s1_repair.py": Path(__file__),
        "run_s1.py": Path(inspect.getfile(s1)),
        "controlled_mask.py": ROOT / "scripts/controlled_mask.py",
        "test_controlled_mask.py": ROOT / "scripts/test_controlled_mask.py",
        "native_pipeline_qwenimage_edit.py": Path(inspect.getfile(QwenImageEditPipeline)),
        "flow_match_euler_discrete.py": Path(inspect.getfile(FlowMatchEulerDiscreteScheduler)),
    }
    for name, path in code_paths.items():
        shutil.copy2(path, snapshots / name)
    s1.write(out / "preregistration.json", {
        "phase": "FROZEN_BEFORE_GENERATION", "created_unix": time.time(),
        "reference_attempt": spec["reference_attempt"],
        "planned_requests": 6, "planned_dit": 48,
        "order": [{k: v for k, v in variant.items() if not k.startswith("_")} for variant in variants],
        "cfg": 1, "steps": 8, "dtype": "bfloat16", "target_hw": [1024, 1024],
        "scheduler_config": s1.SHIFT3, "seed": 2026102201,
        "explicit_noise_policy": "exact S1 tensor loaded from its file; no RNG resampling",
        "input_file_sha256": input_hashes,
        "noise_tensor_sha256": tensor_digest(noise_cpu),
        "code_sha256": {name: s1.sha(path) for name, path in code_paths.items()},
        "source_reference": {case_id: {key: value for key, value in ref.items() if key not in ("source", "prior_row")} for case_id, ref in references.items()},
        "local_execution": False, "quantization": False, "teachers": 0,
        "training_updates": 0, "rl_updates": 0, "warmup_requests": 0,
        "rgb_copy": False, "timing_claim": False,
        "failure_policy": "stop at first failure; persist partial state and ledger; no automatic retry",
        "counter_semantics": "invoked immediately before original transformer.forward; completed immediately after it returns, before finite validation; rejected preflight is not a body invocation",
    })
    counts = {"invoked_dit": 0, "completed_dit": 0, "successful_requests": 0}
    active = {}
    native = masked = original_forward = None

    def status(phase, **extra):
        s1.write(out / "status.json", {"phase": phase, "updated_unix": time.time(), **counts, **extra})

    status("loading_model")
    try:
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        native = QwenImageEditPipeline.from_pretrained(s1.BASE, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
        native.scheduler = FlowMatchEulerDiscreteScheduler.from_config(s1.SHIFT3)
        native.vae.enable_tiling()
        native.transformer.set_attention_backend("native")
        native.set_progress_bar_config(disable=True)
        for model in (native.transformer, native.text_encoder, native.vae):
            model.eval().requires_grad_(False)
        native.load_lora_weights(s1.LORA, weight_name=s1.LORA_FILE, adapter_name="lightning8", local_files_only=True)
        native.fuse_lora(adapter_names=["lightning8"], safe_fusing=True)
        native.unload_lora_weights()
        masked = ControlledMaskQwenImageEditPipeline(**native.components)
        masked.set_progress_bar_config(disable=True)
        require(native.vae_scale_factor == 8 and native.transformer.config.in_channels == 64, "unexpected model geometry")
        s1.write(out / "environment.json", {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "user_confirmed_gpu": "H800", "driver_gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "packages": {name: importlib.metadata.version(name) for name in ("diffusers", "transformers", "huggingface_hub", "peft", "torchao", "accelerate", "safetensors", "numpy", "Pillow")},
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True),
            "weights": weights, "scheduler_config": dict(native.scheduler.config),
            "transformer_sha256": s1.sha(inspect.getfile(type(native.transformer))),
        })
        noise = noise_cpu.to(device="cuda")
        pair_inputs = {}
        original_forward = native.transformer.forward

        def validate_forward_inputs(positional, fields):
            require(bool(active) and not positional, "unexpected positional/unowned DiT call")
            require(active["calls"] < 8 and counts["invoked_dit"] < 48, "hard DiT budget exceeded")
            hidden = fields["hidden_states"]
            require(tuple(hidden.shape) == (1, 8192, 64), "DiT input grid changed")
            finite(hidden, "DiT input")
            require(torch.equal(hidden[:, 4096:], active["source_gpu"]), "source condition differs from frozen S1 source")
            if active["calls"] == 0:
                require(torch.equal(hidden[:, :4096], noise), "first target differs from S1 explicit noise")
                active["ref_cond"] = hidden[:, 4096:].detach().cpu().clone()
                active["text_sha256"] = tensor_digest(fields["encoder_hidden_states"])
                active["text_mask_sha256"] = tensor_digest(fields["encoder_hidden_states_mask"]) if fields.get("encoder_hidden_states_mask") is not None else None
                active["timesteps"] = native.scheduler.timesteps.detach().cpu().tolist()
                active["sigmas"] = native.scheduler.sigmas.detach().cpu().tolist()
                require(active["timesteps"] == EXPECTED_TIMESTEPS, "actual Lightning8 timesteps changed")
                require(len(active["sigmas"]) == 9 and active["sigmas"][0] == 1 and active["sigmas"][-1] == 0, "actual sigma endpoints/count changed")
                require(active["sigmas"] == references[active["case_id"]]["prior_row"]["sigmas"], "sigma schedule differs from frozen S1")
                pair_key = active["case_id"]
                signature = (tensor_digest(active["ref_cond"]), active["text_sha256"], active["text_mask_sha256"])
                if active["arm"] == "NATIVE_FULL":
                    require(pair_key not in pair_inputs, "duplicate native repair pair")
                    pair_inputs[pair_key] = signature
                elif pair_key in pair_inputs:
                    require(pair_inputs[pair_key] == signature, "native/masked repair pair condition or text differs")

        def counted_forward(*positional, **fields):
            validate_forward_inputs(positional, fields)
            counts["invoked_dit"] += 1
            active["calls"] += 1
            event = {"variant_id": active["variant_id"], "case_id": active["case_id"], "arm": active["arm"], "step": active["calls"] - 1}
            s1.append(out / "calls.jsonl", {"event": "invoked", **event, **counts})
            result = original_forward(*positional, **fields)
            counts["completed_dit"] += 1
            s1.append(out / "calls.jsonl", {"event": "completed", **event, **counts})
            require(tuple(result[0].shape) == (1, 8192, 64), "DiT output grid changed")
            finite(result[0], "DiT output")
            return result

        native.transformer.forward = counted_forward
        for variant in variants:
            variant_dir = out / variant["variant_id"]
            variant_dir.mkdir()
            control = variant["_mask_control"]
            source_cpu = references[variant["case_id"]]["source"]
            active.clear()
            active.update(
                variant_id=variant["variant_id"], case_id=variant["case_id"], arm=variant["arm"],
                calls=0, states=[], steps=[], source_gpu=source_cpu.to(device="cuda"),
                output_dir=variant_dir,
            )
            status("running", variant_id=variant["variant_id"], case_id=variant["case_id"], arm=variant["arm"])
            variant["_mask"].save(variant_dir / "input_mask.png")
            record = None
            if variant["arm"] == "NATIVE_FULL":
                def trace(pipe, index, timestep, values):
                    require(index == len(active["states"]) and pipe.scheduler.step_index == index + 1, "native callback/scheduler order changed")
                    state = values["latents"].detach().cpu().clone()
                    active["states"].append(state)
                    active["steps"].append({
                        "step": index, "timestep": float(timestep.item()),
                        "scheduler_step_index": pipe.scheduler.step_index,
                        "sigma_current": float(pipe.scheduler.sigmas[index].item()),
                        "sigma_next": float(pipe.scheduler.sigmas[index + 1].item()),
                        "state_sha256": tensor_digest(state),
                    })
                    return {}

                result = native(
                    image=variant["_image"], prompt=variant["prompt"], true_cfg_scale=1,
                    num_inference_steps=8, height=1024, width=1024, latents=noise.clone(),
                    output_type="latent", callback_on_step_end=trace,
                )
            else:
                masked.last_mask_run = None
                result = masked(
                    image=variant["_image"], prompt=variant["prompt"], mask_image=variant["_mask"],
                    initial_noise=noise, true_cfg_scale=1, num_inference_steps=8,
                    output_type="latent", trace_latents=True,
                )
                record = masked.last_mask_run
                active["states"] = record.step_latents
                active["steps"] = record.steps
                require(torch.equal(record.source.cpu(), source_cpu), "mask source differs from original S1 source control")
                require(torch.equal(record.ref_cond.cpu(), active["ref_cond"]), "captured mask reference differs from actual model input")
                for name in ("pixel_mask", "latent_mask", "packed_mask"):
                    require(torch.equal(getattr(record, name).cpu(), control[name]), "runtime mask differs from preflight mask: " + name)
                require(torch.equal(record.active_target_ids.cpu(), control["active_ids"]), "runtime active ids differ from preflight mask")
            require(active["calls"] == 8 and len(active["states"]) == 8, "request did not complete eight recorded steps")
            final = result.images
            require(tuple(final.shape) == (1, 4096, 64), "unexpected final latent shape")
            finite(final, "final latent")
            source_equal = torch.equal(active["ref_cond"], source_cpu)
            require(source_equal, "actual source/reference was not frozen S1 source")
            diagnostics = {"initial_noise_reused_from_s1": True, "source_bitwise_s1": source_equal, "eight_steps_completed": True}
            require(torch.equal(noise.detach().cpu(), noise_cpu), "shared S1 initial noise was mutated")
            diagnostics["shared_noise_unchanged"] = True
            if variant["arm"] == "MASK_FULL" and variant["case_id"] in pair_inputs:
                diagnostics["paired_source_and_text_bitwise_equal"] = True
            if record is not None:
                final_cpu = final.detach().cpu()
                outside = ~control["packed_mask"]
                diagnostics["final_background_bitwise_source"] = torch.equal(final_cpu[outside], source_cpu[outside])
                require(diagnostics["final_background_bitwise_source"], "final masked background was not source")
            torch.save({
                "source": source_cpu.clone(), "ref_cond": active["ref_cond"].clone(),
                "noise": noise_cpu.clone(), **control,
            }, variant_dir / "control.pt")
            torch.save({
                "final": final.detach().cpu(), "states": active["states"],
                "source": source_cpu.clone(), "ref_cond": active["ref_cond"].clone(),
            }, variant_dir / "result.pt")
            s1.decode(native, final).save(variant_dir / "result.png")
            s1.decode(native, active["source_gpu"]).save(variant_dir / "SOURCE_RECON.png")
            counts["successful_requests"] += 1
            row = {
                "variant_id": variant["variant_id"], "case_id": variant["case_id"], "arm": variant["arm"],
                "prompt": variant["prompt"], "intervention": variant["intervention"],
                "calls": 8, "timesteps": active["timesteps"], "sigmas": active["sigmas"],
                "next_sigmas": active["sigmas"][1:], "steps": active["steps"],
                "source_image_sha256": variant["source_sha256"], "input_mask_sha256": variant["mask_sha256"],
                "noise_tensor_sha256": tensor_digest(noise_cpu), "source_tensor_sha256": tensor_digest(source_cpu),
                "condition_sha256": tensor_digest(active["ref_cond"]),
                "text_sha256": active["text_sha256"], "text_mask_sha256": active["text_mask_sha256"],
                "final_sha256": tensor_digest(final), "packed_mask_sha256": tensor_digest(control["packed_mask"]),
                "active_target_count": control["active_ids"].numel(),
                "active_target_fraction": control["active_ids"].numel() / 4096,
                "checks": diagnostics, "mask_applied": variant["arm"] == "MASK_FULL",
                "rgb_copy": False, "timing_claim": False, **counts,
            }
            if record is not None:
                row["mask_record"] = record.manifest()
            s1.write(variant_dir / "result.json", row)
            s1.append(out / "runs.jsonl", row)
            status("running", variant_id=variant["variant_id"], case_id=variant["case_id"], arm=variant["arm"])
        require(counts == {"invoked_dit": 48, "completed_dit": 48, "successful_requests": 6}, "repair final ledger does not match its fixed budget")
        status("generation_complete_quality_pending")
    except BaseException as exc:
        # Capture request-local snapshots even when the controlled call fails
        # after a subset of its callbacks. No automatic continuation occurs.
        partial = {"counts": dict(counts), "variant_id": active.get("variant_id"), "case_id": active.get("case_id"), "arm": active.get("arm"), "calls": active.get("calls", 0)}
        try:
            partial["states"] = active.get("states", [])
            partial["steps"] = active.get("steps", [])
            if active.get("arm") == "MASK_FULL" and masked is not None and masked.last_mask_run is not None:
                failed_record = masked.last_mask_run
                partial.update(states=failed_record.step_latents, steps=failed_record.steps,
                               source=failed_record.source.detach().cpu(), ref_cond=failed_record.ref_cond.detach().cpu(),
                               packed_mask=failed_record.packed_mask.detach().cpu())
            if "ref_cond" in active:
                partial["actual_ref_cond"] = active["ref_cond"]
            torch.save(partial, out / "failed_partial.pt")
        except BaseException as partial_exc:
            partial["partial_save_error"] = type(partial_exc).__name__ + ": " + str(partial_exc)
        status("failed", error=type(exc).__name__ + ": " + str(exc), active_variant=active.get("variant_id"),
               active_case=active.get("case_id"), active_arm=active.get("arm"), partial_save_error=partial.get("partial_save_error"))
        traceback.print_exc()
        raise
    finally:
        if native is not None and original_forward is not None:
            native.transformer.forward = original_forward


if __name__ == "__main__":
    main()
