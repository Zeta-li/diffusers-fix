import argparse
import hashlib
import inspect
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import traceback
import subprocess

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image
from diffusers import QwenImageEditPipeline, FlowMatchEulerDiscreteScheduler
from controlled_mask import ControlledMaskQwenImageEditPipeline, tensor_digest

ROOT = Path(__file__).resolve().parents[1]
BASE = "/workspace/model-weights/Qwen-Image-Edit"
LORA = "/workspace/model-weights/Qwen-Image-Lightning"
LORA_FILE = "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"
SHIFT3 = dict(base_image_seq_len=256, base_shift=math.log(3), invert_sigmas=False,
              max_image_seq_len=8192, max_shift=math.log(3), num_train_timesteps=1000,
              shift=1.0, shift_terminal=None, stochastic_sampling=False,
              time_shift_type="exponential", use_beta_sigmas=False, use_dynamic_shifting=True,
              use_exponential_sigmas=False, use_karras_sigmas=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def decode(pipe, packed):
    latent = pipe._unpack_latents(packed, 1024, 1024, pipe.vae_scale_factor).to(pipe.vae.dtype)
    mean = torch.tensor(pipe.vae.config.latents_mean, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
    invstd = 1.0 / torch.tensor(pipe.vae.config.latents_std, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
    latent = latent / invstd + mean
    value = pipe.vae.decode(latent, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(value, output_type="pil")[0]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="s1_v1")
    args = parser.parse_args()
    out = ROOT / "runs" / args.attempt
    if out.exists():
        raise RuntimeError("preserve existing attempt; use a new explicit attempt name")
    cases_path = ROOT / "data/cases_s1.json"
    spec = json.loads(cases_path.read_text())
    adapter_receipt = json.loads((ROOT / "data/adapter_environment_check.json").read_text())
    assert adapter_receipt["status"] == "PASS"
    assert all(importlib.metadata.version(n) == v for n,v in adapter_receipt["packages"].items())
    cpu_receipt = json.loads((ROOT / "data/cpu_contract_v2.json").read_text())
    assert cpu_receipt["exit_code"] == 0 and cpu_receipt["result"]["errors"] == cpu_receipt["result"]["failures"] == 0
    assert all(sha(ROOT / "scripts" / n) == h for n,h in cpu_receipt["code_sha256"].items())
    assert spec["n_sources"] == 8 and len(spec["cases"]) == 8
    for case in spec["cases"]:
        assert case["official_split"] == "train" and case["role"] == "nonholdout_training_pilot"
        for key, hashkey in [("image", "source_sha256"), ("user_mask", "user_mask_sha256"), ("allow_mask", "allow_mask_sha256")]:
            assert sha(ROOT / case[key]) == case[hashkey]
    receipt = Path("/workspace/model-weights/deployment_20261028.json")
    weights = json.loads(receipt.read_text())
    assert weights["phase"] == "complete"
    assert weights["models"][0]["revision"] == "ac7f9318f633fc4b5778c59367c8128225f1e3de"
    assert weights["models"][1]["revision"] == "d468c534b360e0754c6bf9e38ed49e1e09cd2e2a"
    assert all(f.get("verified") for m in weights["models"] for f in m["files"])
    assert sha(Path(LORA) / LORA_FILE) == "36ee4ff860856559b9c581eaa69414900488f3de8be2f2448586daec77e201b9"
    out.mkdir(parents=True)
    files = [Path(__file__), ROOT / "scripts/controlled_mask.py", cases_path,
             Path(inspect.getfile(QwenImageEditPipeline)), Path(inspect.getfile(FlowMatchEulerDiscreteScheduler))]
    protocol = dict(phase="FROZEN_BEFORE_GENERATION", created_unix=time.time(), purpose=spec["purpose"],
                    cases_sha256=sha(cases_path), weights_receipt_sha256=sha(receipt), scheduler_config=SHIFT3,
                    seed=2026102201, cfg=1, dtype="bfloat16", target_hw=[1024,1024], steps=8,
                    planned_main_requests=16, planned_main_dit=128, planned_extra_diagnostic_requests=2,
                    planned_extra_diagnostic_dit=16, planned_total_dit=144, extra_teachers=0, warmups=0,
                    fitting_updates=0, rl_updates=0, rgb_copy=False, timing_claim=False,
                    quality_gate=dict(mask_semantic_min=7, per_stratum_min=3, severe_mask_boundary_failures=0),
                    failure_policy="stop on first runtime or equivalence failure; retain partial ledger; no silent retries",
                    pipeline="project native plus post-scheduler binary source/noise blend; not official unmodified inpaint",
                    code_sha256={str(p):sha(p) for p in files}, test_media_read=False, target_media_read=False,
                    case_ids=[x["case_id"] for x in spec["cases"]])
    write(out / "preregistration.json", protocol)
    write(out / "cases_s1.json", spec)
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
        for model in [native.transformer, native.text_encoder, native.vae]:
            model.eval().requires_grad_(False)
        native.load_lora_weights(LORA, weight_name=LORA_FILE, adapter_name="lightning8", local_files_only=True)
        native.fuse_lora(adapter_names=["lightning8"], safe_fusing=True)
        native.unload_lora_weights()
        masked = ControlledMaskQwenImageEditPipeline(**native.components)
        masked.set_progress_bar_config(disable=True)
        assert native.vae_scale_factor == 8 and native.transformer.config.in_channels == 64
        write(out / "environment.json", dict(torch=torch.__version__, cuda=torch.version.cuda,
              user_confirmed_gpu="H800", driver_gpu=torch.cuda.get_device_name(),
              capability=list(torch.cuda.get_device_capability()), weights=weights,
              packages={n:importlib.metadata.version(n) for n in ["diffusers","transformers","huggingface_hub","peft","torchao","accelerate","safetensors","numpy","Pillow"]},
              git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              git_status=subprocess.check_output(["git","status","--short","--branch"],cwd=ROOT,text=True),
              scheduler_config=dict(native.scheduler.config),
              transformer_sha256=sha(inspect.getfile(type(native.transformer)))))

        paired_inputs = {}
        def pre_hook(module, positional, fields):
            assert active and not positional and active["calls"] < 8
            assert fields["hidden_states"].shape == (1, 8192, 64)
            assert torch.isfinite(fields["hidden_states"]).all().item()
            if active["calls"] == 0:
                assert torch.equal(fields["hidden_states"][:, :4096], active["noise"])
                active["ref_cond"] = fields["hidden_states"][:, 4096:].detach().cpu().clone()
                active["timesteps"] = native.scheduler.timesteps.cpu().tolist()
                active["sigmas"] = native.scheduler.sigmas.cpu().tolist()
                active["text_sha256"] = tensor_digest(fields["encoder_hidden_states"])
                active["text_mask_sha256"] = tensor_digest(fields["encoder_hidden_states_mask"]) if fields.get("encoder_hidden_states_mask") is not None else None
                if active["arm"] == "NATIVE_FULL":
                    paired_inputs[active["case_id"]] = (active["ref_cond"].clone(),active["text_sha256"],active["text_mask_sha256"])
                else:
                    ref,text,text_mask = paired_inputs[active["case_id"]]
                    assert torch.equal(ref,active["ref_cond"]) and text == active["text_sha256"] and text_mask == active["text_mask_sha256"]
            assert len(native.scheduler.timesteps) == 8 and len(native.scheduler.sigmas) == 9
            assert native.scheduler.timesteps.cpu().tolist() == [1000.,954.5454711914062,900.,833.3333740234375,750.,642.8570556640625,500.,300.]

        original_forward = native.transformer.forward
        def counted_forward(*positional, **fields):
            pre_hook(native.transformer,positional,fields)
            counts["invoked_dit"] += 1
            active["calls"] += 1
            append(out / "calls.jsonl", dict(event="invoked", case_id=active["case_id"], arm=active["arm"], step=active["calls"]-1, **counts))

            result = original_forward(*positional, **fields)
            counts["completed_dit"] += 1
            append(out / "calls.jsonl", dict(event="completed", case_id=active["case_id"], arm=active["arm"], step=active["calls"]-1, **counts))
            value = result[0]
            assert value.shape == (1,8192,64) and torch.isfinite(value).all().item()
            return result

        native.transformer.forward = counted_forward
        noise = torch.randn((1,4096,64), device="cuda", dtype=torch.bfloat16,
                            generator=torch.Generator(device="cuda").manual_seed(2026102201))
        torch.save(noise.cpu(), out / "explicit_noise.pt")
        diag = next(x for x in spec["cases"] if x["img_id"] == "15953")
        order = [(diag,"NATIVE_FULL"),(diag,"MASK_ALL_WHITE"),(diag,"MASK_ALL_BLACK")]
        order += [(c,a) for c in spec["cases"] for a in ["NATIVE_FULL","MASK_FULL"] if not (c is diag and a=="NATIVE_FULL")]
        native_diag = None
        for case, arm in order:
            case_dir = out / case["case_id"]; case_dir.mkdir(exist_ok=True)
            if (case_dir / (arm + ".json")).exists():
                raise RuntimeError("unexpected existing result")
            image = Image.open(ROOT / case["image"]).convert("RGB")
            mask = Image.open(ROOT / case["allow_mask"]).convert("L")
            active.clear(); active.update(case_id=case["case_id"], arm=arm, calls=0, noise=noise, states=[])
            status("running", case_id=case["case_id"], arm=arm)
            started = time.perf_counter()
            if arm == "NATIVE_FULL":
                def trace(pipe,index,t,values):
                    active["states"].append(values["latents"].detach().cpu().clone())
                    return {}
                result = native(image=image, prompt=case["prompt"], true_cfg_scale=1,
                                num_inference_steps=8, height=1024, width=1024, latents=noise.clone(),
                                output_type="latent", callback_on_step_end=trace)
                record = None
            else:
                if arm in ["MASK_ALL_WHITE","MASK_ALL_BLACK"]:
                    mask = Image.new("L", image.size, 255 if arm=="MASK_ALL_WHITE" else 0)
                result = masked(image=image, prompt=case["prompt"], mask_image=mask,
                                initial_noise=noise, true_cfg_scale=1, num_inference_steps=8,
                                output_type="latent", trace_latents=True)
                record = masked.last_mask_run
                active["states"] = record.step_latents
                torch.save(dict(source=record.source.cpu(), packed_mask=record.packed_mask.cpu(),
                                pixel_mask=record.pixel_mask.cpu(), latent_mask=record.latent_mask.cpu(),
                                active_ids=record.active_target_ids.cpu()), case_dir / (arm + "_control.pt"))
            final = result.images
            assert active["calls"] == 8 and len(active["states"]) == 8
            diagnostics = {}
            if arm == "NATIVE_FULL" and case is diag:
                native_diag = dict(states=[x.clone() for x in active["states"]], ref=active["ref_cond"].clone())
            elif arm == "MASK_ALL_WHITE":
                diagnostics["native_stepwise_bitwise_equal"] = all(torch.equal(a,b) for a,b in zip(active["states"],native_diag["states"]))
                diagnostics["native_condition_bitwise_equal"] = torch.equal(active["ref_cond"],native_diag["ref"])
                assert all(diagnostics.values()), diagnostics
            elif arm == "MASK_ALL_BLACK":
                diagnostics["final_bitwise_source"] = torch.equal(final,record.source)
                assert diagnostics["final_bitwise_source"]
            decode(native, final).save(case_dir / (arm + ".png"))
            if arm == "NATIVE_FULL":
                decode(native,active["ref_cond"].to(device="cuda")).save(case_dir / "SOURCE_RECON.png")
            torch.save(dict(final=final.cpu(), states=active["states"], ref_cond=active["ref_cond"]), case_dir / (arm + ".pt"))
            counts["successful_requests"] += 1
            row = dict(case_id=case["case_id"], arm=arm, calls=8, diagnostic_seconds=time.perf_counter()-started,
                       final_sha256=tensor_digest(final), timesteps=active["timesteps"], sigmas=active["sigmas"],
                       condition_sha256=tensor_digest(active["ref_cond"]),text_sha256=active["text_sha256"],
                       text_mask_sha256=active["text_mask_sha256"],
                       diagnostic_checks=diagnostics, **counts)
            if record is not None:
                row["mask_record"] = record.manifest()
            write(case_dir / (arm + ".json"),row)
            append(out / "runs.jsonl",row)
            status("running",case_id=case["case_id"],arm=arm)
        native.transformer.forward = original_forward
        assert counts == dict(invoked_dit=144,completed_dit=144,successful_requests=18)
        status("generation_complete_quality_pending")
    except BaseException as exc:
        if "original_forward" in locals():
            native.transformer.forward = original_forward
        status("failed", error=type(exc).__name__+": "+str(exc), active_case=active.get("case_id"), active_arm=active.get("arm"))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
