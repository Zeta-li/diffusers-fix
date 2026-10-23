import contextlib
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from run_pilot import (
    BASE,
    CONFIGS,
    ROOT,
    SEED,
    SHIFT3,
    FlowMatchEulerDiscreteScheduler,
    Image,
    QwenImageEditPipeline,
    Recorder,
    calculate_dimensions,
    diffusers,
    tensor_hash,
    torch,
    write_json,
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions():
    result = {}
    for name in ["torch", "transformers", "huggingface_hub", "peft", "accelerate", "safetensors", "torchao", "Pillow"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def run(record, json_path, output_path):
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = next(item for item in CONFIGS if item["id"] == "C_base8_cfg1_shift3")
    assert config == dict(id="C_base8_cfg1_shift3", steps=8, cfg=1.0, scheduler="shift3", lora="none")
    rows_path = ROOT / "data" / "runs.jsonl"
    matches = [
        json.loads(line)
        for line in rows_path.read_text().splitlines()
        if line.strip()
    ]
    matches = [
        row for row in matches
        if row.get("status") == "complete"
        and row.get("run_id") == "C_base8_cfg1_shift3__blocks_text__measurement_0"
    ]
    assert len(matches) == 1, f"Expected one original C measurement, found {len(matches)}"
    reference = matches[0]
    assert reference["config"] == config
    assert reference["seed"] == SEED
    assert reference["dit_calls"] == 8
    for key, value in SHIFT3.items():
        assert reference["scheduler_config"][key] == value, (key, reference["scheduler_config"].get(key), value)
    cases = json.loads((ROOT / "cases.json").read_text())["cases"]
    case = next(item for item in cases if item["case_id"] == "blocks_text")
    assert cases[0]["case_id"] == case["case_id"]
    source_path = ROOT / case["image"]
    image = Image.open(source_path).convert("RGB")
    width, height, _ = calculate_dimensions(1024 * 1024, image.width / image.height)
    latent_path = ROOT / "latents" / "blocks_text.pt"
    latent = torch.load(latent_path, map_location="cpu", weights_only=True)
    latent_hash = tensor_hash(latent)
    assert latent_hash == reference["initial_latent_sha256"], "Saved initial latent differs from reference C"
    reference_output = ROOT / reference["output"]
    reference_file_hash = file_hash(reference_output)
    assert reference_file_hash == reference["output_sha256"], "Reference C PNG no longer matches its recorded SHA"
    try:
        code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(diffusers.__file__).resolve().parent, text=True).strip()
    except subprocess.CalledProcessError:
        code_commit = None
    record.update(
        config=config,
        seed=SEED,
        negative_prompt=" ",
        base=BASE,
        source_image=str(source_path.relative_to(ROOT)),
        source_image_sha256=file_hash(source_path),
        prompt=case["prompt"],
        reference_run_id=reference["run_id"],
        reference_attempt_id=reference.get("attempt_id"),
        reference_runs_jsonl_sha256=file_hash(rows_path),
        initial_latent_path=str(latent_path.relative_to(ROOT)),
        initial_latent_sha256=latent_hash,
        reference_initial_latent_sha256=reference["initial_latent_sha256"],
        initial_latent_sha256_equal=True,
        reference_output=reference["output"],
        reference_output_sha256=reference["output_sha256"],
        environment=dict(
            python=sys.version,
            executable=sys.executable,
            platform=platform.platform(),
            packages=package_versions(),
            diffusers_source=diffusers.__file__,
            diffusers_source_version=diffusers.__version__,
            code_commit=code_commit,
            dtype="bfloat16",
            batch_size=1,
            vae_tiling=True,
            cpu_offload=False,
            cross_request_condition_cache=False,
            attention_backend="native PyTorch SDPA",
            tf32=torch.backends.cuda.matmul.allow_tf32,
            gpu_name=torch.cuda.get_device_name(),
            device_capability=list(torch.cuda.get_device_capability()),
            device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            torch_flash_sdp_enabled=torch.backends.cuda.flash_sdp_enabled(),
            torch_mem_efficient_sdp_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch_math_sdp_enabled=torch.backends.cuda.math_sdp_enabled(),
        ),
    )
    write_json(json_path, record)
    print(json.dumps(dict(phase="loading_base", time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()), ensure_ascii=False), flush=True)
    load_started = time.perf_counter()
    pipe = QwenImageEditPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.transformer.set_attention_backend("native")
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    torch.cuda.synchronize()
    record["model_load_seconds"] = time.perf_counter() - load_started
    record["environment"]["attention_processors"] = sorted({type(item).__name__ for item in pipe.transformer.attn_processors.values()})
    multiple = pipe.vae_scale_factor * 2
    width, height = width // multiple * multiple, height // multiple * multiple
    assert [height, width] == reference["target_hw"]
    sequence = (height // multiple) * (width // multiple)
    assert list(latent.shape) == [1, sequence, pipe.transformer.config.in_channels]
    assert latent.dtype == torch.bfloat16
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    regenerated = torch.randn(tuple(latent.shape), generator=generator, device="cuda", dtype=torch.bfloat16)
    regenerated_hash = tensor_hash(regenerated)
    assert regenerated_hash == latent_hash, "Same-seed regenerated initial latent differs from saved latent"
    del regenerated
    initial = latent.to("cuda").clone()
    assert tensor_hash(initial) == latent_hash
    assert not any("lora_" in name for name, _ in pipe.transformer.named_parameters())
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(SHIFT3)
    recorder = Recorder()
    recorder.wrap(pipe, "encode_prompt", "prompt_encode")
    recorder.wrap(pipe, "_encode_vae_image", "vae_encode")
    recorder.wrap(pipe.transformer, "forward", "dit", shapes=True)
    recorder.wrap(pipe.vae, "decode", "vae_decode")
    recorder.wrap(pipe.scheduler, "step", "scheduler")
    record["target_hw"] = [height, width]
    record["regenerated_initial_latent_sha256"] = regenerated_hash
    write_json(json_path, record)
    print(json.dumps(dict(phase="running_single_cold_diagnostic", reference_run_id=reference["run_id"]), ensure_ascii=False), flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    recorder.active = True
    started = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            image=image,
            prompt=case["prompt"],
            negative_prompt=" ",
            true_cfg_scale=config["cfg"],
            num_inference_steps=config["steps"],
            height=height,
            width=width,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
            latents=initial,
            output_type="pil",
        ).images[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    recorder.active = False
    assert len(recorder.shapes) == 8, f"Expected 8 DiT forwards, got {len(recorder.shapes)}"
    assert json.loads(json.dumps(recorder.shapes[0])) == reference["first_cond_shapes"]
    assert pipe.scheduler.timesteps.detach().cpu().tolist() == reference["timesteps"]
    assert pipe.scheduler.sigmas.detach().cpu().tolist() == reference["sigmas"]
    result.save(output_path)
    output_hash = file_hash(output_path)
    reference_image = Image.open(reference_output).convert("RGB")
    output_rgb = result.convert("RGB")
    record.update(
        status="complete",
        completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        output=str(output_path.relative_to(ROOT)),
        output_sha256=output_hash,
        output_sha256_equal=output_hash == reference["output_sha256"],
        rgb_pixels_equal=output_rgb.size == reference_image.size and output_rgb.tobytes() == reference_image.tobytes(),
        output_wh=list(result.size),
        e2e_cold_diagnostic_seconds=elapsed,
        cuda_event_seconds=recorder.seconds(),
        dit_calls=len(recorder.shapes),
        first_cond_shapes=recorder.shapes[0],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        scheduler_config=dict(pipe.scheduler.config),
        timesteps=pipe.scheduler.timesteps.detach().cpu().tolist(),
        sigmas=pipe.scheduler.sigmas.detach().cpu().tolist(),
        initial_latent_after_request_sha256=tensor_hash(initial),
    )
    write_json(json_path, record)
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def main():
    diagnostic_dir = ROOT / "diagnostics"
    diagnostic_dir.mkdir(exist_ok=True)
    attempt_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    stem = f"C_blocks_text_environment_regression_{attempt_id}"
    json_path = diagnostic_dir / f"{stem}.json"
    output_path = diagnostic_dir / f"{stem}.png"
    log_path = diagnostic_dir / f"{stem}.log"
    assert not any(path.exists() for path in [json_path, output_path, log_path])
    record = dict(
        status="running",
        diagnostic_id=stem,
        started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        kind="environment_regression_diagnostic",
        included_in_main_protocol=False,
        timing_scope="one cold diagnostic request, model load reported separately, no warmup or performance inference",
        request_count=1,
        script_sha256=file_hash(Path(__file__)),
        run_pilot_script_sha256=file_hash(Path(__file__).with_name("run_pilot.py")),
        log=str(log_path.relative_to(ROOT)),
    )
    write_json(json_path, record)
    with log_path.open("x") as log_stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_stream)), contextlib.redirect_stderr(Tee(sys.stderr, log_stream)):
            handler = logging.StreamHandler(log_stream)
            logging.getLogger("diffusers").addHandler(handler)
            logging.getLogger("transformers").addHandler(handler)
            try:
                run(record, json_path, output_path)
            except Exception as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                write_json(json_path, record)
                traceback.print_exc()
                raise
            finally:
                logging.getLogger("diffusers").removeHandler(handler)
                logging.getLogger("transformers").removeHandler(handler)


if __name__ == "__main__":
    main()
