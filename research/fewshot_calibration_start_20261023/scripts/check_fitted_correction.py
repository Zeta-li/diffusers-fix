import argparse
import copy
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from value_fitted_correction import ValueFittedExecutor, coefficient_payload_hash


CONFIG = dict(patch_size=2, in_channels=8, out_channels=2, num_layers=2,
              attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
              axes_dims_rope=(4, 6, 6))
SHAPES = [[(1, 2, 3), (1, 2, 2)]]
SELECTIONS = {0: list(range(6)), 1: [4, 1], 2: [1], 3: [4, 1],
              4: [0, 3], 5: [3], 6: [2, 5], 7: list(range(6))}
IDENTITY = "synthetic-tiny-seed1873-not-a-real-calibrated-model"
CALLS = dict(local=0, dense_reference=0)


def flow():
    scheduler = FlowMatchEulerDiscreteScheduler(base_shift=math.log(3), max_shift=math.log(3),
        use_dynamic_shifting=True, time_shift_type="exponential", shift_terminal=None)
    scheduler.set_timesteps(sigmas=[1 - step / 8 for step in range(8)], device="cpu", mu=math.log(3))
    return scheduler


def fixture(dtype=torch.float32):
    torch.manual_seed(1873)
    model = QwenImageTransformer2DModel(**CONFIG).to(device="cpu", dtype=dtype).eval()
    model.set_attention_backend("native")
    executor = ValueFittedExecutor(model)
    scheduler = flow()
    generator = torch.Generator(device="cpu").manual_seed(5817)
    raw = torch.randn((1, 10, 8), generator=generator).to(dtype=dtype)
    text = torch.randn((1, 5, 24), generator=generator).to(dtype=dtype)
    mask = torch.tensor([[True, False, True, True, False]])
    return model, executor, scheduler, raw, text, mask


def artifact_for(executor, timesteps, identity=False):
    contract = executor.coefficient_contract(timesteps, model_identity=IDENTITY)
    shape = (contract["layers"], 8, 8, contract["heads"], contract["head_dim"])
    alpha = torch.full(shape, 1.2, dtype=torch.float32)
    alpha[..., 0] = -0.5
    delta = torch.linspace(-0.1, 0.2, math.prod(shape)).reshape(shape)
    observed = torch.tril(torch.ones((8, 8), dtype=torch.bool), diagonal=-1)
    observed[4, 0] = False
    if identity:
        observed.zero_()
    return dict(alpha=alpha, delta=delta, observed_pairs=observed, contract=contract,
        source="Synthetic CPU correctness fixture, not Train8 calibration",
        source_hash=hashlib.sha256(b"synthetic-fitted-cpu-v1").hexdigest(),
        training_image_ids=[f"synthetic-{index}" for index in range(8)])


def rejected(function):
    try:
        function()
    except (ValueError, RuntimeError, TypeError) as error:
        return str(error)
    raise AssertionError("Invalid fitted correction artifact or state was accepted")


def table_checks():
    _, executor, scheduler, _, _, _ = fixture()
    artifact = artifact_for(executor, scheduler.timesteps)
    before = [artifact[name].clone() for name in ["alpha", "delta", "observed_pairs"]]
    metadata = executor.prepare_fitted_tables(scheduler.timesteps, artifact, model_identity=IDENTITY)
    assert metadata["route"] == "B_fitted_raw_value_affine"
    assert metadata["calibration_performed_here"] is False and metadata["sample_count_verified"] is False
    assert metadata["unseen_pair_fallback"] == "identity"
    assert metadata["declared_training_image_count"] == 8
    diagonal = torch.arange(8)
    assert torch.equal(executor.alpha[:, diagonal, diagonal], torch.ones((2, 8, 2, 16)))
    assert torch.count_nonzero(executor.delta[:, diagonal, diagonal]) == 0
    unseen = ~executor.observed_pairs
    assert torch.equal(executor.alpha[unseen], torch.ones_like(executor.alpha[unseen]))
    assert torch.count_nonzero(executor.delta[unseen]) == 0
    assert float(executor.alpha[:, 1, 0, :, 0].max()) < 0
    for name, value in zip(["alpha", "delta", "observed_pairs"], before):
        assert torch.equal(artifact[name], value)
    assert executor.assert_schedule(scheduler.timesteps, model_identity=IDENTITY)
    json.dumps(metadata)
    digest = metadata["effective_coefficient_sha256"]
    ptrs = [executor.alpha.data_ptr(), executor.delta.data_ptr()]
    executor.clear()
    assert ptrs == [executor.alpha.data_ptr(), executor.delta.data_ptr()]
    layered = copy.deepcopy(artifact)
    layered["observed_pairs"] = artifact["observed_pairs"][None].expand(2, -1, -1).clone()
    second = executor.prepare_tables(scheduler.timesteps, layered, model_identity=IDENTITY)
    assert second["effective_coefficient_sha256"] == digest
    layered["coefficient_sha256"] = coefficient_payload_hash(layered["alpha"], layered["delta"], layered["observed_pairs"])
    executor.prepare_fitted_tables(scheduler.timesteps, layered, model_identity=IDENTITY)
    return dict(metadata=metadata, no_clipping=True, diagonal_forced_identity=True,
                unseen_forced_identity=True, two_dimensional_mask_broadcast_exact=True,
                input_artifact_unchanged=True, content_digest_checked=True, tables_persist_after_clear=True)


def guard_checks():
    model, executor, scheduler, _, _, _ = fixture()
    artifact = artifact_for(executor, scheduler.timesteps)
    executor.prepare_fitted_tables(scheduler.timesteps, artifact, model_identity=IDENTITY)
    digest = executor.table_metadata["effective_coefficient_sha256"]
    invalid = []
    for label in ["missing_delta", "wrong_alpha_shape", "wrong_delta_shape", "wrong_contract", "wrong_dtype",
                  "nan_unseen_alpha", "infinite_delta", "float_mask", "wrong_mask_shape", "future_anchor",
                  "duplicate_images", "missing_images", "bad_source_hash", "bad_coefficient_hash"]:
        value = copy.deepcopy(artifact)
        if label == "missing_delta":
            del value["delta"]
        elif label == "wrong_alpha_shape":
            value["alpha"] = value["alpha"][:, :, :, :, :-1]
        elif label == "wrong_delta_shape":
            value["delta"] = value["delta"][:1]
        elif label == "wrong_contract":
            value["contract"]["model_identity"] = "different-weight-contract"
        elif label == "wrong_dtype":
            value["contract"]["model_dtype"] = "torch.bfloat16"
        elif label == "nan_unseen_alpha":
            value["alpha"][0, 0, 7, 0, 0] = float("nan")
        elif label == "infinite_delta":
            value["delta"][0, 1, 0, 0, 0] = float("inf")
        elif label == "float_mask":
            value["observed_pairs"] = value["observed_pairs"].float()
        elif label == "wrong_mask_shape":
            value["observed_pairs"] = torch.zeros((1, 8, 8), dtype=torch.bool)
        elif label == "future_anchor":
            value["observed_pairs"][0, 7] = True
        elif label == "duplicate_images":
            value["training_image_ids"] = ["same", "same"]
        elif label == "missing_images":
            value["training_image_ids"] = []
        elif label == "bad_source_hash":
            value["source_hash"] = "not-a-digest"
        elif label == "bad_coefficient_hash":
            value["coefficient_sha256"] = "0" * 64
        invalid.append(dict(case=label, error=rejected(lambda value=value: executor.prepare_fitted_tables(
            scheduler.timesteps, value, model_identity=IDENTITY))))
        assert executor.table_metadata["effective_coefficient_sha256"] == digest
    invalid.append(dict(case="statistics_api_is_not_B", error=rejected(lambda: executor.prepare_tables(
        scheduler.timesteps, {"mean": torch.zeros((2, 32)), "variance": torch.ones((2, 32))}, model_identity=IDENTITY))))
    changed_time = scheduler.timesteps.clone()
    changed_time[2] -= 1
    invalid.append(dict(case="changed_schedule", error=rejected(lambda: executor.assert_schedule(changed_time, model_identity=IDENTITY))))
    invalid.append(dict(case="changed_model_identity", error=rejected(lambda: executor.assert_schedule(scheduler.timesteps, model_identity="other"))))
    executor.mode = "affine"
    with torch.no_grad():
        model.proj_out.weight.add_(0.01)
    invalid.append(dict(case="parameter_mutation", error=rejected(lambda: executor.begin(6, 4, [1, 4], [0, 7]))))
    _, empty, _, _, _, _ = fixture()
    empty.mode = "affine"
    invalid.append(dict(case="unprepared_begin", error=rejected(lambda: empty.begin(6, 4, [1, 4], [0, 7]))))
    return dict(invalid=invalid, rejected_count=len(invalid), failed_loads_preserve_previous_tables=True)


def cache_checks(dtype):
    _, executor, scheduler, _, _, _ = fixture(dtype)
    executor.prepare_fitted_tables(scheduler.timesteps, artifact_for(executor, scheduler.timesteps), model_identity=IDENTITY)
    executor.mode = "affine"
    executor.begin(6, 4, [4, 1], [0, 7], per_step_indices=SELECTIONS)
    generator = torch.Generator().manual_seed(5)
    cv = torch.randn((1, 6, 2, 16), generator=generator).to(dtype)
    ck = torch.randn((1, 6, 2, 16), generator=generator).to(dtype)
    executor.layer_cache[0] = (ck, cv)
    executor.age[:] = torch.tensor([0, 2, 0, 1, 0, 0])
    executor.step = 3
    selected = executor.indices[3]
    before_v, before_k, before_age = cv.clone(), ck.clone(), executor.age.clone()
    actual = executor.read_value_cache(0, cv, selected)
    effective = before_age.clone()
    effective[selected.cpu()] = 3
    expected = cv.clone()
    for index, anchor in enumerate(effective.tolist()):
        expected[:, index] = (cv[:, index].float() * executor.alpha[0, 3, anchor]
                             + executor.delta[0, 3, anchor]).to(dtype)
    assert torch.equal(actual, expected)
    assert torch.equal(cv, before_v) and torch.equal(ck, before_k) and torch.equal(executor.age, before_age)
    assert torch.equal(actual.index_select(1, selected), cv.index_select(1, selected))
    assert actual.data_ptr() != cv.data_ptr()
    executor.step = 4
    executor.age.zero_()
    executor._value_read_step = None
    selected = executor.indices[4]
    unseen = executor.read_value_cache(0, cv, selected)
    assert torch.equal(unseen, cv)
    assert torch.count_nonzero(executor.age) == 0
    replacing = rejected(lambda: executor.prepare_fitted_tables(scheduler.timesteps,
        artifact_for(executor, scheduler.timesteps), model_identity=IDENTITY))
    executor.mode = "none"
    mode_error = rejected(lambda: executor.read_value_cache(0, cv, selected))
    return dict(dtype=str(dtype), per_token_formula_exact=True, raw_kv_and_age_unchanged=True,
        selected_value_exact=True, unseen_pair_identity_exact=True,
        replacing_tables_mid_request_rejected=replacing, changing_mode_rejected=mode_error)


def run_trajectory(dtype, label, full=False):
    model, executor, scheduler, raw, text, mask = fixture(dtype)
    artifact = artifact_for(executor, scheduler.timesteps, identity=label == "identity")
    executor.prepare_fitted_tables(scheduler.timesteps, artifact, model_identity=IDENTITY)
    executor.mode = "none" if label == "none" else "affine"
    executor.install()
    executor.begin(6, 4, [4, 1], list(range(8)) if full else [0, 7], per_step_indices=SELECTIONS)
    latent, condition = raw[:, :6].clone(), raw[:, 6:].clone()
    outputs, audits = [], []
    original_read = executor.read_value_cache
    def audit(layer, cv, selected):
        before_v, before_k, before_age = cv.clone(), executor.layer_cache[layer][0].clone(), executor.age.clone()
        value = original_read(layer, cv, selected)
        assert torch.equal(cv, before_v) and torch.equal(executor.layer_cache[layer][0], before_k)
        assert torch.equal(executor.age, before_age)
        assert torch.equal(value.index_select(1, selected), cv.index_select(1, selected))
        audits.append((executor.step, layer))
        return value
    executor.read_value_cache = audit
    with torch.inference_mode():
        for step, timestep in enumerate(scheduler.timesteps):
            kwargs = dict(hidden_states=torch.cat([latent, condition], dim=1), encoder_hidden_states=text,
                encoder_hidden_states_mask=mask, timestep=timestep.expand(1).to(dtype=dtype) / 1000,
                img_shapes=SHAPES, return_dict=False)
            if step == 1 and label == "fitted" and not full:
                wrong = dict(kwargs)
                wrong["timestep"] = kwargs["timestep"] + 0.125
                rejected(lambda: model(**wrong))
                assert executor.step == 1
            output = model(**kwargs)[0].clone()
            CALLS["local"] += 1
            if full or step in [0, 7]:
                dense = executor.original_forward(**kwargs)[0]
                CALLS["dense_reference"] += 1
                assert torch.equal(output, dense)
            assert bool(torch.isfinite(output).all())
            outputs.append(output)
            latent = scheduler.step(output[:, :6], timestep, latent, return_dict=False)[0]
    assert executor.step == 8 and len(audits) == 16
    assert bool(torch.isfinite(latent).all())
    if not full:
        assert any(row["selected_target"] < 6 for row in executor.trace)
        assert any(row["newly_activated_from_current_latent"] > 0 for row in executor.trace)
    return outputs, dict(label=label, dtype=str(dtype), full=full, steps=8,
        layer_read_audits=len(audits), cache_and_anchors_unchanged_on_read=True,
        full_steps_dense_exact=True, finite=True)


def trajectory_checks(dtype):
    none, a = run_trajectory(dtype, "none")
    identity, b = run_trajectory(dtype, "identity")
    fitted, c = run_trajectory(dtype, "fitted")
    _, d = run_trajectory(dtype, "fitted", full=True)
    assert all(torch.equal(x, y) for x, y in zip(none, identity))
    assert not torch.equal(fitted[1], identity[1])
    return dict(dtype=str(dtype), identity_equals_uncorrected_all_steps=True,
                fitted_correction_changes_partial_output=True, trajectories=[a, b, c, d])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtypes", nargs="+", choices=["float32", "bfloat16"], default=["float32", "bfloat16"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = dict(status="running", device="cpu", cuda_used=False, synthetic_only=True,
        real_20b_model_forwards=0, real_train8_images_used=0, training_performed=False, tests=[])
    try:
        result["tests"].append(dict(name="contract_tables_and_identity_fallback", result=table_checks()))
        result["tests"].append(dict(name="invalid_artifact_and_lifecycle_guards", result=guard_checks()))
        for name in args.dtypes:
            dtype = getattr(torch, name)
            result["tests"].append(dict(name=f"{name}_raw_cache_read", result=cache_checks(dtype)))
            result["tests"].append(dict(name=f"{name}_eight_step_trajectory", result=trajectory_checks(dtype)))
        result["status"] = "pass"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    result["tiny_model_forward_counts"] = CALLS
    result["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")}
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
