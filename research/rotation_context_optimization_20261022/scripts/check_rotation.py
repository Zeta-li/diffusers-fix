import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ["F", "R2", "FIX0", "FIX1", "FIX2", "ROT"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_report(path, report, no_write):
    if not no_write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


def previous_root(root, supplied, manifest):
    candidates = [supplied, root.parent / "局部执行与边界验证_20261022", root.parent / "local_executor_boundary_20261022", manifest.get("previous_root")]
    for value in candidates:
        if value is not None:
            path = Path(value).resolve()
            if (path / "masks/regions.json").exists():
                return path
    raise FileNotFoundError("Pass --previous-root pointing to the frozen local-executor experiment")


def index_list(values, count, sorted_required=True):
    assert isinstance(values, list) and all(type(value) is int for value in values)
    assert len(values) == len(set(values)) and all(0 <= value < count for value in values)
    if sorted_required:
        assert values == sorted(values)
    return set(values)


def reconstruct_order(height, width, core, seed, case_id):
    tiles = []
    for tile_row in range((height + 1) // 2):
        for tile_column in range((width + 1) // 2):
            values = [row * width + column for row in range(tile_row * 2, min(height, tile_row * 2 + 2))
                      for column in range(tile_column * 2, min(width, tile_column * 2 + 2))
                      if row * width + column not in core]
            if values:
                key = hashlib.sha256(f"{seed}:{case_id}:{tile_row}:{tile_column}".encode()).hexdigest()
                tiles.append((key, tile_row, tile_column, values))
    tiles.sort(key=lambda value: value[:3])
    return tiles, [index for _, _, _, values in tiles for index in values]


def manifest_checks(root, plan, previous):
    source_path = previous / "masks/regions.json"
    assert digest(source_path) == plan["source_regions_sha256"]
    source = json.loads(source_path.read_text())
    assert plan["configs"] == CONFIGS and plan["full_steps_1based"] == [1, 8]
    assert plan["partial_steps_1based"] == [2, 3, 4, 5, 6, 7]
    assert plan["warmups_per_case_config"] == 1 and plan["measurement_repeats"] == 3
    assert plan["planned_requests"] == 72 and plan["planned_dit_forwards"] == 576
    assert set(plan["cases"]) == set(source["cases"]) == {"blocks_text", "shirt_color", "remove_hair"}
    results = {}
    for case_id, case in plan["cases"].items():
        old = source["cases"][case_id]
        height, width = case["target_grid_hw"]
        n = height * width
        assert n == case["n_target_tokens"] == old["n_target_tokens"]
        assert case["target_grid_hw"] == old["target_grid_hw"] and case["target_hw"] == old["target_hw"]
        core = index_list(case["core_indices"], n)
        assert case["core_indices"] == old["regions"]["R2"]["indices"]
        assert digest(previous / old["regions"]["R2"]["mask_npy"]) == case["source_R2_mask_sha256"]
        assert digest(previous / case["source_image_previous_relative"]) == case["source_image_sha256"]
        extra = index_list(case["always_fresh_remainder_indices"], n, sorted_required=False)
        groups = [index_list(case["groups"][f"G{index}"]["indices"], n) for index in range(3)]
        q, remainder_count = divmod(n - len(core), 3)
        assert len(extra) == remainder_count == case["n_remainder_tokens"]
        assert all(len(group) == q == case["groups"][f"G{index}"]["n_tokens"] for index, group in enumerate(groups))
        assert q == case["n_tokens_per_group"] and len(core) == case["n_core_tokens"]
        partitions = [core, extra] + groups
        assert sum(len(value) for value in partitions) == n
        assert set.union(*partitions) == set(range(n))
        assert all(not (a & b) for index, a in enumerate(partitions) for b in partitions[index + 1:])
        tiles, ordered = reconstruct_order(height, width, core, plan["seed"], case_id)
        assert ordered == case["background_order_indices"]
        assert hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest() == case["background_order_sha256"]
        assert ordered[:remainder_count] == case["always_fresh_remainder_indices"]
        remainder_order = ordered[remainder_count:]
        assert all(sorted(remainder_order[index * q:(index + 1) * q]) == case["groups"][f"G{index}"]["indices"] for index in range(3))
        owner = {token: label for label, values in [("E", extra)] + [(f"G{i}", group) for i, group in enumerate(groups)] for token in values}
        splits = []
        for _, tile_row, tile_column, values in tiles:
            assignments = {}
            for token in values:
                assignments.setdefault(owner[token], []).append(token)
            if len(assignments) > 1:
                splits.append(dict(tile_row=tile_row, tile_column=tile_column, assignment_indices=assignments))
        assert splits == case["tile_boundary_splits"] and len(splits) == case["tile_boundary_split_count"]
        assert len(core) + len(extra) + q == case["n_fixed_or_rotating_partial_tokens"]
        schedules = {}
        for config in CONFIGS:
            settings = case["configs"][config]
            assert settings["native"] == (config == "F")
            expected_full = list(range(1, 9)) if config == "F" else [1, 8]
            assert settings["full_steps_1based"] == expected_full
            assert set(settings["per_step_indices"]) == {str(step) for step in range(8)}
            last = [-1] * n
            audit_rows = []
            for step in range(8):
                expected = set(range(n)) if step in [0, 7] or config == "F" else core.copy()
                if step not in [0, 7] and config.startswith("FIX"):
                    expected |= extra | groups[int(config[-1])]
                if step not in [0, 7] and config == "ROT":
                    expected |= extra | groups[(step - 1) % 3]
                chosen = settings["per_step_indices"][str(step)]
                assert index_list(chosen, n) == expected
                before = [step - value if value >= 0 else -1 for value in last]
                for token in chosen:
                    last[token] = step
                after = [step - value for value in last]
                recorded = settings["per_step_audit"][step]
                assert recorded["step_1based"] == step + 1 and recorded["selected_target_count"] == len(chosen)
                assert recorded["max_age_at_attention_read"] == max(after)
                assert recorded["age_at_attention_read_histogram"] == {str(key): value for key, value in sorted(Counter(after).items())}
                assert recorded["age_before_update_histogram"] == {str(key): value for key, value in sorted(Counter(before).items())}
                if config == "ROT":
                    assert max(after) <= 2
                if config == "F" or step in [0, 7]:
                    assert max(after) == 0
                audit_rows.append(dict(step=step + 1, selected_target_count=len(chosen), max_read_age=max(after)))
            schedules[config] = audit_rows
            default_group = groups[0] if config == "ROT" else groups[int(config[-1])] if config.startswith("FIX") else set()
            default = set(range(n)) if config == "F" else core | extra | default_group if config != "R2" else core
            assert index_list(settings["indices"], n) == default
            assert settings["n_partial_target_tokens"] == len(default)
            assert settings["target_token_step_budget"] == sum(row["selected_target_count"] for row in audit_rows)
        for step in range(1, 7):
            assert len({schedules[config][step]["selected_target_count"] for config in ["FIX0", "FIX1", "FIX2", "ROT"]}) == 1
        assert all(sum(groups[index] <= set(case["configs"]["ROT"]["per_step_indices"][str(step)]) for step in range(1, 7)) == 2 for index in range(3))
        results[case_id] = dict(status="pass", target_tokens=n, core_tokens=len(core), E=len(extra), group_tokens=q,
            partial_budget=len(core) + len(extra) + q, independently_reconstructed_SHA_tile_order=True,
            coverage_and_disjointness=True, exact_FIX_ROT_per_step_budget=True, all_groups_refreshed_twice=True,
            tile_boundary_split_count=len(splits), ROT_max_read_age=max(row["max_read_age"] for row in schedules["ROT"]),
            config_schedules=schedules)
    return results


def load_fixture(root, previous):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    executor_path = root / "scripts/local_executor.py"
    assert executor_path.exists(), "Place the exact execution copy of local_executor.py beside this checker"
    spec = importlib.util.spec_from_file_location("local_executor", executor_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_executor"] = module
    spec.loader.exec_module(module)
    fixture_path = previous / "scripts/check_tiny_executor.py"
    spec = importlib.util.spec_from_file_location("rotation_fixture_helpers", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    assert fixture.LocalTargetExecutor is module.LocalTargetExecutor
    return fixture, executor_path, fixture_path


def tiny_rotation(fixture, dtype, counters):
    import torch
    from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
    model, executor, raw, text, text_mask = fixture.fixture(dtype)
    n, c = fixture.TARGETS, fixture.CONDITIONS
    assert n == 6
    core, extra, groups = {1}, {0, 5}, [{2}, {3}, {4}]
    selections = {step: list(range(n)) if step in [0, 7] else sorted(core | extra | groups[(step - 1) % 3]) for step in range(8)}
    executor.begin(n, c, selections[1], [0, 7], per_step_indices=selections)
    audit = fixture.ModuleAudit(model)
    flow = fixture.scheduler()
    latent, condition = raw[:, :n].clone(), raw[:, n:].clone()
    expected_last = torch.full((n,), -1, dtype=torch.long)
    records = []
    try:
        for step, timestep in enumerate(flow.timesteps):
            selected = torch.tensor(selections[step], dtype=torch.long)
            inactive = torch.tensor([index for index in range(n) if index not in selections[step]], dtype=torch.long)
            image_indices = torch.cat([selected, torch.arange(n, n + c)])
            current = torch.cat([latent, condition], dim=1)
            expected_input = current.index_select(1, image_indices).clone()
            old_velocity = executor.velocity.clone() if executor.velocity is not None else None
            old_cache = {index: (key.clone(), value.clone()) for index, (key, value) in executor.layer_cache.items()}
            old_age = expected_last.clone()
            kwargs = fixture.inputs(current, text, text_mask, float(timestep / 1000))
            audit.reset()
            audit.active = True
            counters["tiny_local_forwards"] += 1
            output = model(**kwargs)[0].clone()
            audit.active = False
            assert len(audit.raw_inputs) == 1 and torch.equal(audit.raw_inputs[0], expected_input)
            for row in audit.rows:
                expected_tokens = fixture.TEXT if row["name"].startswith("text_") else len(selected) + c
                assert row["shape"][1] == expected_tokens, row
            if inactive.numel():
                assert torch.equal(output.index_select(1, inactive), old_velocity.index_select(1, inactive))
                for index, (key, value) in executor.layer_cache.items():
                    assert torch.equal(key.index_select(1, inactive), old_cache[index][0].index_select(1, inactive))
                    assert torch.equal(value.index_select(1, inactive), old_cache[index][1].index_select(1, inactive))
            block = model.transformer_blocks[0]
            projected = model.img_in(expected_input)
            temb = model.time_text_embed(kwargs["timestep"].to(projected.dtype), projected, None)
            modulation, _ = block.img_mod(temb).chunk(2, dim=-1)
            normalized, _ = block._modulate(block.img_norm1(projected), modulation)
            expected_key = block.attn.to_k(normalized).unflatten(-1, (-1, block.attn.inner_dim // block.attn.heads))
            if block.attn.norm_k is not None:
                expected_key = block.attn.norm_k(expected_key)
            rope = model.pos_embed(fixture.SHAPES, max_txt_seq_len=fixture.TEXT, device="cpu")[0].index_select(0, image_indices)
            expected_key = apply_rotary_emb_qwen(expected_key, rope, use_real=False)
            expected_value = block.attn.to_v(normalized).unflatten(-1, (-1, block.attn.inner_dim // block.attn.heads))
            assert torch.equal(executor.layer_cache[0][0].index_select(1, selected), expected_key[:, :len(selected)])
            assert torch.equal(executor.layer_cache[0][1].index_select(1, selected), expected_value[:, :len(selected)])
            expected_last.index_fill_(0, selected, step)
            assert torch.equal(executor.age, expected_last)
            read_age = step - executor.age
            assert int(read_age.max()) <= 2
            traces = [row for row in executor.trace if row["step"] == step + 1]
            reactivated = int(((old_age.index_select(0, selected) < step - 1) & (old_age.index_select(0, selected) >= 0)).sum()) if step else 0
            assert traces and all(row["newly_activated_from_current_latent"] == reactivated for row in traces)
            full_exact = None
            if step in [0, 7]:
                counters["tiny_native_control_forwards"] += 1
                dense = executor.original_forward(**kwargs)[0].clone()
                full_exact = bool(torch.equal(output, dense))
                assert full_exact
            expected_next = (latent.float() + (flow.sigmas[step + 1] - flow.sigmas[step]) * output[:, :n]).to(output.dtype)
            latent = flow.step(output[:, :n], timestep, latent, return_dict=False)[0]
            assert torch.equal(latent, expected_next)
            records.append(dict(step=step + 1, selected=selected.tolist(), reactivated=reactivated,
                gathered_current_latent_exact=True, inactive_cache_and_velocity_unchanged=True,
                first_layer_current_KV_exact=True, max_read_age=int(read_age.max()), full_output_bitwise_equal=full_exact,
                actual_Euler_bitwise_equal=True))
    finally:
        audit.close()
    assert executor.step == 8 and all(row["max_read_age"] <= 2 for row in records)
    executor.clear()
    assert not executor.layer_cache and executor.velocity is None
    return dict(status="pass", device="cpu", dtype=str(dtype), target_tokens=n, E=sorted(extra),
        core=sorted(core), groups=[sorted(group) for group in groups], steps=records, request_reset=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--previous-root", type=Path)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    plan_path = root / "masks/rotation_plan.json"
    report_path = root / "data/rotation_correctness.json"
    started = time.perf_counter()
    report = dict(status="running", started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        gpu_requests=0, real_20B_forwards=0, tiny_local_forwards=0, tiny_native_control_forwards=0,
        tiny_executed=False, tests=[], manifest_sha256=digest(plan_path), test_script_sha256=digest(Path(__file__)))
    try:
        plan = json.loads(plan_path.read_text())
        previous = previous_root(root, args.previous_root, plan)
        report["previous_root"] = str(previous)
        report["manifest_cases"] = manifest_checks(root, plan, previous)
        report["tests"].append(dict(name="independent_manifest_and_age_audit", status="pass"))
        if not args.manifest_only:
            import torch
            torch.set_num_threads(2)
            fixture, executor_path, fixture_path = load_fixture(root, previous)
            report.update(executor_sha256=digest(executor_path), borrowed_fixture_sha256=digest(fixture_path),
                tiny_executed=True, torch_version=torch.__version__, device="cpu")
            with torch.inference_mode():
                for dtype, name in [(torch.float32, "fp32_rotation_dynamic"), (torch.bfloat16, "bf16_rotation_dynamic")]:
                    result = tiny_rotation(fixture, dtype, report)
                    report["tests"].append(dict(name=name, status="pass", result=result))
            assert report["tiny_local_forwards"] == 16 and report["tiny_native_control_forwards"] == 4
            report["status"] = "pass"
        else:
            report["status"] = "manifest_pass_tiny_not_run"
    except Exception as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report.update(completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), elapsed_seconds=time.perf_counter() - started,
            actual_tiny_model_forwards=report["tiny_local_forwards"] + report["tiny_native_control_forwards"])
        write_report(report_path, report, args.no_write)
        print(json.dumps({key: report.get(key) for key in ["status", "tiny_executed", "gpu_requests", "real_20B_forwards", "actual_tiny_model_forwards", "error"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
