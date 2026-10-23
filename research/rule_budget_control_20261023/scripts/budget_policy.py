from __future__ import annotations

import hashlib
from collections import Counter
from fractions import Fraction

SEED = 2026102301
CONFIGS = ("F", "R2", "B125", "B250", "B333", "B500")
BUDGETS = {"F": None, "R2": Fraction(0), "B125": Fraction(1, 8),
           "B250": Fraction(1, 4), "B333": Fraction(1, 3), "B500": Fraction(1, 2)}
FULL_STEPS = (0, 7)
N_STEPS = 8


def encode_case(case_id, source_case, seed=SEED):
    h, w = map(int, source_case["target_grid_hw"])
    if h <= 0 or w <= 0:
        raise ValueError("target grid dimensions must be positive")
    n = h * w
    if int(source_case.get("n_target_tokens", n)) != n:
        raise ValueError("target token count does not match grid")
    raw = list(source_case["core_indices"])
    core = sorted(set(raw))
    if len(core) != len(raw) or any(type(i) is not int or not 0 <= i < n for i in core):
        raise ValueError("core must contain unique in-range integer indices")
    core_set = set(core)
    blocks = []
    for row in range(0, h, 2):
        for col in range(0, w, 2):
            indices = [r * w + c for r in range(row, min(row + 2, h))
                       for c in range(col, min(col + 2, w)) if r * w + c not in core_set]
            if not indices:
                continue
            
            tr, tc = row // 2, col // 2
            digest = hashlib.sha256(f"{seed}:{case_id}:{tr}:{tc}".encode()).hexdigest()
            blocks.append({"block_id": f"{tr}:{tc}", "tile_row": tr, "tile_col": tc,
                           "indices": indices, "n_tokens": len(indices), "sha256": digest})
    blocks.sort(key=lambda b: (b["sha256"], b["tile_row"], b["tile_col"]))
    result = {"case_id": case_id, "seed": seed, "target_grid_hw": [h, w],
              "target_hw": source_case.get("target_hw"), "n_target_tokens": n,
              "core_indices": core, "n_core_tokens": len(core),
              "n_background_tokens": n - len(core), "blocks": blocks,
              "block_definition": "noncore subset of global 2x2 target-token tile; indivisible"}
    for key in ("source_image_previous_relative", "source_image_sha256", "source_R2_mask_sha256"):
        if key in source_case:
            result[key] = source_case[key]
    return result


def validate_target_fraction(case_info, rho):
    rho = Fraction(str(rho)) if not isinstance(rho, Fraction) else rho
    if rho < Fraction(case_info["n_core_tokens"], case_info["n_target_tokens"]) or rho > 1:
        raise ValueError("total-target fraction must preserve the entire core and be <= 1")
    return rho


def build_policy(case_info, config):
    if config not in BUDGETS:
        raise ValueError(f"unknown config: {config}")
    n = int(case_info["n_target_tokens"])
    core = list(case_info["core_indices"])
    nbg = int(case_info["n_background_tokens"])
    blocks = case_info["blocks"]
    budget = BUDGETS[config]
    nominal_cap = nbg if budget is None else nbg * budget.numerator // budget.denominator
    all_indices = list(range(n))
    last = {b["block_id"]: -1 for b in blocks}
    per_step, audits = {}, []
    for step in range(N_STEPS):
        full = config == "F" or step in FULL_STEPS
        before = []
        for block in blocks:
            before.extend([step - last[block["block_id"]] if last[block["block_id"]] >= 0 else -1]
                          * block["n_tokens"])
        before += ([step - (step - 1) if step else -1] * len(core))
        if full:
            chosen = list(blocks)
            selected = all_indices.copy()
            cap = nbg
        else:
            remaining, chosen = nominal_cap, []
            for block in sorted(blocks, key=lambda b: (last[b["block_id"]], b["sha256"],
                                                       b["tile_row"], b["tile_col"])):
                if block["n_tokens"] <= remaining:
                    chosen.append(block)
                    remaining -= block["n_tokens"]
            selected = sorted(core + [i for block in chosen for i in block["indices"]])
            cap = nominal_cap
        for block in chosen:
            last[block["block_id"]] = step
        ages = [0] * len(core)
        for block in blocks:
            ages.extend([step - last[block["block_id"]]] * block["n_tokens"])
        bg_count = len(selected) - len(core)
        per_step[str(step)] = selected
        audits.append({"step_1based": step + 1, "full": full,
                       "selected_target_count": len(selected), "selected_background_count": bg_count,
                       "background_cap": cap, "nominal_partial_background_cap": nominal_cap,
                       "budget_enforced": not full, "unused_background_cap": cap - bg_count,
                       "selected_background_block_ids": [b["block_id"] for b in chosen],
                       "actual_background_fraction": bg_count / nbg if nbg else None,
                       "target_active_fraction": len(selected) / n,
                       "max_age_before_update": max(before, default=0),
                       "age_before_update_histogram": {str(k): v for k, v in sorted(Counter(before).items())},
                       "max_age_at_attention_read": max(ages, default=0),
                       "age_at_attention_read_histogram": {str(k): v for k, v in sorted(Counter(ages).items())}})
    partial = [audits[s] for s in range(N_STEPS) if s not in FULL_STEPS]
    return {"native": config == "F", "indices": per_step["1"],
            "full_steps_1based": list(range(1, 9)) if config == "F" else [1, 8],
            "per_step_indices": per_step, "per_step_audit": audits,
            "budgetmeta": {"requested_b_fraction": str(budget) if budget is not None else None,
                           "requested_b": float(budget) if budget is not None else None,
                           "partial_background_cap": nominal_cap,
                           "core_target_fraction": len(core) / n,
                           "partial_actual_background_fractions": [a["actual_background_fraction"] for a in partial],
                           "partial_target_active_fractions": [a["target_active_fraction"] for a in partial],
                           "partial_unused_background_caps": [a["unused_background_cap"] for a in partial],
                           "full8_target_fraction": sum(a["selected_target_count"] for a in audits) / (8 * n),
                           "full8_background_fraction": (sum(a["selected_background_count"] for a in audits) / (8 * nbg)
                                                         if nbg else None),
                           "core_preserved_all_steps": True, "whole_blocks_only": True,
                           "fraction_scope": "target tokens only; excludes condition/text and is not latency"}}
