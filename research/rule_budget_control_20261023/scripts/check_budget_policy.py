from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from budget_policy import BUDGETS, CONFIGS, build_policy, encode_case, validate_target_fraction


def audit(case, config, policy):
    n, core = case["n_target_tokens"], set(case["core_indices"])
    bg = set(range(n)) - core
    blocks = case["blocks"]
    assert set().union(set(), *(set(b["indices"]) for b in blocks)) == bg
    assert sum(b["n_tokens"] for b in blocks) == len(bg)
    h, w = case["target_grid_hw"]
    for b in blocks:
        expected = {r*w+c for r in range(2*b["tile_row"], min(2*b["tile_row"]+2,h))
                    for c in range(2*b["tile_col"], min(2*b["tile_col"]+2,w))} - core
        assert set(b["indices"]) == expected
        assert b["n_tokens"] == len(expected)
        digest = hashlib.sha256(f"{case['seed']}:{case['case_id']}:{b['tile_row']}:{b['tile_col']}".encode()).hexdigest()
        assert b["sha256"] == digest
    last = [-1] * n
    for s in range(8):
        raw = policy["per_step_indices"][str(s)]
        selected = set(raw)
        assert raw == sorted(selected) and selected <= set(range(n)) and core <= selected
        full = config == "F" or s in (0, 7)
        budget = BUDGETS[config]
        cap = len(bg) if full else len(bg) * budget.numerator // budget.denominator
        selected_bg = selected - core
        assert len(selected_bg) <= cap
        if full:
            assert len(selected) == n
        for b in blocks:
            assert not (set(b["indices"]) & selected_bg) or set(b["indices"]) <= selected_bg
        
        candidates = sorted(blocks, key=lambda b: (last[b["indices"][0]], b["sha256"], b["tile_row"], b["tile_col"]))
        expected, remaining = set(), cap
        for b in candidates:
            assert len({last[i] for i in b["indices"]}) == 1
            if b["n_tokens"] <= remaining:
                expected.update(b["indices"])
                remaining -= b["n_tokens"]
        assert expected == selected_bg
        before = Counter(s-v if v >= 0 else -1 for v in last)
        for i in selected:
            last[i] = s
        ages = Counter(s-v for v in last)
        a = policy["per_step_audit"][s]
        assert a["step_1based"] == s+1 and a["selected_target_count"] == len(selected)
        assert a["selected_background_count"] == len(selected_bg)
        assert a["background_cap"] == cap and a["unused_background_cap"] == cap-len(selected_bg)
        assert a["max_age_at_attention_read"] == max(ages)
        assert {int(k):v for k,v in a["age_at_attention_read_histogram"].items()} == dict(ages)
        assert {int(k):v for k,v in a["age_before_update_histogram"].items()} == dict(before)
        assert len(ages) and sum(ages.values()) == n
    assert policy["budgetmeta"]["full8_target_fraction"] == sum(len(v) for v in policy["per_step_indices"].values())/(8*n)
    assert policy == build_policy(case, config) == build_policy(case, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).resolve().parent.parent / "masks/budget_plan.json")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan["configs"] == list(CONFIGS)
    tested = []
    for cid, case in plan["cases"].items():
        for config in CONFIGS:
            audit(case, config, case["configs"][config])
            tested.append(f"real:{cid}:{config}")
    synthetic = [("nonsquare", [3,7], [0,1,7,8,14]),
                 ("small_background", [2,2], [0,1,2]),
                 ("all_core", [3,5], list(range(15))),
                 ("no_core", [5,3], []),
                 ("ties_and_skip", [3,5], [0,1,5,6,2,3,7,8])]
    for cid, hw, core in synthetic:
        case = encode_case(cid, {"target_grid_hw": hw, "core_indices": core})
        for config in CONFIGS:
            audit(case, config, build_policy(case, config))
            tested.append(f"synthetic:{cid}:{config}")
        try:
            validate_target_fraction(case, Fraction(len(core), hw[0]*hw[1])-Fraction(1,100))
        except ValueError:
            pass
        else:
            raise AssertionError("infeasible total-target fraction was accepted")
        assert validate_target_fraction(case, Fraction(1)) == 1
    
    case = encode_case("one_bg", {"target_grid_hw": [2,2], "core_indices": [0,1,2]})
    assert build_policy(case,"B500")["per_step_audit"][1]["selected_background_count"] == 0
    
    case = encode_case("skip", {"target_grid_hw": [2,3], "core_indices": []})
    p = build_policy(case,"B500")
    assert p["per_step_audit"][1]["selected_background_count"] == 2
    summary = {"status": "PASS", "policy_checks": len(tested), "checks": tested,
               "additional_checks": ["exact fractions", "infeasible total-rho rejected", "zero cap", "skip oversized block"],
               "model_calls": 0, "dit_forwards": 0, "stdlib_only": True,
               "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
