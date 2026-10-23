from __future__ import annotations
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from budget_policy import CONFIGS, SEED, build_policy, encode_case


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, default=Path(os.environ.get(
        "BUDGET_SOURCE_PLAN", root.parent / "背景轮转与上下文优化_20261022/masks/rotation_plan.json")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("BUDGET_OUTPUT_DIR", root)))
    args = parser.parse_args()
    source = args.source_plan.resolve()
    old = json.loads(source.read_text())
    cases = {}
    for cid, previous in old["cases"].items():
        case = encode_case(cid, previous)
        case["configs"] = {name: build_policy(case, name) for name in CONFIGS}
        cases[cid] = case
    plan = {"created_at": datetime.now(timezone.utc).isoformat(),
            "status": "frozen_before_gpu_execution", "seed": SEED, "configs": list(CONFIGS),
            "full_steps_1based_local": [1, 8], "partial_steps_1based": list(range(2, 8)),
            "source_plan": str(source), "source_plan_sha256": sha256(source),
            "generation_source_sha256": {name: sha256(Path(__file__).with_name(name))
                                         for name in ("budget_policy.py", "prepare_budget.py")},
            "block_hash_format": "2026102301:case_id:tile_row:tile_col",
            "tile_coordinates": "zero-based global tile coordinates; token origin=(2*tile_row,2*tile_col)",
            "selection_rule": "last real refresh oldest first, SHA256 then coordinates tie break; greedy whole blocks under exact rational floor cap; skip oversize and continue",
            "timing_contract": "encode_case outside request; build_policy inside each request e2e; no oracle/model calls in planner",
            "cases": cases}
    out = args.output_dir.resolve() / "masks/budget_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(out), "sha256": sha256(out), "cases": {
        cid: {"core": c["n_core_tokens"], "background": c["n_background_tokens"],
              "blocks": len(c["blocks"]), "configs": {name: c["configs"][name]["budgetmeta"]
                                                       for name in CONFIGS}}
        for cid, c in cases.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
