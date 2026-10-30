"""Static local-Q/global-KV bridge for the controlled binary-mask pipeline.

This module never blends predictions or edits raw KV/velocity caches. The
controlled pipeline owns post-Euler latent fusion. Each context is one
eight-step CFG1 request and restores the original forward even on failure.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from legacy_local_executor import LocalTargetExecutor
from legacy_refresh_policy import BudgetController, make_partition


def build_plan(packed_mask, grid_hw, case_id, kind="front", halo=2):
    """Return a JSON-serializable frozen-selection description.

    ``packed_mask`` is exact [1,N,4*C] bool, not a token scalar mask. Halo is
    a square/Chebyshev dilation on the token grid. Empty task masks reject.
    FRONT uses six group units [15,3,0,0,0,0]; fewer than four noncore tokens
    explicitly falls back to all-full rather than inventing a partition.
    """
    if not isinstance(packed_mask, torch.Tensor) or packed_mask.dtype != torch.bool:
        raise ValueError("packed_mask must be a bool tensor")
    if len(grid_hw) != 2 or any(type(v) is not int or v <= 0 for v in grid_hw):
        raise ValueError("grid_hw must contain two positive integers")
    h, w = grid_hw
    n = h * w
    if packed_mask.ndim != 3 or packed_mask.shape[:2] != (1, n) or packed_mask.shape[2] <= 0 or packed_mask.shape[2] % 4:
        raise ValueError("packed_mask must have exact shape [1,h*w,4*C]")
    if type(halo) is not int or halo < 0:
        raise ValueError("halo must be a nonnegative integer")
    if kind not in ("front", "all_selected", "full127"):
        raise ValueError("unknown static plan kind")
    support = packed_mask.detach().cpu().any(-1)[0].reshape(h, w)
    if not support.any().item():
        raise ValueError("empty task selection is outside the local-edit protocol")
    core = F.max_pool2d(support[None, None].float(), 2 * halo + 1, stride=1, padding=halo)[0, 0].bool()
    support_ids = support.flatten().nonzero(as_tuple=False).flatten().tolist()
    core_ids = core.flatten().nonzero(as_tuple=False).flatten().tolist()
    full = list(range(n))
    partition = None
    fallback_reason = None
    if kind == "all_selected":
        selected = [full.copy() for _ in range(8)]
    elif kind == "full127":
        selected = [full.copy() if k in (0, 1, 7) else core_ids.copy() for k in range(8)]
    else:
        partition = make_partition(str(case_id), [h, w], core_ids)
        if not partition["rl_eligible"]:
            fallback_reason = "fewer_than_four_noncore_tokens"
            selected = [full.copy() for _ in range(8)]
        else:
            controller = BudgetController(partition)
            actions = [15, 3, 0, 0, 0, 0]
            selected = [controller.select(k, None if k in (0, 7) else actions[k-1]) for k in range(8)]
    plan = dict(schema_version=1, case_id=str(case_id), kind=kind, halo=halo,
                grid_hw=[h, w], target_count=n, packed_channels=packed_mask.shape[2],
                support_indices=support_ids, core_indices=core_ids,
                selected_indices_per_step=selected,
                target_tokens_per_step=[len(ids) for ids in selected],
                total_target_token_updates=sum(map(len, selected)), partition=partition,
                fallback_reason=fallback_reason, velocity_correction="none",
                halo_metric="square_Chebyshev_token_grid")
    _validate_plan(plan)
    return plan


def _validate_plan(plan):
    n = plan["target_count"]
    if type(n) is not int or n <= 0 or n != plan["grid_hw"][0] * plan["grid_hw"][1]:
        raise ValueError("invalid plan geometry")
    support, core = plan["support_indices"], plan["core_indices"]
    for name, ids in (("support", support), ("core", core)):
        if not ids or ids != sorted(set(ids)) or any(type(i) is not int or not 0 <= i < n for i in ids):
            raise ValueError(f"invalid {name} indices")
    if not set(support).issubset(core):
        raise ValueError("core omits mask support")
    selected = plan["selected_indices_per_step"]
    if len(selected) != 8 or selected[0] != list(range(n)) or selected[7] != list(range(n)):
        raise ValueError("exactly eight steps with full endpoints required")
    for ids in selected:
        if ids != sorted(set(ids)) or any(type(i) is not int or not 0 <= i < n for i in ids):
            raise ValueError("invalid per-step indices")
        if not set(core).issubset(ids):
            raise ValueError("a step omits mandatory mask/halo core")
    counts = list(map(len, selected))
    if counts != plan["target_tokens_per_step"] or sum(counts) != plan["total_target_token_updates"]:
        raise ValueError("plan count metadata mismatch")
    if plan.get("velocity_correction") != "none":
        raise ValueError("U/velocity correction is outside this bridge")


@contextmanager
def install_local_executor(pipe, plan, *, condition_count=None):
    """Yield a request-local executor; run one controlled-pipeline request.

    The real condition count is inferred on the first call unless provided.
    The surrounding controlled wrapper enforces same target/reference grid.
    ``base.bridge_record`` survives cleanup with trace/count metadata; live
    raw caches are intentionally cleared. No module forward hooks are added.
    """
    frozen = copy.deepcopy(plan)
    _validate_plan(frozen)
    model = pipe.transformer
    if getattr(model, "_controlled_local_bridge_active", False):
        raise RuntimeError("nested local bridge contexts are forbidden")
    if condition_count is not None and (type(condition_count) is not int or condition_count <= 0):
        raise ValueError("condition_count must be positive")
    original = model.forward
    base = LocalTargetExecutor(model)
    n = frozen["target_count"]
    model._controlled_local_bridge_active = True
    started = False
    succeeded = False

    def dispatch(*args, **fields):
        nonlocal started
        if args:
            raise ValueError("only named transformer arguments are supported")
        hidden = fields.get("hidden_states")
        if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != frozen["packed_channels"]:
            raise ValueError("unexpected packed transformer input")
        actual_conditions = hidden.shape[1] - n
        if actual_conditions <= 0 or (condition_count is not None and actual_conditions != condition_count):
            raise ValueError("condition count differs from request geometry")
        if not started:
            base.begin(n, actual_conditions, frozen["core_indices"], [0, 7], native=False,
                       per_step_indices={k: ids for k, ids in enumerate(frozen["selected_indices_per_step"])})
            started = True
        if not 0 <= base.step < 8 or actual_conditions != base.condition_count:
            raise RuntimeError("request step/condition geometry changed")
        selected = base.all_targets_cpu if base.step in base.full_steps else base.indices_cpu[base.step]
        if selected.tolist() != frozen["selected_indices_per_step"][base.step]:
            raise RuntimeError("runtime selection differs from frozen plan")
        if not set(frozen["support_indices"]).issubset(selected.tolist()):
            raise RuntimeError("mask-bearing target token was omitted")
        return base.forward(**fields)

    model.forward = dispatch
    try:
        yield base
        if base.step != 8:
            raise RuntimeError("local context must complete one eight-step request")
        succeeded = True
    finally:
        base.bridge_record = dict(completed=succeeded, steps=base.step,
                                  plan=frozen, condition_count=getattr(base, "condition_count", None),
                                  trace=copy.deepcopy(base.trace), age_trace=copy.deepcopy(base.age_trace),
                                  raw_cache_bytes_before_clear=base.cache_bytes())
        model.forward = original
        model._controlled_local_bridge_active = False
        base.clear()
