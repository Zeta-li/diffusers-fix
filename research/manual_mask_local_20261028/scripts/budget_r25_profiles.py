import hashlib
import math

from free_region import build_region_plan


PROFILES = {"R25": ([0, 1, 4, 7], 0.25), "C": ([0, 1, 4, 7], 0.5), "B": ([0, 1, 7], 0.25), "B50": ([0, 1, 7], 0.5),
            "S": ([0, 1], 0.25), "FULL": (list(range(8)), 1.0)}


def make_budget_plan(case, profile="B", grid=(64, 64), channels=64):
    if profile not in PROFILES or any(x % 2 for x in grid):
        raise ValueError("Unknown profile or odd grid")
    x0, y0, x1, y1 = case["bbox"]
    w, h = case["source_wh"]
    envelope = [max(0, x0 - 32), max(0, y0 - 32), min(w, x1 + 32), min(h, y1 + 32)]
    plan = build_region_plan(case["bbox"], case["source_wh"], list(grid), case["case_id"],
                             halo=0, target_envelope_xyxy=envelope, packed_channels=channels)
    gh, gw = grid
    n = gh * gw
    tiles = [[y * gw + x, y * gw + x + 1, (y + 1) * gw + x, (y + 1) * gw + x + 1]
             for y in range(0, gh, 2) for x in range(0, gw, 2)]
    core = set(plan["core_indices"])
    mandatory = {k for k, ids in enumerate(tiles) if core.intersection(ids)}
    anchors, fraction = PROFILES[profile]
    fraction = next(p for p in (0.25, 0.5, 0.75, 1.0)
                    if p >= fraction and p * n >= 4 * len(mandatory))
    total_tiles = math.ceil(fraction * n / 4)
    extra = total_tiles - len(mandatory)
    probe_count = extra // 8
    ys = [i // gw for i in core]
    xs = [i % gw for i in core]
    bounds = (min(xs), min(ys), max(xs), max(ys))
    candidates = [k for k in range(len(tiles)) if k not in mandatory]

    def distance(k):
        y, x = divmod(tiles[k][0], gw)
        return max(bounds[0] - x - 1, x - bounds[2], bounds[1] - y - 1, y - bounds[3], 0)

    near = sorted(candidates, key=lambda k: (distance(k), k))[:extra - probe_count]
    remote = [k for k in candidates if k not in set(near)]
    remote.sort(key=lambda k: hashlib.sha256((case["case_id"] + ":" + str(k)).encode()).digest())
    selected = []
    for step in range(8):
        if step in anchors:
            ids = list(range(n))
        else:
            probes = [remote[(step * probe_count + j) % len(remote)] for j in range(probe_count)]
            ids = sorted(i for k in mandatory.union(near, probes) for i in tiles[k])
        selected.append(ids)
    plan.update(kind="budget_" + profile, profile=profile, full_steps_zero_based=anchors,
                requested_fraction=PROFILES[profile][1], effective_local_fraction=fraction,
                core_tile_count=len(mandatory), extra_probe_tiles=probe_count,
                selected_indices_per_step=selected, target_tokens_per_step=list(map(len, selected)),
                total_target_token_updates=sum(map(len, selected)), partition=None,
                fallback_reason="core_budget_upshift" if fraction > PROFILES[profile][1] else None)
    validate_budget_plan(plan)
    return plan


def validate_budget_plan(plan):
    n = plan["target_count"]
    anchors = plan["full_steps_zero_based"]
    steps = plan["selected_indices_per_step"]
    if len(steps) != 8 or not {0, 1}.issubset(anchors) or n != math.prod(plan["grid_hw"]):
        raise ValueError("Invalid budget plan")
    for step, ids in enumerate(steps):
        if ids != sorted(set(ids)) or not set(plan["core_indices"]).issubset(ids):
            raise ValueError("Unsorted indices or omitted core")
        if not ids or ids[0] < 0 or ids[-1] >= n:
            raise ValueError("Invalid selected index")
        if step in anchors and ids != list(range(n)):
            raise ValueError("Anchor must be full target")
    if list(map(len, steps)) != plan["target_tokens_per_step"]:
        raise ValueError("Incorrect row count")
    if sum(map(len, steps)) != plan["total_target_token_updates"]:
        raise ValueError("Incorrect budget")
