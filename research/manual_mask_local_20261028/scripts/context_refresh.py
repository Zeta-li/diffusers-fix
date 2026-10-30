import copy
from contextlib import contextmanager

import torch

from legacy_local_executor import LocalTargetExecutor
from refresh_supervision import candidate_regions


def make_refresh_plan(plan, extra_tokens=64, steps=(3, 5)):
    n = plan['target_count']
    extras = [[] for _ in range(8)]
    selected = copy.deepcopy(plan['selected_indices_per_step'])
    for k in steps:
        extras[k] = candidate_regions(torch.zeros(n), selected[k], plan['core_indices'],
                                     plan['grid_hw'], extra_tokens, 20261028+k)['NEAR']
        selected[k] = sorted(set(selected[k]) | set(extras[k]))
    return dict(base_plan=copy.deepcopy(plan), extra_indices=extras, selected_indices=selected,
                target_tokens_per_step=list(map(len, selected)),
                total_target_token_updates=sum(map(len, selected)),
                selection_features='fixed geometry only, no teacher, no future output',
                candidate_region_rule='same NEAR 2x2 tiles as frozen diagnostic; stable row-major tie break')


@contextmanager
def install_context_refresh(pipe, plan, *, preserve_extra_velocity, extra_tokens=64, steps=(3, 5),
                            observer=None):
    model = pipe.transformer
    if getattr(model, '_context_refresh_active', False):
        raise RuntimeError('nested context refresh')
    original = model.forward
    base = LocalTargetExecutor(model)
    actual = make_refresh_plan(plan, extra_tokens, steps)
    n = plan['target_count']
    velocity_age = torch.full((n,), -1, dtype=torch.long)
    record = dict(plan=actual, preserve_extra_velocity=preserve_extra_velocity,
                  velocities=[], gates=[], velocity_age_trace=[], completed=False)
    started = False

    def dispatch(*args, **fields):
        nonlocal started
        if args or fields.get('return_dict', True):
            raise ValueError('named tuple calls required')
        if not started:
            base.begin(n, fields['hidden_states'].shape[1]-n, plan['core_indices'], [0, 7],
                       per_step_indices=dict(enumerate(actual['selected_indices'])))
            started = True
        k = base.step
        extras = actual['extra_indices'][k]
        ids = torch.tensor(extras, dtype=torch.long, device=fields['hidden_states'].device)
        old = base.velocity.index_select(1, ids).clone() if extras else None
        result = base.forward(**fields)
        fresh = result[0].index_select(1, ids).clone() if extras else None
        if extras and preserve_extra_velocity:
            base.velocity.index_copy_(1, ids, old)
            if not torch.equal(result[0].index_select(1, ids), old):
                raise RuntimeError('velocity gate did not retain prior value')
        accepted = plan['selected_indices_per_step'][k] if preserve_extra_velocity else actual['selected_indices'][k]
        velocity_age[accepted] = k
        record['velocities'].append(result[0][:, :n].detach().cpu().clone())
        record['velocity_age_trace'].append(dict(step=k, stamps=velocity_age.tolist()))
        if extras:
            record['gates'].append(dict(step=k, count=len(extras),
                extra_kv_fresh=bool((base.age[extras] == k).all()),
                extra_velocity_retained=preserve_extra_velocity and torch.equal(result[0].index_select(1, ids), old),
                extra_velocity_stamps=velocity_age[extras].tolist(),
                rejected_fresh_vs_old_max_abs=float((fresh.float()-old.float()).abs().max())))
        if observer:
            observer(k, base, result, extras, old, fresh, velocity_age)
        return result

    model._context_refresh_active = True
    model.forward = dispatch
    try:
        yield record
        if base.step != 8:
            raise RuntimeError('incomplete eight-step request')
        record['completed'] = True
        record['trace'] = copy.deepcopy(base.trace)
        record['kv_age_trace'] = copy.deepcopy(base.age_trace)
    finally:
        model.forward = original
        model._context_refresh_active = False
        base.clear()
