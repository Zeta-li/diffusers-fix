import copy
from contextlib import contextmanager

import torch

from legacy_local_executor import LocalTargetExecutor


def saved_state(executor, clone=False):
    state = dict(executor.__dict__)
    if clone:
        state['layer_cache'] = {k: tuple(t.clone() for t in v) for k, v in state['layer_cache'].items()}
        state['velocity'] = None if state['velocity'] is None else state['velocity'].clone()
        state['age'] = state['age'].clone()
        for name in ('trace', 'age_trace'):
            state[name] = copy.deepcopy(state[name])
        for name in ('indices', 'indices_cpu'):
            state[name] = dict(state[name])
    return state


def restore_state(executor, state, clone=False):
    executor.__dict__.update(state)
    if clone:
        executor.__dict__.update(saved_state(executor, clone=True))


def distance_map(core, grid):
    h, w = grid
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    cy = torch.tensor(core) // w
    cx = torch.tensor(core) % w
    return torch.maximum((yy-cy.min()).clamp(max=0).abs(), (yy-cy.max()).clamp(min=0)).maximum(
        torch.maximum((xx-cx.min()).clamp(max=0).abs(), (xx-cx.max()).clamp(min=0))).flatten().float()


def candidate_regions(error, selected, core, grid, budget, seed):
    h, w = grid
    occupied = set(selected)
    tiles = [[(y+dy)*w+x+dx for dy in range(2) for dx in range(2)]
             for y in range(0, h, 2) for x in range(0, w, 2)]
    tiles = [v for v in tiles if not occupied.intersection(v)]
    if budget % 4 or len(tiles)*4 < budget:
        raise ValueError('insufficient equal-sized inactive tiles')
    error = error.detach().float().cpu()
    distance = distance_map(core, grid)
    scores = torch.tensor([float(error[v].mean()) for v in tiles])
    distances = torch.tensor([float(distance[v].mean()) for v in tiles])
    orders = {
        'ERROR': torch.argsort(scores, descending=True, stable=True),
        'NEAR': torch.argsort(distances, stable=True),
        'FAR': torch.argsort(distances, descending=True, stable=True),
        'RANDOM': torch.randperm(len(tiles), generator=torch.Generator().manual_seed(seed)),
    }
    return {name: sorted(i for t in order[:budget//4].tolist() for i in tiles[t])
            for name, order in orders.items()}


def losses(student, teacher, current, dt, selected):
    n = current.shape[1]
    error = (student[:, :n].float()-teacher[:, :n].float()).square().mean(-1)[0]
    a = (current.float()+dt*student[:, :n]).to(student.dtype)
    b = (current.float()+dt*teacher[:, :n]).to(teacher.dtype)
    update_error = (a.float()-b.float()).square().mean(-1)[0]
    inactive = sorted(set(range(n))-set(selected))
    return dict(velocity_mse=float(error.mean()), update_mse=float(update_error.mean()),
                active_velocity_mse=float(error[selected].mean()),
                inactive_velocity_mse=float(error[inactive].mean()) if inactive else 0.), error, update_error


@contextmanager
def install_probe(pipe, plan, emit_call, emit_observation, teacher_steps=(2, 3, 4, 5, 6),
                  probe_steps=(3, 5), budget=64, seed=20261028, candidate_mode='diagnostic'):
    model = pipe.transformer
    original = model.forward
    base = LocalTargetExecutor(model)
    n = plan['target_count']
    observations = []
    velocities = []
    previous_input = None
    warmup_clean = None
    started = False

    def invoke(kind, step, action, fn, fields):
        emit_call('invoked', kind, step, action)
        result = fn(**fields)
        emit_call('completed', kind, step, action)
        return result[0].detach().clone()

    def dispatch(*args, **fields):
        nonlocal started, previous_input, warmup_clean
        if args or fields.get('return_dict', True):
            raise ValueError('named tuple-output calls required')
        if not started:
            base.begin(n, fields['hidden_states'].shape[1]-n, plan['core_indices'], [0, 7],
                       per_step_indices=dict(enumerate(plan['selected_indices_per_step'])))
            started = True
        k = base.step
        selected = plan['selected_indices_per_step'][k]
        current = fields['hidden_states'][:, :n]
        dt = pipe.scheduler.sigmas[k+1]-pipe.scheduler.sigmas[k]
        pre = saved_state(base, clone=True) if k in probe_steps else None
        age_before = (k-base.age).cpu().clone()
        baseline = invoke('student', k, 'FIX32', base.forward, fields)
        velocities.append(baseline[:, :n].cpu())
        if k == 1:
            warmup_clean = (current.float()-pipe.scheduler.sigmas[k]*baseline[:, :n].float()).cpu()
        if k in teacher_steps:
            post = saved_state(base)
            watched = [t for pair in base.layer_cache.values() for t in pair]+[base.velocity, base.age]
            versions = [t._version for t in watched]
            input_before = fields['hidden_states'].clone()
            teacher = invoke('teacher', k, 'DENSE_SAME_STATE', original, fields)
            if not torch.equal(fields['hidden_states'], input_before) or versions != [t._version for t in watched]:
                raise RuntimeError('teacher mutated student state')
            summary, error, update_error = losses(baseline, teacher, current, dt, selected)
            if candidate_mode not in ('diagnostic', 'controls'):
                raise ValueError('unknown candidate mode')
            candidates = {}
            if k in probe_steps:
                candidates = ({'REPLAY': [], 'FULL': sorted(set(range(n))-set(selected))}
                              if candidate_mode == 'controls' else
                              candidate_regions(error, selected, plan['core_indices'], plan['grid_hw'], budget, seed+k))
            row = dict(step=k, selected=selected, same_state_teacher=True, **summary,
                       error_map=error.cpu(), update_error_map=update_error.cpu(),
                       age_before=age_before, current=current.cpu(),
                       student_velocity=baseline[:, :n].cpu(), teacher_velocity=teacher[:, :n].cpu(),
                       current_motion=(current.float()-previous_input.float()).cpu(),
                       warmup_clean=warmup_clean, condition=fields['hidden_states'][:, n:].cpu(),
                       actions=[])
            try:
                for action, extra in candidates.items():
                    restore_state(base, pre, clone=True)
                    added = sorted(set(selected)|set(extra))
                    ids = torch.tensor(added, dtype=torch.long)
                    base.indices_cpu[k] = ids
                    base.indices[k] = ids.to(current.device)
                    value = invoke('counterfactual', k, action, base.forward, fields)
                    score, e, ue = losses(value, teacher, current, dt, selected)
                    row['actions'].append(dict(action=action, extra=extra, additional_tokens=len(extra),
                        **score, velocity_gain=summary['velocity_mse']-score['velocity_mse'],
                        update_gain=summary['update_mse']-score['update_mse'],
                        velocity_gain_map=(error-e).cpu(), update_gain_map=(update_error-ue).cpu()))
            finally:
                restore_state(base, post)
            if versions != [t._version for t in watched] or base.step != k+1:
                raise RuntimeError('counterfactual mutated retained student state')
            emit_observation(row)
            observations.append({key: value for key, value in row.items() if not isinstance(value, torch.Tensor) and key != 'actions'})
        previous_input = current.detach().clone()
        return (baseline,)

    model.forward = dispatch
    receipt = dict(observations=observations, velocities=velocities)
    try:
        yield receipt
        receipt['trace'] = copy.deepcopy(base.trace)
        receipt['age_trace'] = copy.deepcopy(base.age_trace)
        receipt['completed_steps'] = base.step
    finally:
        model.forward = original
        base.clear()
