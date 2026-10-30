import copy
from contextlib import contextmanager

import torch
from torch import nn

from controlled_mask import tensor_digest
from context_refresh import make_refresh_plan
from legacy_local_executor import LocalTargetExecutor
from refresh_supervision import distance_map


ACTIONS = ('N', 'C', 'F')
FEATURE_NAMES = ['step', 'sigma', 'core_fraction', 'selected_fraction', 'extra_fraction',
                 'core_aspect', 'extra_distance'] + [
    region + '_' + metric for region in ('core', 'extra', 'inactive')
    for metric in ('motion_rms', 'latent_rms', 'velocity_rms', 'kv_age', 'velocity_age', 'warmup_difference_rms')]


def state_features(k, sigma, current, previous, old_velocity, kv_stamps, velocity_stamps,
                   warmup, condition, plan, extras):
    n = plan['target_count']; w = plan['grid_hw'][1]; core = plan['core_indices']
    selected = plan['selected_indices_per_step'][k]
    xs = [i % w for i in core]; ys = [i // w for i in core]
    distance = distance_map(core, plan['grid_hw'])
    values = [k/7, float(sigma), len(core)/n, len(selected)/n, len(extras)/n,
              (max(xs)-min(xs)+1)/(max(ys)-min(ys)+1), float(distance[extras].mean())/max(plan['grid_hw'])]
    motion = current.float()-previous.float()
    difference = warmup.float()-condition.float()
    for region in (core, extras, sorted(set(range(n))-set(selected))):
        ids = torch.tensor(region, dtype=torch.long, device=current.device)
        def rms(value):
            return float(value[:, ids].float().square().mean().sqrt()) if len(region) else 0.
        values.extend([rms(motion), rms(current), rms(old_velocity[:, :n]),
            float((k-kv_stamps[region]).float().mean())/7 if region else 0.,
            float((k-velocity_stamps[region]).float().mean())/7 if region else 0., rms(difference)])
    result = torch.tensor(values, dtype=torch.float32)
    if result.numel() != len(FEATURE_NAMES) or not torch.isfinite(result).all():
        raise RuntimeError('invalid online features')
    return result


class ActionValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 16), nn.Tanh(), nn.Linear(16, 3))

    def forward(self, x):
        return self.net(x)


class Predictor:
    def __init__(self, checkpoint, device='cuda'):
        data = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if data['feature_names'] != FEATURE_NAMES or tuple(data['actions']) != ACTIONS:
            raise ValueError('checkpoint contract mismatch')
        self.model = ActionValueHead().to(device).eval()
        self.model.load_state_dict(data['state_dict']); self.model.requires_grad_(False)
        self.mean = data['mean'].to(device); self.std = data['std'].to(device)
        self.last_scores = None

    @torch.no_grad()
    def __call__(self, features, step):
        x = ((features.to(self.mean.device)-self.mean)/self.std).clamp(-5, 5)
        q = self.model(x)
        self.last_scores = q.cpu().tolist()
        return ACTIONS[int(q.argmax())]


@contextmanager
def install_action_policy(pipe, plan, policy, extra_tokens=64, observer=None):
    model = pipe.transformer
    if getattr(model, '_action_policy_active', False):
        raise RuntimeError('nested action policy')
    original = model.forward; base = LocalTargetExecutor(model)
    additions = make_refresh_plan(plan, extra_tokens)
    n = plan['target_count']; velocity_stamps = torch.full((n,), -1, dtype=torch.long)
    previous = None; warmup = None; started = False
    record = dict(decisions=[], velocities=[], selected_target_per_step=[], velocity_age_trace=[], completed=False,
                  base_plan=copy.deepcopy(plan), maximum_extra_target_updates=2*extra_tokens)

    def dispatch(*args, **fields):
        nonlocal started, previous, warmup
        if args or fields.get('return_dict', True):
            raise ValueError('named tuple calls required')
        if not started:
            base.begin(n, fields['hidden_states'].shape[1]-n, plan['core_indices'], [0, 7],
                       per_step_indices=dict(enumerate(plan['selected_indices_per_step'])))
            started = True
        k = base.step; current = fields['hidden_states'][:, :n]
        candidates = additions['extra_indices'][k]; action = 'N'; row = None
        if candidates:
            features = state_features(k, pipe.scheduler.sigmas[k], current, previous, base.velocity,
                base.age, velocity_stamps, warmup, fields['hidden_states'][:, n:], plan, candidates)
            row = dict(step=k, features=features.tolist(), candidates=candidates,
                state_signature=[tensor_digest(x) for x in (current, base.velocity[:, :n], base.age, velocity_stamps)])
            action = policy(features.clone(), k)
            if action not in ACTIONS:
                raise ValueError('unknown action')
            row.update(action=action, predicted_scores=getattr(policy, 'last_scores', None))
            record['decisions'].append(row)
        extras = candidates if action != 'N' else []
        selected = sorted(set(plan['selected_indices_per_step'][k]) | set(extras))
        ids = torch.tensor(selected, dtype=torch.long)
        base.indices_cpu[k] = ids; base.indices[k] = ids.to(current.device)
        extra_ids = torch.tensor(extras, dtype=torch.long, device=current.device)
        old = base.velocity.index_select(1, extra_ids).clone() if extras else None
        result = base.forward(**fields)
        if action == 'C':
            base.velocity.index_copy_(1, extra_ids, old)
            if not torch.equal(result[0].index_select(1, extra_ids), old):
                raise RuntimeError('velocity retention failed')
        accepted = plan['selected_indices_per_step'][k] if action == 'C' else selected
        velocity_stamps[accepted] = k
        if row is not None:
            row.update(extra_kv_fresh=bool((base.age[extras] == k).all()) if extras else None,
                accepted_new_velocity=action == 'F', extra_velocity_stamps=velocity_stamps[candidates].tolist(),
                added_tokens=len(extras))
        if k == 1:
            warmup = (current.float()-pipe.scheduler.sigmas[k]*result[0][:, :n].float()).detach().clone()
        previous = current.detach().clone()
        record['velocities'].append(result[0][:, :n].detach().cpu().clone())
        record['selected_target_per_step'].append(len(selected))
        record['velocity_age_trace'].append(velocity_stamps.tolist())
        if observer:
            observer(k, base, result, row, velocity_stamps)
        return result

    model._action_policy_active = True; model.forward = dispatch
    try:
        yield record
        if base.step != 8:
            raise RuntimeError('incomplete request')
        record.update(completed=True, trace=copy.deepcopy(base.trace), kv_age_trace=copy.deepcopy(base.age_trace))
    finally:
        model.forward = original; model._action_policy_active = False; base.clear()
