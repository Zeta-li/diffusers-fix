import hashlib
import json
import math
import random

import torch
from torch import nn
from torch.distributions import Categorical

SCHEMA_VERSION = 1
PARTITION_SEED = 2026102311
MIDDLE_STEPS = (1, 2, 3, 4, 5, 6)
TOTAL_UNITS = 6
FEATURE_NAMES = ['step_fraction', 'sigma', 'remaining_units_fraction', 'remaining_steps_fraction',
                 'core_fraction', 'remainder_fraction', 'global_drift_log_rms',
                 'global_velocity_log_rms', 'global_age_mean_fraction', 'global_age_max_fraction']
FEATURE_NAMES += [f'previous_action_bit_{i}' for i in range(4)]
GROUP_FEATURES = ['drift_log_rms', 'drift_log_mean_abs', 'drift_log_max_token_rms',
                  'velocity_log_rms', 'velocity_log_mean_abs', 'age_mean_fraction',
                  'age_max_fraction', 'distance_mean', 'distance_min', 'distance_max']
FEATURE_NAMES += [f'group_{g}_{name}' for g in range(4) for name in GROUP_FEATURES]
FEATURE_DIM = len(FEATURE_NAMES)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_partition(case_id, grid_hw, core_indices, seed=PARTITION_SEED):
    h, w = map(int, grid_hw)
    _require(h > 0 and w > 0, 'Positive grid required')
    raw = list(core_indices)
    core = sorted(set(raw))
    _require(core and len(core) == len(raw) and all(type(i) is int and 0 <= i < h*w for i in core), 'Unique in-range integer core required')
    core_set = set(core)
    tiles = []
    for r in range(0, h, 2):
        for c in range(0, w, 2):
            ids = [y*w+x for y in range(r, min(r+2, h)) for x in range(c, min(c+2, w)) if y*w+x not in core_set]
            if ids:
                key = hashlib.sha256(f'{seed}:{case_id}:{r//2}:{c//2}'.encode()).hexdigest()
                tiles.append(dict(tile=[r//2, c//2], indices=ids, sha256=key))
    tiles.sort(key=lambda t: (t['sha256'], t['tile']))
    ordered = [i for tile in tiles for i in tile['indices']]
    q, rem = divmod(len(ordered), 4)
    remainder = sorted(ordered[:rem])
    groups = [sorted(ordered[rem+g*q:rem+(g+1)*q]) for g in range(4)]
    owners = {i: -1 for i in remainder}
    for g, ids in enumerate(groups):
        owners.update({i: g for i in ids})
    split = [dict(tile=t['tile'], owners=sorted({owners[i] for i in t['indices']}), indices=t['indices'])
             for t in tiles if len({owners[i] for i in t['indices']}) > 1]
    result = dict(schema_version=SCHEMA_VERSION, case_id=str(case_id), seed=int(seed), grid_hw=[h, w],
        target_count=h*w, core_indices=core, remainder_indices=remainder, groups=groups,
        group_size=q, background_count=len(ordered), remainder_count=rem, rl_eligible=q > 0,
        total_middle_group_units=TOTAL_UNITS, split_tiles=split,
        partition_semantics='SHA ordered 2x2 noncore tiles, row-major within tile, first remainder tokens always fresh, then four equal token groups; group boundaries may split a tile')
    result['partition_sha256'] = _digest(result)
    return result


def validate_partition(partition):
    expected = make_partition(partition['case_id'], partition['grid_hw'], partition['core_indices'], partition['seed'])
    _require(expected == partition, 'Partition content/hash mismatch')
    return True


class BudgetController:
    def __init__(self, partition):
        validate_partition(partition)
        _require(partition['rl_eligible'], 'Fewer than four background tokens: no meaningful RL partition')
        self.partition = partition
        self.remaining_units = TOTAL_UNITS
        self.next_step = 0
        self.history = []

    def action_mask(self, step):
        _require(step == self.next_step and step in MIDDLE_STEPS, 'Mask requires next middle step')
        left_after = 6-step
        return [0 <= self.remaining_units-a.bit_count() <= 4*left_after for a in range(16)]

    def select(self, step, action=None):
        _require(step == self.next_step and step in range(8), 'Sequential eight-step calls required')
        if step in (0, 7):
            _require(action is None, 'No policy action at endpoint')
            if step == 7:
                _require(self.remaining_units == 0, 'Six units must be spent before final full step')
            ids = list(range(self.partition['target_count']))
            cost = 0
        else:
            _require(type(action) is int and 0 <= action < 16, 'Action must be a four-bit integer')
            _require(self.action_mask(step)[action], 'Infeasible action must not be silently projected')
            cost = action.bit_count()
            ids = self.partition['core_indices']+self.partition['remainder_indices']
            ids = ids+[i for g, group in enumerate(self.partition['groups']) if action & (1 << g) for i in group]
            ids = sorted(ids)
            self.remaining_units -= cost
        _require(len(ids) == len(set(ids)), 'Duplicate selected target')
        _require(set(self.partition['core_indices']).issubset(ids), 'Mandatory core lost')
        self.history.append(dict(step_0based=step, action=action, charged_group_units=cost,
                                 remaining_units=self.remaining_units, selected_count=len(ids), indices=ids))
        self.next_step += 1
        return ids


def build_features(step, sigma, remaining_units, partition, z_current, z_anchor,
                   anchor_velocity, cache_age, boundary_distance, previous_action=None):
    _require(step in MIDDLE_STEPS and 0 <= remaining_units <= TOTAL_UNITS, 'Middle-step state required')
    n = partition['target_count']
    values = [x.detach().float() for x in (z_current, z_anchor, anchor_velocity)]
    z, anchor, velocity = values
    _require(z.ndim == 3 and z.shape[0] == 1 and z.shape[1] == n and z.shape == anchor.shape == velocity.shape, 'Packed latent geometry mismatch')
    _require(all(bool(torch.isfinite(x).all()) for x in values), 'Nonfinite latent features')
    age = torch.as_tensor(cache_age, device=z.device).detach().float().reshape(-1)
    distance = torch.as_tensor(boundary_distance, device=z.device).detach().float().reshape(-1)
    _require(age.numel() == distance.numel() == n and bool(torch.isfinite(age).all()) and bool(torch.isfinite(distance).all()), 'Invalid age/distance')
    _require(bool(((age >= 0) & (age <= step) & (age == age.floor())).all()), 'Pre-step true cache age required')
    drift = z-anchor
    def rms(x):
        return math.log1p(float(x.square().mean().sqrt()))
    def meanabs(x):
        return math.log1p(float(x.abs().mean()))
    features = [step/7, float(sigma), remaining_units/TOTAL_UNITS, (7-step)/6,
                len(partition['core_indices'])/n, len(partition['remainder_indices'])/n,
                rms(drift), rms(velocity), float(age.mean())/7, float(age.max())/7]
    previous_action = 0 if previous_action is None else previous_action
    _require(type(previous_action) is int and 0 <= previous_action < 16, 'Previous action must be executed action')
    features += [float(bool(previous_action & (1 << g))) for g in range(4)]
    for group in partition['groups']:
        ids = torch.as_tensor(group, device=z.device, dtype=torch.long)
        d, v = drift.index_select(1, ids), velocity.index_select(1, ids)
        a, b = age.index_select(0, ids), distance.index_select(0, ids).abs().clamp(max=32)/32
        features += [rms(d), meanabs(d), math.log1p(float(d.square().mean(-1).sqrt().max())),
                     rms(v), meanabs(v), float(a.mean())/7, float(a.max())/7,
                     float(b.mean()), float(b.min()), float(b.max())]
    _require(len(features) == FEATURE_DIM and all(math.isfinite(x) for x in features), 'Nonfinite serialized features')
    return features


def rule_action(controller, step, rule, cache_age=None, rng=None):
    mask = controller.action_mask(step)
    if rule == 'rotation':
        action = 1 << ((step-1) % 4)
    elif rule == 'full_mid':
        action = {1: 1, 2: 0, 3: 15, 4: 0, 5: 2, 6: 0}[step]
    elif rule == 'age':
        _require(cache_age is not None, 'Age baseline requires actual pre-step age')
        age = torch.as_tensor(cache_age).detach().float().reshape(-1)
        order = sorted(range(4), key=lambda g: (-float(age[controller.partition['groups'][g]].max()),
                                                -float(age[controller.partition['groups'][g]].mean()), g))
        action = 1 << order[0]
    elif rule == 'random':
        _require(rng is not None, 'Random baseline needs an explicit seeded RNG')
        return rng.choice([i for i, valid in enumerate(mask) if valid])
    else:
        raise ValueError('Unknown fixed rule')
    _require(mask[action], 'Rule called on a prefix outside its own budget schedule')
    return action


class ActorCritic(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=64):
        super().__init__()
        _require(feature_dim == FEATURE_DIM and hidden_dim == 64, 'Frozen 54/64 architecture required')
        self.trunk = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.actor = nn.Linear(hidden_dim, 16)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        hidden = self.trunk(features)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def evaluate_actions(self, features, masks, actions):
        logits, value = self(features)
        _require(masks.dtype == torch.bool and masks.shape == logits.shape and bool(masks.any(-1).all()), 'Invalid feasible-action masks')
        _require(bool(masks.gather(-1, actions.long().unsqueeze(-1)).all()), 'Executed action was infeasible')
        distribution = Categorical(logits=logits.masked_fill(~masks, -torch.inf))
        return distribution.log_prob(actions), value, distribution.entropy()

    @torch.no_grad()
    def act(self, features, mask, deterministic=False, generator=None):
        device = next(self.parameters()).device
        x = torch.as_tensor(features, dtype=torch.float32, device=device).reshape(1, FEATURE_DIM)
        valid = torch.as_tensor(mask, dtype=torch.bool, device=device).reshape(1, 16)
        _require(bool(valid.any()), 'No feasible action')
        logits, value = self(x)
        distribution = Categorical(logits=logits.masked_fill(~valid, -torch.inf))
        action = distribution.logits.argmax(-1) if deterministic else torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1)
        return dict(action=int(action.item()), log_prob=float(distribution.log_prob(action).item()), value=float(value.item()))


def _batch(policy, observations, masks, actions):
    device = next(policy.parameters()).device
    x = torch.tensor(observations, dtype=torch.float32, device=device)
    mask = torch.tensor(masks, dtype=torch.bool, device=device)
    action = torch.tensor(actions, dtype=torch.long, device=device)
    _require(x.ndim == 2 and x.shape[1] == FEATURE_DIM and bool(torch.isfinite(x).all()), 'Finite serialized observations required')
    return x, mask, action


def supervised_update(policy, optimizer, observations, masks, actions, epochs=200, seed=2026102312,
                      value_targets=None, grad_clip=.5):
    x, mask, action = _batch(policy, observations, masks, actions)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    targets = None if value_targets is None else torch.tensor(value_targets, dtype=torch.float32, device=x.device)
    policy.train()
    history = []
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator).to(x.device)
        optimizer.zero_grad(set_to_none=True)
        logp, value, _ = policy.evaluate_actions(x[order], mask[order], action[order])
        loss = -logp.mean()
        if targets is not None:
            loss = loss+.5*(value-targets[order]).square().mean()
        _require(bool(torch.isfinite(loss)), 'Nonfinite supervised loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        _require(bool(torch.isfinite(norm)), 'Nonfinite supervised gradients')
        optimizer.step()
        history.append(float(loss.detach()))
    return dict(optimizer_steps=epochs, examples=len(x), first_loss=history[0] if history else None,
                final_loss=history[-1] if history else None, full_dit_calls=0)


def ppo_update(policy, optimizer, episodes, epochs=4, minibatch_size=24, clip=.1, gamma=1.,
               gae_lambda=.95, value_coef=.5, entropy_coef=.01, grad_clip=.5,
               seed=2026102313, target_kl=.02):
    data, advantages, returns = [], [], []
    for episode in episodes:
        transitions = episode['transitions']
        _require(len(transitions) == 6 and all(t['done'] == (i == 5) for i,t in enumerate(transitions)), 'Six-step terminated episode required')
        remaining = TOTAL_UNITS
        for i, t in enumerate(transitions):
            expected = [0 <= remaining-a.bit_count() <= 4*(5-i) for a in range(16)]
            _require(list(t['mask']) == expected and expected[t['action']], 'Episode hard budget/mask mismatch')
            remaining -= t['action'].bit_count()
            _require(all(math.isfinite(float(t[k])) for k in ['log_prob', 'value', 'reward']), 'Nonfinite PPO rollout')
        _require(remaining == 0, 'Episode did not spend exactly six units')
        gae, local_adv = 0., [0.]*6
        for i in reversed(range(6)):
            t = transitions[i]
            next_value = transitions[i+1]['value'] if i < 5 else 0.
            delta = t['reward']+gamma*next_value-float(t['value'])
            gae = delta+gamma*gae_lambda*gae
            local_adv[i] = gae
        data.extend(transitions)
        advantages.extend(local_adv)
        returns.extend(a+float(t['value']) for a,t in zip(local_adv,transitions))
    _require(data, 'At least one complete episode required')
    x, mask, action = _batch(policy, [t['features'] for t in data], [t['mask'] for t in data], [t['action'] for t in data])
    old = torch.tensor([t['log_prob'] for t in data], device=x.device, dtype=torch.float32)
    adv = torch.tensor(advantages, device=x.device, dtype=torch.float32)
    adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
    target = torch.tensor(returns, device=x.device, dtype=torch.float32)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    history = []
    stopped = False
    policy.train()
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=generator).to(x.device)
        for index in range(0, len(x), minibatch_size):
            ids = order[index:index+minibatch_size]
            logp, value, entropy = policy.evaluate_actions(x[ids], mask[ids], action[ids])
            log_ratio = logp-old[ids]
            ratio = log_ratio.exp()
            kl = ((ratio-1)-log_ratio).mean()
            if target_kl is not None and float(kl.detach()) > target_kl:
                stopped = True
                break
            surrogate = torch.minimum(ratio*adv[ids], ratio.clamp(1-clip,1+clip)*adv[ids])
            actor_loss = -surrogate.mean()
            value_loss = (value-target[ids]).square().mean()
            loss = actor_loss+value_coef*value_loss-entropy_coef*entropy.mean()
            _require(bool(torch.isfinite(loss)), 'Nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            _require(bool(torch.isfinite(norm)), 'Nonfinite PPO gradients')
            optimizer.step()
            history.append(dict(epoch=epoch, loss=float(loss.detach()), actor_loss=float(actor_loss.detach()),
                value_loss=float(value_loss.detach()), entropy=float(entropy.mean().detach()), approximate_kl=float(kl.detach())))
        if stopped:
            break
    return dict(episodes=len(episodes), transitions=len(data), optimizer_steps=len(history),
                early_stopped_by_kl=stopped, updates=history, full_dit_calls=0)
