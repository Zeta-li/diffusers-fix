import torch
from torch import nn
from torch.nn import functional as F

ACTIONS = ('R4', 'R25', 'R6')
FEATURE_NAMES = ['box_x', 'box_y', 'box_w', 'box_h', 'box_area', 'core_fraction', 'local_fraction', 'prompt_length'] + [
    r + '_' + m for r in ('core', 'outside') for m in ('state_rms', 'source_rms', 'v1_rms', 'v2_rms', 'dv_rms', 'v_alignment')
] + ['target_grad_x', 'target_grad_y', 'source_grad_x', 'source_grad_y'] + ['text_' + str(i) for i in range(8)]


def features(case, noise, states, condition, text, text_mask, sigmas):
    assert len(states) == 2
    z1, z2 = [z.float() for z in states]
    noise, condition = noise.float(), condition.float()
    v1 = (z1-noise)/(sigmas[1]-sigmas[0])
    v2 = (z2-z1)/(sigmas[2]-sigmas[1])
    x0, y0, x1, y1 = case['bbox']
    w, h = case['source_wh']
    plan = case['plans']['R25']
    core = torch.zeros(4096, dtype=torch.bool, device=z2.device)
    core[plan['core_indices']] = True
    values = torch.tensor([x0/w, y0/h, (x1-x0)/w, (y1-y0)/h, (x1-x0)*(y1-y0)/(w*h),
        len(plan['core_indices'])/4096, plan['effective_local_fraction'], min(len(case['prompt']), 512)/512], device=z2.device)
    parts = [values]
    for mask in (core, ~core):
        if mask.any():
            a, b = v1[:,mask], v2[:,mask]
            stats = [z[:,mask].square().mean().sqrt() for z in (z2, condition, v1, v2, v2-v1)]
            stats.append((a*b).mean()/(a.square().mean().sqrt()*b.square().mean().sqrt()+1e-8))
            parts.append(torch.stack(stats))
        else:
            parts.append(torch.zeros(6, device=z2.device))
    grad = []
    for z in (z2, condition):
        image = z.reshape(1,64,64,64)
        grad.extend([(image[:,:,1:]-image[:,:,:-1]).abs().mean(), (image[:,1:]-image[:,:-1]).abs().mean()])
    parts.append(torch.stack(grad))
    mask = torch.ones(text.shape[:2], device=text.device) if text_mask is None else text_mask.float()
    pooled = (text.float()*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)
    parts.append(F.adaptive_avg_pool1d(pooled[:,None],8).flatten().to(z2.device))
    result = torch.cat(parts).float()
    assert result.numel() == len(FEATURE_NAMES) and torch.isfinite(result).all()
    return result


class RefreshHead(nn.Module):
    def __init__(self, kind='mlp'):
        super().__init__()
        self.kind = kind
        dim = 8 if kind == 'geometry' else len(FEATURE_NAMES)
        self.net = nn.Linear(dim,2) if kind == 'geometry' else nn.Sequential(nn.Linear(dim,8),nn.Tanh(),nn.Linear(8,2))

    def forward(self, x):
        v = self.net(x[...,:8] if self.kind == 'geometry' else x)
        return torch.stack([v[...,0], torch.zeros_like(v[...,0]), v[...,1]], -1)


def load_head(path, device='cpu'):
    data = torch.load(path, weights_only=True, map_location=device)
    assert data['feature_names'] == FEATURE_NAMES and tuple(data['actions']) == ACTIONS
    head = RefreshHead(data['kind']).to(device)
    head.load_state_dict(data['state_dict'])
    return head, data


def normalize(x, data):
    return ((x-data['mean'])/data['std']).clamp(-5,5)


def switch_plan(executor, case, arm):
    assert executor.step == 2
    plan = case['plans'][arm]
    assert all(plan['selected_indices_per_step'][i] == executor.plan['selected_indices_per_step'][i] for i in (0,1))
    assert plan['total_target_token_updates'] == executor.plan['total_target_token_updates']
    executor.plan = plan
    executor.full_steps = set(plan['full_steps_zero_based'])
    executor.selected = [torch.tensor(ids, dtype=torch.long, device=executor.velocity.device) for ids in plan['selected_indices_per_step']]
