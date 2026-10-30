import torch
from torch import nn
from torch.nn import functional as F


class SpatialVelocityStudent(nn.Module):
    def __init__(self, channels=64, width=128, blocks=4):
        super().__init__()
        self.channels, self.width, self.blocks_count = channels, width, blocks
        self.input = nn.Conv2d(channels * 4 + 5, width, 1)
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.GroupNorm(8, width), nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.SiLU(), nn.Conv2d(width, width * 2, 1), nn.SiLU(), nn.Conv2d(width * 2, width, 1)
        ) for _ in range(blocks))
        self.film = nn.Sequential(nn.Linear(67, width), nn.SiLU(), nn.Linear(width, width * 2))
        self.output = nn.Conv2d(width, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state, source, raw_velocity, fresh_delta, active, ages, sigma, text_pool, core, grid):
        h, w = grid
        batch, n, _ = state.shape
        if n != h * w or source.shape != state.shape:
            raise ValueError("Spatial student requires aligned source and target grids")
        def spatial(v):
            return v.float().transpose(1, 2).reshape(batch, -1, h, w)
        budget = active.float().mean().reshape(1, 1)
        sigma = torch.as_tensor(sigma, device=state.device).float().reshape(1, 1)
        fields = torch.cat([active, ages / 8, torch.ones_like(active) * sigma,
                            torch.ones_like(active) * budget, core], dim=-1)
        value = self.input(spatial(torch.cat([state.float(), source.float(), raw_velocity.float(),
                                             fresh_delta.float(), fields], dim=-1)))
        control = torch.cat([text_pool.float(), sigma, sigma.square(), budget], dim=-1)
        scale, shift = self.film(control).chunk(2, dim=-1)
        value = value * (1 + 0.1 * scale[:, :, None, None]) + 0.1 * shift[:, :, None, None]
        for block in self.blocks:
            value = value + block(value)
        residual = self.output(value).flatten(2).transpose(1, 2)
        return residual * (1 - active.float())


def region_groups(active, grid):
    h, w = grid
    a = active.float().reshape(1, 1, h, w)
    dilated = F.max_pool2d(a, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-a, 3, stride=1, padding=1)
    boundary = (dilated - eroded).flatten().bool()
    selected = a.flatten().bool()
    return selected & ~boundary, boundary, ~selected & ~boundary


def distillation_loss(student_v, teacher_v, state, sigma, active, grid, scales):
    groups = region_groups(active, grid)
    loss = student_v.float().sum() * 0
    values = []
    for i, (indices, weight) in enumerate(zip(groups, (1, 2, 1))):
        if indices.any():
            diff = (student_v[:, indices].float() - teacher_v[:, indices].float()) / scales[i]
            term = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=1)
            loss = loss + weight * term
            values.append(float(term.detach()))
        else:
            values.append(0.0)
    edge = student_v.float().sum() * 0
    if float(sigma) <= 0.75:
        error = -float(sigma) * (student_v.float() - teacher_v.float())
        h, w = grid
        error = error.reshape(1, h, w, -1)
        boundary = groups[1].reshape(h, w)
        for diff, mask in ((error[:, 1:] - error[:, :-1], boundary[1:] | boundary[:-1]),
                           (error[:, :, 1:] - error[:, :, :-1], boundary[:, 1:] | boundary[:, :-1])):
            if mask.any():
                edge = edge + diff[:, mask].abs().mean() / scales[1]
    return loss + 0.1 * edge, dict(group_huber=values, edge=float(edge.detach()))
