import time

import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from local_executor import LocalTargetExecutor


def error_metrics(candidate, reference):
    a, b = candidate.float(), reference.float()
    if a.numel() == 0:
        return dict(count=0, mse=0., rmse=0., relative_rmse=0., cosine=None,
                    squared_error_sum=0., reference_squared_sum=0.)
    e2 = (a - b).square().sum().item()
    r2 = b.square().sum().item()
    cosine = torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
    return dict(count=a.numel(), mse=e2 / a.numel(), rmse=(e2 / a.numel()) ** .5,
                relative_rmse=(e2 / max(r2, 1e-30)) ** .5, cosine=cosine,
                squared_error_sum=e2, reference_squared_sum=r2)


def mix_prediction(local, dense, selected, target_count, mode):
    if mode not in ['B', 'V', 'K', 'D']:
        raise ValueError(mode)
    if mode == 'B':
        return local.clone()
    result = dense.clone()
    if mode == 'K':
        result[:, :target_count].index_copy_(1, selected, local[:, :target_count].index_select(1, selected))
    elif mode == 'V':
        result = local.clone()
        result[:, :target_count].index_copy_(1, selected, dense[:, :target_count].index_select(1, selected))
        result[:, target_count:] = dense[:, target_count:]
    return result


class AttributionExecutor(LocalTargetExecutor):
    def begin_attribution(self, target_count, condition_count, indices, mode, sigmas):
        self.begin(target_count, condition_count, indices, full_steps=[0, 7], native=mode == 'D')
        self.mode = mode
        self.sigmas = sigmas
        self.records = []
        self.captures = []
        self.local_calls = 0
        self.oracle_calls = 0
        self.native_calls = 0

    def forward(self, *args, **kwargs):
        if self.native:
            self.native_calls += 1
            return super().forward(*args, **kwargs)
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        current_step = self.step
        selected = self.all_targets if current_step in self.full_steps else self.indices[current_step]
        active_mask = torch.zeros(self.target_count, device=hidden.device, dtype=torch.bool)
        active_mask[selected] = True
        inactive = (~active_mask).nonzero().flatten()
        target = hidden[:, :self.target_count].detach().cpu().clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        local = super().forward(*args, **kwargs)
        self.local_calls += 1
        local = (local.sample if kwargs.get('return_dict', True) else local[0]).clone()
        torch.cuda.synchronize()
        local_seconds = time.perf_counter() - start
        probes = []
        for layer in [0, len(self.model.transformer_blocks) // 2, len(self.model.transformer_blocks) - 1]:
            ck, cv = self.layer_cache[layer]
            ids = inactive[:8] if inactive.numel() else selected[:8]
            probes.append((ck, cv, ids, ck.index_select(1, ids).clone(), cv.index_select(1, ids).clone()))
        before_velocity = self.velocity.clone()
        start = time.perf_counter()
        dense = self.original_forward(*args, **kwargs)
        self.oracle_calls += 1
        dense = (dense.sample if kwargs.get('return_dict', True) else dense[0]).clone()
        torch.cuda.synchronize()
        oracle_seconds = time.perf_counter() - start
        assert torch.equal(before_velocity, self.velocity)
        for ck, cv, ids, k, v in probes:
            assert torch.equal(k, ck.index_select(1, ids)) and torch.equal(v, cv.index_select(1, ids))
        returned = mix_prediction(local, dense, selected, self.target_count, self.mode)
        ae = error_metrics(local[:, :self.target_count].index_select(1, selected), dense[:, :self.target_count].index_select(1, selected))
        ie = error_metrics(local[:, :self.target_count].index_select(1, inactive), dense[:, :self.target_count].index_select(1, inactive))
        ce = error_metrics(local[:, :self.target_count], dense[:, :self.target_count])
        difference = local[:, :self.target_count].float() - dense[:, :self.target_count].float()
        active_difference = difference * active_mask[None, :, None]
        inactive_difference = difference * (~active_mask)[None, :, None]
        cross_term = (active_difference * inactive_difference).sum().item()
        factorization_max_abs = (difference - active_difference - inactive_difference).abs().max().item()
        assert cross_term == 0. and factorization_max_abs == 0.
        assert abs(ce['squared_error_sum'] - ae['squared_error_sum'] - ie['squared_error_sum']) <= max(1e-5, ce['squared_error_sum'] * 1e-5)
        if current_step in self.full_steps:
            assert torch.equal(local, dense)
        delta_tensor = self.sigmas[current_step + 1] - self.sigmas[current_step]
        delta = float(delta_tensor)
        def euler(prediction):
            return (hidden[:, :self.target_count].float() + delta_tensor * prediction[:, :self.target_count]).to(prediction.dtype)
        local_next, dense_next, returned_next = euler(local), euler(dense), euler(returned)
        self.records.append(dict(step=current_step + 1, full_step=current_step in self.full_steps,
            active_count=selected.numel(), inactive_count=inactive.numel(), sigma_delta=delta,
            active_error=ae, inactive_error=ie, combined_error=ce,
            returned_error=error_metrics(returned[:, :self.target_count], dense[:, :self.target_count]),
            inactive_error_energy_fraction=ie['squared_error_sum'] / ce['squared_error_sum'] if ce['squared_error_sum'] else None,
            orthogonality_cross_term=cross_term, factorization_max_abs=factorization_max_abs, oracle_sampled_cache_unchanged=True,
            euler_error=dict(active_rmse=abs(delta) * ae['rmse'], inactive_rmse=abs(delta) * ie['rmse'], combined_rmse=abs(delta) * ce['rmse']),
            rounded_euler_error=error_metrics(local_next, dense_next),
            local_seconds=local_seconds, oracle_seconds=oracle_seconds))
        self.captures.append(dict(current_latent=target, local=local[:, :self.target_count].cpu(),
            dense=dense[:, :self.target_count].cpu(), returned=returned[:, :self.target_count].cpu(), selected=selected.cpu(),
            expected_post_latent=returned_next.cpu()))
        if self.mode == 'V':
            self.velocity[:, :self.target_count].index_copy_(1, selected, dense[:, :self.target_count].index_select(1, selected))
            self.velocity[:, self.target_count:] = dense[:, self.target_count:]
        if kwargs.get('return_dict', True):
            return Transformer2DModelOutput(sample=returned)
        return (returned,)
