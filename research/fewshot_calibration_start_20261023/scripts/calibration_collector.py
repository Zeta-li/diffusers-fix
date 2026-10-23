import hashlib
import time

import numpy as np
import torch

from local_executor import LocalTargetExecutor
from moments import ImageMoments


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class CalibrationCollector(LocalTargetExecutor):
    def __init__(self, model):
        super().__init__(model)
        self.handles = []
        self.collect_phase = None
        for layer, block in enumerate(model.transformer_blocks):
            self.handles.append(block.img_norm1.register_forward_hook(self.norm_hook(layer)))
            self.handles.append(block.attn.to_v.register_forward_hook(self.value_hook(layer)))

    def begin_collection(self, image_id, target_count, condition_count, indices, sample_count=128, seed=2026102302):
        self.begin(target_count, condition_count, indices, full_steps=[0, 7])
        self.image_id = str(image_id)
        self.moments = ImageMoments(self.image_id)
        self.local_calls = self.oracle_calls = 0
        self.local_completed = self.oracle_completed = 0
        self.records = []
        self.pair_seen = set()
        self.anchor_seen = set()
        self.samples = {}
        inactive = sorted(set(range(target_count)) - set(indices))
        if not inactive:
            raise ValueError('Calibration requires at least one inactive target token')
        for layer in range(len(self.model.transformer_blocks)):
            entropy = int.from_bytes(hashlib.sha256(f'{seed}:{image_id}:{layer}'.encode()).digest()[:8], 'big')
            rng = np.random.default_rng(entropy)
            chosen = sorted(rng.choice(inactive, min(sample_count, len(inactive)), replace=False).tolist())
            self.samples[layer] = torch.tensor(chosen, device=self.all_targets.device, dtype=torch.long)
        self.collect_phase = None
        return {str(layer): ids.cpu().tolist() for layer, ids in self.samples.items()}

    def norm_hook(self, layer):
        def hook(module, inputs, output):
            if self.collect_phase != 'student' or self.collect_step != 0:
                return
            if layer in self.anchor_seen:
                raise RuntimeError('Duplicate anchor norm1 hook')
            h = output[0].index_select(0, self.samples[layer]).double()
            self.moments.add_a_sums(layer, 0, h.shape[0], h.sum(0).cpu().numpy(), h.square().sum(0).cpu().numpy())
            self.anchor_seen.add(layer)
        return hook

    def value_hook(self, layer):
        def hook(module, inputs, output):
            if self.collect_phase != 'shadow':
                return
            pair = (layer, self.collect_step, 0)
            if pair in self.pair_seen:
                raise RuntimeError('Duplicate dense Value hook')
            y = output[0].index_select(0, self.samples[layer]).double()
            x = self.old_values[layer][0].flatten(1).double()
            if x.shape != y.shape:
                raise RuntimeError('Old and fresh raw Value coordinates disagree')
            self.moments.add_b_sums(layer, self.collect_step, 0, x.shape[0],
                x.sum(0).cpu().numpy(), y.sum(0).cpu().numpy(), x.square().sum(0).cpu().numpy(),
                (x*y).sum(0).cpu().numpy(), y.square().sum(0).cpu().numpy())
            self.pair_seen.add(pair)
        return hook

    def forward(self, *args, **kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        k = self.step
        if k >= 8:
            raise RuntimeError('A new calibration image requires begin_collection')
        self.collect_step = k
        self.collect_phase = 'student'
        self.local_calls += 1
        input_hash = digest(hidden)
        sync(hidden.device)
        started = time.perf_counter()
        result = super().forward(*args, **kwargs)
        self.local_completed += 1
        sync(hidden.device)
        local_seconds = time.perf_counter() - started
        prediction = result.sample if kwargs.get('return_dict', True) else result[0]
        row = dict(step_0based=k, student_input_sha256=input_hash, local_seconds=local_seconds,
            local_calls=self.local_calls, oracle_calls=self.oracle_calls, shadow=False)
        if k in range(1, 7):
            self.old_values = {}
            old_keys = {}
            pointers = {}
            for layer, (ck, cv) in self.layer_cache.items():
                ids = self.samples[layer]
                assert bool((self.age.index_select(0, ids.cpu()) == 0).all())
                self.old_values[layer] = cv.index_select(1, ids).clone()
                old_keys[layer] = ck.index_select(1, ids).clone()
                pointers[layer] = (ck.data_ptr(), cv.data_ptr())
            old_age = self.age.clone()
            old_velocity = self.velocity.clone()
            old_cpu_rng = torch.get_rng_state().clone()
            old_cuda_rng = torch.cuda.get_rng_state(hidden.device).clone() if hidden.device.type == 'cuda' else None
            self.collect_phase = 'shadow'
            self.oracle_calls += 1
            sync(hidden.device)
            started = time.perf_counter()
            shadow = self.original_forward(*args, **kwargs)
            self.oracle_completed += 1
            sync(hidden.device)
            row.update(shadow=True, oracle_seconds=time.perf_counter()-started,
                oracle_input_sha256=digest(hidden), oracle_calls=self.oracle_calls)
            assert digest(hidden) == input_hash
            assert torch.equal(self.velocity, old_velocity) and torch.equal(self.age, old_age)
            assert torch.equal(torch.get_rng_state(), old_cpu_rng)
            if old_cuda_rng is not None:
                assert torch.equal(torch.cuda.get_rng_state(hidden.device), old_cuda_rng)
            for layer, (ck, cv) in self.layer_cache.items():
                assert pointers[layer] == (ck.data_ptr(), cv.data_ptr())
                assert torch.equal(ck.index_select(1, self.samples[layer]), old_keys[layer])
                assert torch.equal(cv.index_select(1, self.samples[layer]), self.old_values[layer])
                assert (layer, k, 0) in self.pair_seen
            row.update(input_unchanged=True, sampled_all_layer_cache_unchanged=True, velocity_age_rng_unchanged=True,
                layers_sampled=len(self.samples), same_student_state=True)
            del shadow, old_velocity, old_age, old_keys, self.old_values
        self.collect_phase = None
        delta = self.sigmas[k+1] - self.sigmas[k]
        self.expected_post = (hidden[:, :self.target_count].float() + delta * prediction[:, :self.target_count]).to(prediction.dtype)
        row['student_prediction_sha256'] = digest(prediction[:, :self.target_count])
        self.records.append(row)
        return result

    def confirm_update(self, latents):
        assert torch.equal(latents, self.expected_post)
        self.records[-1]['student_only_euler_bitwise_equal'] = True
        self.records[-1]['post_latent_sha256'] = digest(latents)
        del self.expected_post

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
