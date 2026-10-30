import copy
import unittest

import torch
from diffusers import QwenImageTransformer2DModel

from budget_profiles import make_budget_plan
from budget_cache_executor import BudgetCacheExecutor
from trainable_budget_executor import TrainableBudgetExecutor
from cache_aware_adapters import CacheAdapters
from spatial_velocity_student import SpatialVelocityStudent, region_groups, distillation_loss


CALLS = 0


def invoke(call, state, fields, step):
    global CALLS
    CALLS += 1
    return call(hidden_states=state, timestep=torch.tensor([1 - step / 9], dtype=state.dtype), **fields)[0]


class TrainingContracts(unittest.TestCase):
    def setup_model(self, dtype=torch.float32):
        torch.manual_seed(73)
        model = QwenImageTransformer2DModel(patch_size=2, in_channels=8, out_channels=2,
            num_layers=5, attention_head_dim=16, num_attention_heads=1, joint_attention_dim=8,
            guidance_embeds=False, axes_dims_rope=(4, 6, 6)).to(dtype=dtype).eval().requires_grad_(False)
        model.set_attention_backend('native')
        fields = dict(encoder_hidden_states=torch.randn(1, 3, 8, dtype=dtype),
            encoder_hidden_states_mask=torch.tensor([[1, 1, 0]], dtype=torch.bool),
            img_shapes=[[(1, 4, 4), (1, 4, 4)]], return_dict=False)
        state = torch.randn(1, 32, 8, dtype=dtype)
        case = dict(case_id='tiny_training', bbox=[0, 0, 20, 20], source_wh=[500, 500])
        return model, state, fields, case

    def test_zero_student_and_full_match_previous_executor(self):
        for dtype in (torch.float32, torch.bfloat16):
            model, state, fields, case = self.setup_model(dtype)
            adapters = CacheAdapters(model, rank=2)
            for profile in ('B', 'FULL'):
                plan = make_budget_plan(case, profile, (4, 4), 8)
                baseline = BudgetCacheExecutor(model, plan)
                student = TrainableBudgetExecutor(model, plan, spatial=SpatialVelocityStudent(8, 16, 2))
                with torch.no_grad():
                    for step in range(8):
                        a = invoke(baseline.forward, state, fields, step)
                        b = invoke(student.forward, state, fields, step)
                        self.assertTrue(torch.equal(a, b))
                        state = state - a * 0.01
                baseline.clear(); student.clear()

    def test_checkpoint_gradients_two_step_and_teacher_isolation(self):
        model, state, fields, case = self.setup_model()
        plan = make_budget_plan(case, 'B', (4, 4), 8)
        base_model = copy.deepcopy(model)
        original = invoke(base_model.forward, state, fields, 0)
        adapters = CacheAdapters(model, rank=2)
        for layer in adapters.layers.values():
            torch.nn.init.normal_(layer.lora_B, std=0.01)
        spatial = SpatialVelocityStudent(8, 16, 2)
        torch.nn.init.normal_(spatial.output.weight, std=0.002)
        other_model, other_spatial = copy.deepcopy(model), copy.deepcopy(spatial)
        all_gradients = []
        for current, network, checkpointed in ((model, spatial, False), (other_model, other_spatial, True)):
            executor = TrainableBudgetExecutor(current, plan, spatial=network, checkpoint_blocks=checkpointed)
            z = state.clone()
            with torch.no_grad():
                for step in range(2):
                    v = invoke(executor.forward, z, fields, step)
                    z = torch.cat([z[:, :16] - 0.05 * v[:, :16], state[:, 16:]], 1)
            executor.detach_window()
            cached = {k: tuple(v.clone() for v in kv) for k, kv in executor.layer_cache.items()}
            if current is model:
                with adapters.mode(False), torch.no_grad():
                    teacher = invoke(model.forward, state, fields, 0)
                self.assertTrue(torch.equal(teacher, original))
                for k in cached:
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(cached[k], executor.layer_cache[k])))
            for step in (2, 3):
                v = invoke(executor.forward, z, fields, step)
                z = torch.cat([z[:, :16] - 0.05 * v[:, :16], state[:, 16:]], 1)
            loss = z[:, :16].square().mean()
            loss.backward()
            trainable = [p for p in current.parameters() if p.requires_grad] + list(network.parameters())
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in trainable), 0)
            all_gradients.append([p.grad.clone() for p in trainable])
            self.assertTrue(all(p.grad is None for p in current.parameters() if not p.requires_grad))
            executor.clear()
        for a, b in zip(*all_gradients):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_raw_cache_unpolluted_loss_groups_and_serialization(self):
        model, state, fields, case = self.setup_model()
        adapters = CacheAdapters(model, rank=2)
        saved = adapters.state_dict()
        adapters.load_state_dict(saved)
        network = SpatialVelocityStudent(8, 16, 2)
        with torch.no_grad():
            network.output.bias.fill_(0.2)
        executor = TrainableBudgetExecutor(model, make_budget_plan(case, 'B', (4, 4), 8), spatial=network)
        with torch.no_grad():
            for step in range(3):
                out = invoke(executor.forward, state, fields, step)
        active = executor.last_active.flatten().bool()
        self.assertTrue(torch.equal(out[:, :16][:, active], executor.velocity[:, :16][:, active]))
        self.assertFalse(torch.equal(out[:, :16][:, ~active], executor.velocity[:, :16][:, ~active]))
        groups = region_groups(executor.last_active, (4, 4))
        self.assertTrue(torch.equal(sum(g.int() for g in groups), torch.ones(16, dtype=torch.int)))
        value = out[:, :16].detach().requires_grad_()
        loss, parts = distillation_loss(value, torch.zeros_like(value), state[:, :16], .5,
                                       executor.last_active, (4, 4), [1., 1., 1.])
        loss.backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        layer = next(iter(adapters.layers.values()))
        with torch.no_grad():
            layer.lora_B.normal_(std=0.01)
            inputs = torch.randn(2, 4, layer.base.in_features)
            before = layer(inputs)
            layer.merge()
            torch.testing.assert_close(before, layer(inputs), rtol=1e-5, atol=1e-6)
        executor.clear()


if __name__ == '__main__':
    torch.set_num_threads(2)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TrainingContracts))
    print('CPU_TINY_DIT_CALLS=' + str(CALLS))
    raise SystemExit(0 if result.wasSuccessful() else 1)
