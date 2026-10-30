import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit as native_module

import test_controlled_mask as fixtures
from budget_profiles import make_budget_plan, validate_budget_plan
from budget_cache_executor import budget_request
from diagnostic_schedule import native_request


TINY_CALLS = 0


class BudgetContracts(unittest.TestCase):
    def setUp(self):
        p = patch.object(native_module, "calculate_dimensions", return_value=(64, 64, 1.0))
        p.start()
        self.addCleanup(p.stop)
        self.image = Image.fromarray(np.zeros((500, 500, 3), dtype=np.uint8))
        self.case = dict(case_id="tiny_budget", bbox=[0, 0, 20, 20], source_wh=[500, 500])
        self.noise = torch.randn(1, 16, 8, generator=torch.Generator().manual_seed(91))
        self.text = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(92))

    def pipeline(self, dtype=torch.float32):
        pipe = QwenImageEditPipeline(**fixtures.make_components(dtype))
        pipe.set_progress_bar_config(disable=True)
        def count(*args):
            global TINY_CALLS
            TINY_CALLS += 1
        pipe.transformer.register_forward_pre_hook(count)
        return pipe

    def plan(self, profile):
        return make_budget_plan(self.case, profile, (4, 4), 8)

    def test_budget_contains_core_and_speed_profile_has_no_final_anchor(self):
        for profile in ("C", "B", "S", "FULL"):
            p = self.plan(profile)
            validate_budget_plan(p)
            self.assertEqual(len(p["selected_indices_per_step"]), 8)
            self.assertTrue(all(set(p["core_indices"]).issubset(x) for x in p["selected_indices_per_step"]))
        self.assertEqual(self.plan("S")["target_tokens_per_step"][-1], 4)
        self.assertEqual(self.plan("B")["target_tokens_per_step"][-1], 16)
        p = self.plan("B")
        p["selected_indices_per_step"][2] = []
        with self.assertRaises(ValueError):
            validate_budget_plan(p)

    def test_full_equivalence_padding_and_request_reset(self):
        for dtype in (torch.float32, torch.bfloat16):
            pipe = self.pipeline(dtype)
            noise, text = self.noise.to(dtype), self.text.to(dtype)
            mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
            ref, _ = native_request(pipe, self.image, None, noise, output_type="latent",
                                    prompt_embeds=text, prompt_embeds_mask=mask)
            for mode in ("source", "all"):
                out, record = budget_request(pipe, self.image, None, noise, self.plan("FULL"), mode,
                                             output_type="latent", prompt_embeds=text, prompt_embeds_mask=mask)
                self.assertTrue(torch.equal(ref.images, out.images))
                self.assertTrue(record["completed"])
                self.assertEqual(record["steps"], 8)
                self.assertFalse(pipe.transformer._controlled_local_bridge_active)

    def test_skips_real_branches_and_preserves_euler(self):
        for mode in ("source", "all"):
            pipe = self.pipeline()
            plan = self.plan("B")
            text_calls, image_rows, velocities, states = [], [], [], []
            pipe.transformer.transformer_blocks[0].txt_mlp.register_forward_pre_hook(
                lambda m, args: text_calls.append(args[0].shape[1]))
            pipe.transformer.transformer_blocks[0].img_mlp.register_forward_pre_hook(
                lambda m, args: image_rows.append(args[0].shape[1]))
            pipe.transformer.register_forward_hook(lambda m, args, out: velocities.append(out[0][:, :16].clone()))
            def callback(p, step, t, values):
                states.append(values["latents"].clone())
                return {}
            result, record = budget_request(pipe, self.image, None, self.noise, plan, mode,
                                             callback, output_type="latent", prompt_embeds=self.text)
            self.assertEqual(len(text_calls), 8 if mode == "source" else 3)
            self.assertEqual(image_rows, [32, 32, 4, 4, 4, 4, 4, 32])
            self.assertTrue(torch.isfinite(result.images).all())
            prev = self.noise
            for k, state in enumerate(states):
                expected = (prev.float() + (pipe.scheduler.sigmas[k + 1] - pipe.scheduler.sigmas[k]) * velocities[k]).to(state.dtype)
                self.assertTrue(torch.equal(expected, state))
                self.assertFalse(torch.equal(prev[:, 5:], state[:, 5:]))
                prev = state
            partial = [r for r in record["trace"] if not r["anchor"]]
            self.assertTrue(all(r["fresh_condition"] == 0 for r in partial))
            if mode == "all":
                self.assertTrue(all(r["fresh_text"] == 0 and r["joint_query_tokens"] == 4 for r in partial))

    def test_exception_restores_forward(self):
        pipe = self.pipeline()
        original = pipe.transformer.forward
        def fail(*args):
            raise RuntimeError("intentional")
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            budget_request(pipe, self.image, None, self.noise, self.plan("S"), "all", fail,
                           output_type="latent", prompt_embeds=self.text)
        self.assertEqual(original, pipe.transformer.forward)
        self.assertFalse(pipe.transformer._controlled_local_bridge_active)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BudgetContracts))
    print("CPU_TINY_DIT_CALLS=" + str(TINY_CALLS), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
