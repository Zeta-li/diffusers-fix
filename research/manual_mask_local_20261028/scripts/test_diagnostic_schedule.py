import collections
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit as native_module

import test_controlled_mask as fixtures
from diagnostic_schedule import make_diagnostic_plan, native_request
from free_region import build_region_plan, run_free_region


class DiagnosticContracts(unittest.TestCase):
    def test_equal_budget_and_same_spatial_multiset(self):
        case = dict(case_id="temporal", bbox=[165, 180, 250, 395], source_wh=[500, 500])
        plans = [make_diagnostic_plan(case, name) for name in ("EARLY", "MIDDLE", "LATE")]
        self.assertEqual(len({p["total_target_token_updates"] for p in plans}), 1)
        counts = [collections.Counter(i for step in p["selected_indices_per_step"] for i in step) for p in plans]
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[0], counts[2])
        for p in plans:
            for k in (0, 1, 7):
                self.assertEqual(p["selected_indices_per_step"][k], list(range(4096)))
        self.assertNotEqual(plans[0]["target_tokens_per_step"], plans[1]["target_tokens_per_step"])

    def test_lean_entry_and_historical_wrapper_are_identical(self):
        with patch.object(native_module, "calculate_dimensions", return_value=(64, 64, 1.0)):
            pipe = QwenImageEditPipeline(**fixtures.make_components(torch.float32))
            pipe.set_progress_bar_config(disable=True)
            image = Image.fromarray(np.random.default_rng(9).integers(0, 256, (64, 64, 3), dtype=np.uint8))
            noise = torch.randn(1, 16, 8, generator=torch.Generator().manual_seed(10))
            text = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(11))
            plan = build_region_plan([0, 0, 16, 16], [64, 64], [4, 4], "tiny", halo=0, packed_channels=8)
            for current in (None, plan):
                old = run_free_region(pipe, image, None, noise, plan=current, prompt_embeds=text).images
                saved = []
                def callback(p, k, t, values):
                    saved.append(values["latents"].clone())
                    return {}
                new, record = native_request(pipe, image, None, noise, current, callback,
                                              output_type="latent", prompt_embeds=text)
                self.assertTrue(torch.equal(old, new.images))
                self.assertEqual(len(saved), 8)
                self.assertFalse(getattr(pipe.transformer, "_controlled_local_bridge_active", False))
                for a, b in zip(saved, pipe.last_free_region_run.step_latents):
                    self.assertTrue(torch.equal(a, b))

    def test_equal_budget_schedules_execute_and_cleanup(self):
        with patch.object(native_module, "calculate_dimensions", return_value=(64, 64, 1.0)):
            pipe = QwenImageEditPipeline(**fixtures.make_components(torch.float32))
            pipe.set_progress_bar_config(disable=True)
            image = Image.fromarray(np.zeros((500, 500, 3), dtype=np.uint8))
            noise = torch.randn(1, 16, 8, generator=torch.Generator().manual_seed(12))
            text = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(13))
            case = dict(case_id="tiny_timing", bbox=[0, 0, 20, 20], source_wh=[500, 500])
            original = pipe.transformer.forward
            for name in ("EARLY", "MIDDLE", "LATE"):
                plan = make_diagnostic_plan(case, name, grid=(4, 4), channels=8)
                result, record = native_request(pipe, image, None, noise, plan,
                                                output_type="latent", prompt_embeds=text)
                self.assertTrue(bool(torch.isfinite(result.images).all()))
                self.assertEqual(record["steps"], 8)
                self.assertEqual(pipe.transformer.forward, original)


if __name__ == "__main__":
    unittest.main()
