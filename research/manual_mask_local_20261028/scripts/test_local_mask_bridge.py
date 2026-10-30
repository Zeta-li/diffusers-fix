"""CPU-only integration checks; no model weights, GPU, or latency claims."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import test_controlled_mask as helpers
from controlled_mask import ControlledMaskQwenImageEditPipeline
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit as native_module
from local_mask_bridge import build_plan, install_local_executor


class LocalBridgeContracts(unittest.TestCase):
    def setUp(self):
        geometry = patch.object(native_module, "calculate_dimensions", return_value=(64, 64, 1.0))
        geometry.start()
        self.addCleanup(geometry.stop)
        self.image = Image.fromarray(np.random.default_rng(23).integers(0, 256, (64, 64, 3), dtype=np.uint8), "RGB")
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:8, :8] = 255  # one latent subposition, not whole-token white.
        self.mask = Image.fromarray(mask, "L")
        self.noise = torch.randn(1, 16, 8, generator=torch.Generator().manual_seed(24))
        self.prompt = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(25))

    def pipeline(self, dtype=torch.float32):
        pipe = ControlledMaskQwenImageEditPipeline(**helpers.make_components(dtype))
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def run_pipe(self, pipe, *, noise=None, prompt=None, mask=None):
        return pipe(image=self.image, prompt_embeds=(self.prompt if prompt is None else prompt).to(pipe.transformer.dtype),
                    mask_image=self.mask if mask is None else mask,
                    initial_noise=(self.noise if noise is None else noise).to(pipe.transformer.dtype),
                    output_type="latent", trace_latents=True).images

    def packed(self):
        from controlled_mask import pack_binary_mask
        return pack_binary_mask(self.mask, self.image.size, 64, 64, 8, 2, device="cpu")[2]

    def test_full_selected_eight_steps_equals_mask_full_fp32_bf16(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                pipe = self.pipeline(dtype)
                expected = self.run_pipe(pipe)
                expected_steps = [x.clone() for x in pipe.last_mask_run.step_latents]
                original = pipe.transformer.forward
                plan = build_plan(self.packed(), [4, 4], "full-equivalence", "all_selected")
                with install_local_executor(pipe, plan) as base:
                    actual = self.run_pipe(pipe)
                    self.assertEqual(base.step, 8)
                    self.assertEqual([x["selected_target"] for x in base.age_trace], [16]*8)
                    for left, right in zip(expected_steps, pipe.last_mask_run.step_latents):
                        self.assertTrue(torch.equal(left, right), f"full-selected state differs for {dtype}")
                    self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(pipe.transformer.forward, original)
                self.assertTrue(base.bridge_record["completed"])
                self.assertEqual(base.layer_cache, {})

    def test_front_counts_independent_halo_and_partial_subposition(self):
        packed = torch.zeros(1, 20, 8, dtype=torch.bool)
        packed[0, [0, 7, 19], 3] = True
        plan = build_plan(packed, [4, 5], "counts", "front", halo=0)
        self.assertEqual(plan["target_tokens_per_step"], [20, 20, 12, 4, 4, 4, 4, 20])
        self.assertEqual(plan["total_target_token_updates"], 88)
        self.assertEqual(plan["partition"]["group_size"], 4)
        self.assertEqual(plan["partition"]["remainder_count"], 1)
        halo = build_plan(packed, [4, 5], "halo", "full127", halo=2)
        expected = sorted({y*5+x for index in [0, 7, 19]
                           for y in range(4) for x in range(5)
                           if abs(y-index//5) <= 2 and abs(x-index%5) <= 2})
        self.assertEqual(halo["core_indices"], expected)
        self.assertEqual(halo["support_indices"], [0, 7, 19])
        self.assertEqual(halo["target_tokens_per_step"], [20, 20]+[len(expected)]*5+[20])
        white = build_plan(torch.ones_like(packed), [4, 5], "white", "front")
        self.assertEqual(white["target_tokens_per_step"], [20]*8)
        self.assertEqual(white["fallback_reason"], "fewer_than_four_noncore_tokens")
        json.dumps(plan)

    def test_partial_raw_cache_and_velocity_survive_scheduler_fusion(self):
        pipe = self.pipeline()
        plan = build_plan(self.packed(), [4, 4], "partial", "front", halo=0)
        snapshots = {}
        with install_local_executor(pipe, plan) as base:
            def before(module, args, fields):
                if "velocity" in snapshots:
                    # Previous post-forward cache must survive scheduler+mask unchanged.
                    self.assertTrue(torch.equal(base.velocity, snapshots["velocity"]))

            def after(module, args, output):
                k = base.step-1
                ids = plan["selected_indices_per_step"][k]
                inactive = sorted(set(range(16))-set(ids))
                if k and inactive:
                    for layer, (key, value) in base.layer_cache.items():
                        old_key, old_value = snapshots["kv"][layer]
                        self.assertTrue(torch.equal(key[:, inactive], old_key[:, inactive]))
                        self.assertTrue(torch.equal(value[:, inactive], old_value[:, inactive]))
                self.assertEqual(base.age[ids].tolist(), [k]*len(ids))
                self.assertTrue(set(plan["support_indices"]).issubset(ids))
                snapshots["velocity"] = base.velocity.clone()
                snapshots["kv"] = {layer: (key.clone(), value.clone()) for layer, (key, value) in base.layer_cache.items()}

            handles = [pipe.transformer.register_forward_pre_hook(before, with_kwargs=True),
                       pipe.transformer.register_forward_hook(after)]
            try:
                self.run_pipe(pipe)
                self.assertTrue(torch.equal(base.velocity, snapshots["velocity"]))
                trace = base.trace
                self.assertEqual(len(base.age_trace), 8)
                for row in trace:
                    expected_count = plan["target_tokens_per_step"][row["step"]-1]
                    self.assertEqual(row["image_qkv_tokens"], expected_count+16)
                    self.assertEqual(row["image_mlp_tokens"], expected_count+16)
                    self.assertEqual(row["joint_kv_tokens"], 32+row["text_tokens"])
                    self.assertEqual(row["joint_query_tokens"], expected_count+16+row["text_tokens"])
                self.assertLess(trace[3]["joint_query_tokens"], trace[0]["joint_query_tokens"])
            finally:
                for handle in handles:
                    handle.remove()

    def test_a_b_a_reset_and_failure_restores_forward(self):
        pipe = self.pipeline()
        original = pipe.transformer.forward
        plan = build_plan(self.packed(), [4, 4], "A", "front", halo=0)
        def run(plan, **kwargs):
            with install_local_executor(pipe, plan, condition_count=16) as base:
                result = self.run_pipe(pipe, **kwargs)
                steps = [x.clone() for x in pipe.last_mask_run.step_latents]
                trace = copy.deepcopy(base.age_trace)
            return result.clone(), steps, trace
        first, states, ages = run(plan)
        other_mask = Image.new("L", (64, 64), 255)
        other = build_plan(torch.ones(1, 16, 8, dtype=torch.bool), [4, 4], "B", "front")
        run(other, noise=self.noise+2, prompt=self.prompt-1, mask=other_mask)
        second, states_again, ages_again = run(plan)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(ages, ages_again)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(states, states_again)))
        self.assertEqual(pipe.transformer.forward, original)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with install_local_executor(pipe, plan):
                raise RuntimeError("deliberate failure")
        self.assertEqual(pipe.transformer.forward, original)
        with self.assertRaisesRegex(RuntimeError, "eight-step"):
            with install_local_executor(pipe, plan):
                pass
        self.assertEqual(pipe.transformer.forward, original)
        invalid = copy.deepcopy(plan)
        invalid["selected_indices_per_step"][3] = []
        with self.assertRaises(ValueError):
            with install_local_executor(pipe, invalid):
                pass
        with self.assertRaises(ValueError):
            build_plan(torch.zeros_like(self.packed()), [4, 4], "empty")


if __name__ == "__main__":
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LocalBridgeContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(suite="controlled_mask_local_bridge_CPU_tiny", tests_run=result.testsRun,
                          failures=len(result.failures), errors=len(result.errors),
                          cpu_tiny_dit_forward_entries=helpers.CPU_TINY_DIT_CALLS,
                          real_weight_gpu_dit_calls=0, quality_or_latency_claim=False)))
    sys.exit(0 if result.wasSuccessful() else 1)
