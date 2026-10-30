"""Real tiny native-Qwen CPU tests for unrestricted regional computation.

Only FakeModeVAE/make_components fixtures are reused from the earlier CPU
test module; no controlled/masked pipeline is instantiated or executed.
There are no downloaded weights, GPU runs, quality or latency claims here.
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from diffusers import QwenImageEditPipeline
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit as native_module
import test_controlled_mask as fixtures

from free_region import build_region_plan, run_free_region


class FreeRegionContracts(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(native_module, "calculate_dimensions", return_value=(64, 64, 1.0))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.image = Image.fromarray(np.random.default_rng(74).integers(0, 256, (64, 64, 3), dtype=np.uint8), "RGB")
        self.noise = torch.randn(1, 16, 8, generator=torch.Generator().manual_seed(75))
        self.text = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(76))

    def pipeline(self, dtype=torch.float32):
        pipe = QwenImageEditPipeline(**fixtures.make_components(dtype))
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def plan(self, bbox=(0, 0, 16, 16), kind="front", case="A", envelope=None):
        return build_region_plan(bbox, [64, 64], [4, 4], case, halo=0,
                                 target_envelope_xyxy=envelope, kind=kind, packed_channels=8)

    def run_pipe(self, pipe, plan=None, noise=None, text=None):
        return run_free_region(
            pipe, self.image, None,
            self.noise.to(pipe.transformer.dtype) if noise is None else noise.to(pipe.transformer.dtype),
            plan=plan, prompt_embeds=self.text.to(pipe.transformer.dtype) if text is None else text.to(pipe.transformer.dtype),
            trace_latents=True,
        ).images

    def native(self, pipe):
        states = []

        def callback(native_pipe, index, timestep, values):
            states.append(values["latents"].detach().cpu().clone())
            return {}

        result = pipe(
            image=self.image, prompt_embeds=self.text.to(pipe.transformer.dtype),
            latents=self.noise.to(pipe.transformer.dtype).clone(),
            true_cfg_scale=1, num_inference_steps=8, output_type="latent",
            callback_on_step_end=callback,
        ).images
        return result, states

    def test_native_and_free_full_are_stepwise_identical_fp32_bf16(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                pipe = self.pipeline(dtype)
                expected, states = self.native(pipe)
                actual = self.run_pipe(pipe)
                record = pipe.last_free_region_run
                self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(record.dit_calls, 8)
                self.assertIsNone(record.executor_record)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(states, record.step_latents)))
                self.assertFalse(record.manifest()["background_fusion"])
                self.assertFalse(record.manifest()["image_crop"])
                self.assertFalse(record.manifest()["rgb_copy"])

    def test_all_selected_equivalence_and_different_bbox_cannot_crop(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                pipe = self.pipeline(dtype)
                baseline = self.run_pipe(pipe)
                baseline_states = [value.clone() for value in pipe.last_free_region_run.step_latents]
                original_forward = pipe.transformer.forward
                for plan in (self.plan(kind="all_selected"), self.plan((40, 8, 64, 56), kind="all_selected", case="far")):
                    actual = self.run_pipe(pipe, plan)
                    record = pipe.last_free_region_run
                    self.assertTrue(torch.equal(baseline, actual))
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline_states, record.step_latents)))
                    self.assertEqual(record.executor_record["steps"], 8)
                    self.assertTrue(record.executor_record["completed"])
                    self.assertEqual(pipe.transformer.forward, original_forward)
                    self.assertFalse(plan["source_bbox_is_output_boundary"])
                    self.assertEqual(plan["target_tokens_per_step"], [16] * 8)

    def test_enlarged_target_envelope_and_conservative_mapping(self):
        plain = self.plan()
        enlarged = self.plan(envelope=(0, 0, 32, 48))
        self.assertEqual(plain["source_bbox_indices"], [0])
        self.assertEqual(enlarged["support_indices"], [0, 1, 4, 5, 8, 9])
        self.assertTrue(set(plain["core_indices"]).issubset(enlarged["core_indices"]))
        self.assertFalse(enlarged["target_envelope_is_output_boundary"])
        self.assertEqual(enlarged["roi_role"], "compute_hint_only_not_generation_permission")
        halo = build_region_plan((0, 0, 16, 16), [64, 64], [4, 4], "halo", halo=1,
                                 target_envelope_xyxy=(0, 0, 32, 48), packed_channels=8)
        self.assertEqual(halo["core_indices"], [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
        fractional = build_region_plan((24.9, 0.1, 25.1, 19.9), [100, 60], [3, 4], "fractional", halo=0, packed_channels=8)
        self.assertEqual(fractional["source_bbox_indices"], [0, 1])
        json.dumps(halo)
        with self.assertRaises(ValueError):
            self.plan((0, 0, 0, 1))
        with self.assertRaises(ValueError):
            self.plan(envelope=(-1, 0, 20, 20))

    def test_local_scheduler_updates_outside_hint_with_unmodified_euler(self):
        pipe = self.pipeline()
        plan = self.plan()
        velocities = []

        def capture_velocity(module, positional, output):
            velocities.append(output[0][:, :16].detach().cpu().clone())

        handle = pipe.transformer.register_forward_hook(capture_velocity)
        try:
            self.run_pipe(pipe, plan)
        finally:
            handle.remove()
        record = pipe.last_free_region_run
        outside = sorted(set(range(16)) - set(plan["support_indices"]))
        previous = self.noise.clone()
        self.assertEqual(len(velocities), 8)
        changed_outside = []
        for index, state in enumerate(record.step_latents):
            dt = torch.tensor(record.sigmas[index + 1]) - torch.tensor(record.sigmas[index])
            expected = (previous.float() + dt * velocities[index].float()).to(velocities[index].dtype)
            # This check covers ALL target tokens and would fail if any ROI
            # complement were overwritten with a source/noise trajectory.
            self.assertTrue(torch.equal(state, expected))
            changed_outside.append(not torch.equal(state[:, outside], previous[:, outside]))
            previous = state
        self.assertTrue(all(changed_outside))
        trace = record.executor_record["trace"]
        self.assertTrue(any(row["selected_target"] < 16 for row in trace))
        self.assertTrue(all(row["joint_kv_tokens"] == 32 + row["text_tokens"] for row in trace))

    def test_inactive_velocity_correction_survives_outside_roi(self):
        pipe = self.pipeline()
        plan = self.plan()
        baseline = self.run_pipe(pipe, plan)
        baseline_states = [value.clone() for value in pipe.last_free_region_run.step_latents]
        altered_steps = []

        def correction(module, positional, output):
            index = pipe.last_free_region_run.dit_calls - 1
            inactive = sorted(set(range(16)) - set(plan["selected_indices_per_step"][index]))
            if not inactive:
                return output
            prediction = output[0].clone()  # never write the executor's raw cache.
            prediction[:, inactive] += 0.5
            altered_steps.append((index, inactive))
            return (prediction,) + tuple(output[1:])

        handle = pipe.transformer.register_forward_hook(correction)
        try:
            altered = self.run_pipe(pipe, plan)
        finally:
            handle.remove()
        self.assertTrue(altered_steps)
        first_step, inactive = altered_steps[0]
        changed = pipe.last_free_region_run.step_latents[first_step]
        self.assertFalse(torch.equal(changed[:, inactive], baseline_states[first_step][:, inactive]))
        self.assertFalse(torch.equal(altered, baseline))

    def test_a_b_a_cache_reset_and_failure_cleanup(self):
        pipe = self.pipeline()
        original_forward = pipe.transformer.forward
        plan = self.plan()
        first = self.run_pipe(pipe, plan)
        states = [value.clone() for value in pipe.last_free_region_run.step_latents]
        ages = copy.deepcopy(pipe.last_free_region_run.executor_record["age_trace"])
        self.run_pipe(pipe, self.plan((32, 32, 64, 64), case="B"), noise=self.noise + 2, text=self.text - 1)
        last = self.run_pipe(pipe, plan)
        self.assertTrue(torch.equal(first, last))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(states, pipe.last_free_region_run.step_latents)))
        self.assertEqual(ages, pipe.last_free_region_run.executor_record["age_trace"])
        self.assertEqual(pipe.transformer.forward, original_forward)
        self.assertFalse(pipe.transformer._controlled_local_bridge_active)
        self.assertFalse(pipe._free_region_request_active)

        def fail(module, positional, fields):
            raise RuntimeError("deliberate native failure")

        handle = pipe.transformer.register_forward_pre_hook(fail, with_kwargs=True)
        try:
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                self.run_pipe(pipe, plan)
        finally:
            handle.remove()
        self.assertEqual(pipe.transformer.forward, original_forward)
        self.assertFalse(pipe.transformer._controlled_local_bridge_active)
        self.assertFalse(pipe._free_region_request_active)
        self.assertFalse(pipe.last_free_region_run.completed)
        self.run_pipe(pipe, plan)
        self.assertTrue(pipe.last_free_region_run.completed)

    def test_reject_pipeline_subclass_mask_argument_and_modified_plan(self):
        class NonNativePipeline(QwenImageEditPipeline):
            pass

        subclass = NonNativePipeline(**fixtures.make_components())
        with self.assertRaisesRegex(TypeError, "genuine native"):
            self.run_pipe(subclass)
        pipe = self.pipeline()
        with self.assertRaises(TypeError):
            run_free_region(pipe, self.image, None, self.noise, prompt_embeds=self.text, mask_image=Image.new("L", self.image.size))
        wrong = self.plan()
        wrong["selected_indices_per_step"][2] = []
        with self.assertRaisesRegex(ValueError, "changed after construction"):
            self.run_pipe(pipe, wrong)


if __name__ == "__main__":
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FreeRegionContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "suite": "native_free_region_CPU_real_tiny_Qwen",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "cpu_tiny_transformer_forward_entries": fixtures.CPU_TINY_DIT_CALLS,
        "real_weight_gpu_dit_calls": 0, "quality_or_latency_claim": False,
    }))
    sys.exit(0 if result.wasSuccessful() else 1)
