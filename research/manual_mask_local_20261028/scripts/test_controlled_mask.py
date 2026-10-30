"""CPU-only integration contracts for the project MASK_FULL wrapper.

Run from this directory with the intended checkout on PYTHONPATH::

    CUDA_VISIBLE_DEVICES='' python test_controlled_mask.py

No model download is used. The denoiser is a real, randomly initialized tiny
QwenImageTransformer2DModel; the native QwenImageEditPipeline and the actual
FlowMatchEulerDiscreteScheduler execute their normal denoising loops. Only
VAE encoding and prompt embeddings are synthetic. This validates control
flow and tensor semantics, not real-weight editing quality or GPU speed.
The patched 32-pixel geometry is confined to tests; production keeps native
calculated source/target geometry. Failed preflight calls are also counted.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline, QwenImageTransformer2DModel
from diffusers.configuration_utils import FrozenDict
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit as native_module

from controlled_mask import (
    ControlledMaskQwenImageEditPipeline, blend_after_euler, pack_binary_mask,
    tensor_digest, validate_compute_target_ids,
)


CPU_TINY_DIT_CALLS = 0


class FakeModeVAE(torch.nn.Module):
    """Deterministic mode posterior with nonzero means and unequal stds."""

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.register_buffer("_anchor", torch.zeros((), dtype=dtype))
        self.temperal_downsample = (False, True, True)
        self.config = FrozenDict(z_dim=2, latents_mean=(0.3, -0.2), latents_std=(0.7, 1.4))
        self.mode_calls = 0
        self.sample_calls = 0

    @property
    def dtype(self):
        return self._anchor.dtype

    @property
    def device(self):
        return self._anchor.device

    def encode(self, image):
        if image.shape[0:3] != (1, 3, 1):
            raise AssertionError(f"unexpected fake VAE input {tuple(image.shape)}")
        raw = F.avg_pool2d(image[:, :2, 0], 8)[:, :, None]

        def mode():
            self.mode_calls += 1
            return raw

        def sample(generator=None):
            self.sample_calls += 1
            raise AssertionError("native source encoding must use mode, not posterior sampling")

        return SimpleNamespace(latent_dist=SimpleNamespace(mode=mode, sample=sample))


def make_components(dtype=torch.float32):
    torch.manual_seed(42)
    transformer = QwenImageTransformer2DModel(
        patch_size=2, in_channels=8, out_channels=2,
        num_layers=1, attention_head_dim=16, num_attention_heads=1,
        joint_attention_dim=8, guidance_embeds=False,
        axes_dims_rope=(4, 6, 6),
    ).to(dtype=dtype).eval()

    def count_call(module, args, kwargs):
        global CPU_TINY_DIT_CALLS
        CPU_TINY_DIT_CALLS += 1

    transformer.register_forward_pre_hook(count_call, with_kwargs=True)
    scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True, base_shift=math.log(3), max_shift=math.log(3),
        base_image_seq_len=1, max_image_seq_len=8192,
        shift_terminal=None, stochastic_sampling=False,
    )
    return dict(
        transformer=transformer, scheduler=scheduler, vae=FakeModeVAE(dtype),
        text_encoder=None, tokenizer=None, processor=None,
    )


class MaskContracts(unittest.TestCase):
    def setUp(self):
        self.geometry = patch.object(native_module, "calculate_dimensions", return_value=(32, 32, 1.0))
        self.geometry.start()
        self.addCleanup(self.geometry.stop)
        rng = np.random.default_rng(123)
        self.image = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8), "RGB")
        self.white = Image.new("L", (32, 32), 255)
        self.black = Image.new("L", (32, 32), 0)
        mixed = np.zeros((32, 32), dtype=np.uint8)
        mixed[:, :8] = 255  # partial token subpositions, not token-uniform.
        self.mixed = Image.fromarray(mixed, "L")
        self.noise = torch.randn((1, 4, 8), generator=torch.Generator().manual_seed(987))
        self.prompt = torch.randn((1, 3, 8), generator=torch.Generator().manual_seed(654))

    def pipeline(self, cls=ControlledMaskQwenImageEditPipeline, dtype=torch.float32):
        pipe = cls(**make_components(dtype))
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def run_mask(self, pipe, mask, noise=None, prompt=None, **extra):
        return pipe(
            image=self.image, prompt_embeds=self.prompt.to(pipe.transformer.dtype) if prompt is None else prompt,
            mask_image=mask, initial_noise=self.noise.to(pipe.transformer.dtype) if noise is None else noise,
            output_type="latent", trace_latents=True, **extra,
        ).images

    def test_exact_2x2_channel_packing_and_nearest(self):
        # A non-square grid with different subpositions in the same token.
        pixels = torch.tensor([
            [1, 0, 0, 1, 1, 1], [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 0],
        ], dtype=torch.bool)
        transformed, latent, packed, active = pack_binary_mask(
            pixels, (6, 4), 4, 6, 1, 2, device="cpu",
        )
        expected = []
        for tile_y in range(2):
            for tile_x in range(3):
                expected.append([
                    bool(pixels[2 * tile_y + dy, 2 * tile_x + dx])
                    for channel in range(2) for dy in range(2) for dx in range(2)
                ])
        self.assertTrue(torch.equal(packed, torch.tensor([expected])))
        self.assertEqual(active.tolist(), list(range(6)))
        self.assertTrue(torch.equal(latent[0, 0], pixels))
        self.assertFalse(torch.equal(packed, packed.any(-1, keepdim=True).expand_as(packed)))
        # Deliberately odd dimensions ensure resize coordinates are exercised.
        odd = torch.zeros((5, 7), dtype=torch.bool)
        odd[0, 0] = True
        odd[4, 6] = True
        transformed, latent, _, _ = pack_binary_mask(odd, (7, 5), 8, 12, 2, 1, device="cpu")
        for y in range(4):
            for x in range(6):
                original_y = (2 * y * 5) // 8
                original_x = (2 * x * 7) // 12
                self.assertEqual(bool(latent[0, 0, y, x]), bool(odd[original_y, original_x]))

    def test_white_is_native_eight_step_bitwise_float32_and_bf16(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                components = make_components(dtype)
                native = QwenImageEditPipeline(**components)
                native.set_progress_bar_config(disable=True)
                masked = ControlledMaskQwenImageEditPipeline(**components)
                masked.set_progress_bar_config(disable=True)
                native_steps = []

                def capture(pipe, i, timestep, values):
                    native_steps.append(values["latents"].detach().cpu().clone())
                    return values

                noise = self.noise.to(dtype)
                original_noise = noise.clone()
                generator = torch.Generator().manual_seed(333)
                rng_before = generator.get_state().clone()
                expected = native(
                    image=self.image, prompt_embeds=self.prompt.to(dtype), latents=noise.clone(),
                    num_inference_steps=8, true_cfg_scale=1,
                    output_type="latent", callback_on_step_end=capture, generator=generator,
                ).images
                actual = self.run_mask(masked, self.white, noise=noise, generator=generator)
                self.assertEqual(len(native_steps), 8)
                self.assertEqual(masked.last_mask_run.dit_calls, 8)
                for a, b in zip(native_steps, masked.last_mask_run.step_latents):
                    self.assertTrue(torch.equal(a, b))
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(noise, original_noise))
                self.assertTrue(torch.equal(generator.get_state(), rng_before))
                self.assertEqual(components["vae"].sample_calls, 0)
                self.assertEqual(components["vae"].mode_calls, 2)
                self.assertTrue(masked.last_mask_run.completed)

    def test_black_uses_next_shifted_sigma_then_exact_source(self):
        pipe = self.pipeline()
        actual = self.run_mask(pipe, self.black)
        record = pipe.last_mask_run
        self.assertTrue(torch.equal(actual, record.source))
        self.assertEqual(record.sigmas[0], 1)
        self.assertEqual(record.sigmas[-1], 0)
        self.assertNotAlmostEqual(record.sigmas[1], 7 / 8)
        for index, state in enumerate(record.step_latents):
            sigma = record.sigmas[index + 1]
            expected = record.source if sigma == 0 else (sigma * record.noise.float() + (1 - sigma) * record.source.float()).to(state.dtype)
            self.assertTrue(torch.equal(state, expected))
            if index == 0:
                wrong = record.sigmas[index] * record.noise + (1 - record.sigmas[index]) * record.source
                self.assertFalse(torch.equal(state, wrong))
        # Independent per-channel normalization check, including inverse.
        preprocessed = torch.from_numpy(np.asarray(self.image).copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
        raw = F.avg_pool2d(preprocessed[:, :2], 8)[:, :, None]
        means = torch.tensor([0.3, -0.2]).reshape(1, 2, 1, 1, 1)
        stds = torch.tensor([0.7, 1.4]).reshape(1, 2, 1, 1, 1)
        normalized = (raw - means) / stds
        expected_source = QwenImageEditPipeline._pack_latents(normalized, 1, 2, 4, 4)
        torch.testing.assert_close(record.source, expected_source, atol=1e-6, rtol=0)
        unpacked = QwenImageEditPipeline._unpack_latents(record.source, 32, 32, 8)
        torch.testing.assert_close(unpacked * stds + means, raw, atol=1e-6, rtol=0)

    def test_inactive_velocity_erased_and_missing_mask_token_detected(self):
        pipe = self.pipeline()
        baseline = self.run_mask(pipe, self.mixed)
        baseline_steps = [x.clone() for x in pipe.last_mask_run.step_latents]
        active = pipe.last_mask_run.active_target_ids.tolist()
        inactive = sorted(set(range(4)) - set(active))
        self.assertTrue(active and inactive)
        validate_compute_target_ids(pipe.last_mask_run.packed_mask, torch.tensor(active))
        validate_compute_target_ids(pipe.last_mask_run.packed_mask, torch.arange(4))
        with self.assertRaisesRegex(ValueError, "omits a mask-bearing"):
            validate_compute_target_ids(pipe.last_mask_run.packed_mask, torch.tensor(active[1:]))

        def correction(indices):
            def hook(module, args, output):
                prediction = output[0].clone()
                prediction[:, indices] += 10000
                return (prediction,) + tuple(output[1:])
            return hook

        handle = pipe.transformer.register_forward_hook(correction(inactive))
        try:
            altered = self.run_mask(pipe, self.mixed)
        finally:
            handle.remove()
        self.assertTrue(torch.equal(altered, baseline))
        for a, b in zip(baseline_steps, pipe.last_mask_run.step_latents):
            self.assertTrue(torch.equal(a, b))
        # An intentionally incorrect local selector omits one mask-bearing
        # token: changing that token's prediction must survive mask fusion.
        handle = pipe.transformer.register_forward_hook(correction([active[0]]))
        try:
            incorrect = self.run_mask(pipe, self.mixed)
        finally:
            handle.remove()
        self.assertFalse(torch.equal(incorrect, baseline))

    def test_request_a_b_a_isolation_and_manifest(self):
        pipe = self.pipeline()
        a_first = self.run_mask(pipe, self.mixed)
        first_hashes = pipe.last_mask_run.manifest()["tensor_sha256"]
        first_steps = [x.clone() for x in pipe.last_mask_run.step_latents]
        self.run_mask(pipe, self.black, noise=self.noise + 3, prompt=self.prompt + 2)
        a_second = self.run_mask(pipe, self.mixed)
        record = pipe.last_mask_run
        self.assertTrue(torch.equal(a_first, a_second))
        self.assertEqual(first_hashes, record.manifest()["tensor_sha256"])
        self.assertEqual(record.dit_calls, 8)
        self.assertEqual(len(record.steps), 8)
        for first, second in zip(first_steps, record.step_latents):
            self.assertTrue(torch.equal(first, second))
        json.dumps(record.manifest())

    def test_preflight_rejections_and_failed_request_cleanup(self):
        pipe = self.pipeline()
        with self.assertRaises(ValueError):
            self.run_mask(pipe, Image.new("L", (32, 32), 128))
        with self.assertRaises(ValueError):
            self.run_mask(pipe, Image.new("L", (31, 32), 255))
        with self.assertRaises(ValueError):
            self.run_mask(pipe, self.white, num_inference_steps=7)
        with self.assertRaises(ValueError):
            self.run_mask(pipe, self.white, true_cfg_scale=4)
        with self.assertRaises(ValueError):
            self.run_mask(pipe, self.white, height=48)
        with self.assertRaises(ValueError):
            self.run_mask(pipe, self.white, noise=torch.zeros(1, 4, 7))
        with self.assertRaises(ValueError):
            self.run_mask(pipe, self.white, sigmas=np.linspace(0.9, 0.1, 8))
        self.assertFalse(pipe.last_mask_run.completed)
        self.assertIsNone(pipe._controlled_mask_context)
        # A failed schedule must not leak its hook/context into a good call.
        self.run_mask(pipe, self.white)
        self.assertEqual(pipe.last_mask_run.dit_calls, 8)

    def test_blend_rejects_nan_scalar_mask_and_preserves_inputs(self):
        source = torch.arange(16, dtype=torch.bfloat16).reshape(1, 2, 8)
        noise = source.flip(-1).clone()
        updated = source + 5
        mask = torch.zeros_like(source, dtype=torch.bool)
        mask[:, :, ::2] = True
        originals = [tensor_digest(x) for x in (source, noise, updated, mask)]
        before_rng = torch.random.get_rng_state().clone()
        actual = blend_after_euler(updated, source, noise, mask, torch.tensor(0.31789))
        background = (torch.tensor(0.31789) * noise.float() + (1 - torch.tensor(0.31789)) * source.float()).to(torch.bfloat16)
        self.assertTrue(torch.equal(actual[~mask], background[~mask]))
        self.assertTrue(torch.equal(actual[mask], updated[mask]))
        self.assertEqual(originals, [tensor_digest(x) for x in (source, noise, updated, mask)])
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
        with self.assertRaises(ValueError):
            blend_after_euler(updated, source, noise, mask[:, :, :1], 0.2)
        broken = updated.clone()
        broken[0, 0, 1] = float("nan")  # masked out does not authorize hiding NaN.
        with self.assertRaises(ValueError):
            blend_after_euler(broken, source, noise, mask, 0.2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MaskContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "suite": "controlled_MASK_FULL_CPU_tiny_real_native_pipeline",
        "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "cpu_tiny_dit_calls_including_preflight_failed_forward_entries": CPU_TINY_DIT_CALLS,
        "real_weight_gpu_dit_calls": 0,
        "quality_or_latency_claim": False,
    }, ensure_ascii=False))
    sys.exit(0 if result.wasSuccessful() else 1)
