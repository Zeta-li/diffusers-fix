import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit as native_module

import test_controlled_mask as fixtures
from free_region import build_region_plan, run_free_region
from online_expansion import run_online_region
from expansion_head import side_features, SideExpansionHead, CenteredBudgetPredictor


class OnlineExpansionContracts(unittest.TestCase):
    def setUp(self):
        p=patch.object(native_module,"calculate_dimensions",return_value=(64,64,1.0))
        p.start();self.addCleanup(p.stop)
        self.image=Image.fromarray(np.random.default_rng(3).integers(0,256,(64,64,3),dtype=np.uint8))
        self.case=dict(case_id="A",bbox=[0,0,16,16],source_wh=[64,64])
        self.noise=torch.randn(1,16,8,generator=torch.Generator().manual_seed(4))
        self.text=torch.randn(1,3,8,generator=torch.Generator().manual_seed(5))

    def pipe(self):
        p=QwenImageEditPipeline(**fixtures.make_components(torch.float32))
        p.set_progress_bar_config(disable=True)
        return p

    def test_online_constant_box_matches_static_stepwise(self):
        p=self.pipe();box=[0,0,32,48]
        plan=build_region_plan(self.case["bbox"],[64,64],[4,4],"A",halo=0,
            target_envelope_xyxy=box,packed_channels=8)
        run_free_region(p,self.image,None,self.noise,plan=plan,prompt_embeds=self.text)
        expected=[a.clone() for a in p.last_free_region_run.step_latents];calls=[]
        def predictor(clean,reference,text,case,grid):
            calls.append((clean.clone(),reference.clone(),text.clone()))
            self.assertEqual(grid,[4,4]);return box,False
        run_online_region(p,self.image,None,self.noise,self.case,predictor,prompt_embeds=self.text)
        self.assertEqual(len(calls),1)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,p.last_free_region_run.step_latents)))
        self.assertEqual(p.last_free_region_run.executor_record["online_head"]["prediction_step"],1)

    def test_fallback_matches_native_and_requests_are_isolated(self):
        p=self.pipe();original=p.transformer.forward
        expected=run_free_region(p,self.image,None,self.noise,prompt_embeds=self.text).images
        for box in ([0,0,16,16],[0,0,64,64]):
            actual=run_online_region(p,self.image,None,self.noise,self.case,
                lambda *a:(box,True),prompt_embeds=self.text).images
            self.assertTrue(torch.equal(actual,expected))
            self.assertEqual(p.transformer.forward,original)

    def test_predictor_exception_restores_forward(self):
        p=self.pipe();original=p.transformer.forward
        def fail(*args):raise ValueError("test predictor error")
        with self.assertRaisesRegex(ValueError,"test predictor error"):
            run_online_region(p,self.image,None,self.noise,self.case,fail,prompt_embeds=self.text)
        self.assertEqual(p.transformer.forward,original)
        self.assertFalse(p._free_region_request_active)

    def test_head_bounds_and_gradient(self):
        x,a=side_features(self.noise,torch.zeros_like(self.noise),torch.ones(1,64),
            [8,8,40,40],[64,64],[4,4])
        self.assertEqual(tuple(x.shape),(4,37))
        h=SideExpansionHead();v=h(x,a)
        self.assertTrue(bool(((v>=0)&(v<=a)).all()))
        (v.square().mean()).backward()
        self.assertTrue(bool(torch.isfinite(h.net[-1].bias.grad).all()))
        self.assertGreater(float(h.net[-1].bias.grad.abs().sum()),0)

    def test_matched_rectangle_preserves_each_step_budget(self):
        case=dict(case_id="budget",bbox=[165,180,250,395],source_wh=[500,500])
        for box in ([100,100,280,440],[0,0,310,500],[165,180,250,395]):
            class Constant:
                last=None
                def __call__(self,*args):return box,False
            centered,_=CenteredBudgetPredictor(Constant())(None,None,None,case)
            a=build_region_plan(case["bbox"],[500,500],[64,64],"budget",halo=0,target_envelope_xyxy=box)
            b=build_region_plan(case["bbox"],[500,500],[64,64],"budget",halo=0,target_envelope_xyxy=centered)
            self.assertEqual(a["target_tokens_per_step"],b["target_tokens_per_step"])


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OnlineExpansionContracts))
    print(json.dumps(dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                          gpu_dit=0)))
    raise SystemExit(not result.wasSuccessful())
