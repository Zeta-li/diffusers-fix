import unittest

import torch

from test_free_region import FreeRegionContracts
from context_refresh import install_context_refresh, make_refresh_plan


class ContextRefreshContracts(FreeRegionContracts):
    def test_zero_additions_reproduce_baseline(self):
        for dtype in (torch.float32, torch.bfloat16):
            pipe = self.pipeline(dtype);plan = self.plan()
            self.run_pipe(pipe, plan)
            expected = [t.clone() for t in pipe.last_free_region_run.step_latents]
            with install_context_refresh(pipe, plan, preserve_extra_velocity=True, steps=()):
                self.run_pipe(pipe)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,pipe.last_free_region_run.step_latents)))

    def test_gates_ages_equal_budget_and_native_Euler(self):
        for dtype in (torch.float32, torch.bfloat16):
            for keep in (False, True):
                pipe = self.pipeline(dtype);plan = self.plan();seen=[]
                def inspect(k, base, result, extras, old, fresh, stamps):
                    if extras:
                        ids = torch.tensor(extras)
                        self.assertTrue((base.age[extras]==k).all())
                        self.assertTrue(torch.equal(result[0].index_select(1,ids),old if keep else fresh))
                        self.assertTrue((stamps[extras]<k).all() if keep else (stamps[extras]==k).all())
                        seen.append(k)
                with install_context_refresh(pipe, plan, preserve_extra_velocity=keep, extra_tokens=4, observer=inspect) as receipt:
                    self.run_pipe(pipe)
                self.assertEqual(seen,[3,5]);self.assertTrue(receipt['completed'])
                self.assertEqual(receipt['plan']['total_target_token_updates'],plan['total_target_token_updates']+8)
                previous = self.noise.to(dtype)
                sigmas = torch.tensor(pipe.last_free_region_run.sigmas)
                for k,state in enumerate(pipe.last_free_region_run.step_latents):
                    expected=(previous.float()+(sigmas[k+1]-sigmas[k])*receipt['velocities'][k]).to(dtype)
                    self.assertTrue(torch.equal(state,expected));previous=state

    def test_exception_cleanup(self):
        pipe = self.pipeline();original = pipe.transformer.forward
        def fail(*args):
            raise RuntimeError('injected')
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with install_context_refresh(pipe,self.plan(),preserve_extra_velocity=True,extra_tokens=4,observer=fail):
                self.run_pipe(pipe)
        self.assertEqual(pipe.transformer.forward,original)
        self.assertFalse(pipe.transformer._context_refresh_active)


if __name__ == '__main__':
    suite=unittest.TestSuite(ContextRefreshContracts(name) for name in
        ('test_zero_additions_reproduce_baseline','test_gates_ages_equal_budget_and_native_Euler','test_exception_cleanup'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
