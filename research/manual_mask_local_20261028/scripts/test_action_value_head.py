import itertools
import unittest

import torch

from test_free_region import FreeRegionContracts
from action_value_head import install_action_policy, FEATURE_NAMES
from context_refresh import install_context_refresh


class ActionContracts(FreeRegionContracts):
    def test_fixed_actions_equal_frozen_controls(self):
        for dtype in (torch.float32, torch.bfloat16):
            pipe = self.pipeline(dtype); plan = self.plan()
            for action in ('N', 'C', 'F'):
                if action == 'N':
                    self.run_pipe(pipe, plan)
                else:
                    with install_context_refresh(pipe, plan, preserve_extra_velocity=action=='C', extra_tokens=4):
                        self.run_pipe(pipe)
                expected = [x.clone() for x in pipe.last_free_region_run.step_latents]
                with install_action_policy(pipe, plan, lambda f,k: action, extra_tokens=4) as run:
                    self.run_pipe(pipe)
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,pipe.last_free_region_run.step_latents)))
                self.assertEqual(sum(run['selected_target_per_step'])-plan['total_target_token_updates'],0 if action=='N' else 8)

    def test_action_tree_prefix_features_and_euler(self):
        for dtype in (torch.float32, torch.bfloat16):
            pipe = self.pipeline(dtype); plan = self.plan(); roots=[]; branches={}
            for sequence in itertools.product('NCF', repeat=2):
                policy=lambda f,k: sequence[0 if k==3 else 1]
                with install_action_policy(pipe, plan, policy, extra_tokens=4) as run:
                    self.run_pipe(pipe)
                roots.append(run['decisions'][0])
                branches.setdefault(sequence[0],[]).append(run['decisions'][1])
                previous=self.noise.to(dtype); sigmas=torch.tensor(pipe.last_free_region_run.sigmas)
                for k,state in enumerate(pipe.last_free_region_run.step_latents):
                    expected=(previous.float()+(sigmas[k+1]-sigmas[k])*run['velocities'][k]).to(dtype)
                    self.assertTrue(torch.equal(expected,state));previous=state
                for d in run['decisions']:
                    self.assertEqual(len(d['features']),len(FEATURE_NAMES))
                    if d['action']=='C':self.assertTrue(all(v<d['step'] for v in d['extra_velocity_stamps']))
                    if d['action']=='F':self.assertTrue(all(v==d['step'] for v in d['extra_velocity_stamps']))
            for rows in [roots]+list(branches.values()):
                for row in rows[1:]:
                    self.assertEqual(row['state_signature'],rows[0]['state_signature'])
                    self.assertEqual(row['features'],rows[0]['features'])

    def test_policy_exception_cleanup(self):
        pipe=self.pipeline(); original=pipe.transformer.forward
        def fail(f,k):raise RuntimeError('policy failure')
        with self.assertRaisesRegex(RuntimeError,'policy failure'):
            with install_action_policy(pipe,self.plan(),fail,extra_tokens=4):self.run_pipe(pipe)
        self.assertEqual(original,pipe.transformer.forward)
        self.assertFalse(pipe.transformer._action_policy_active)


if __name__=='__main__':
    names=['test_fixed_actions_equal_frozen_controls','test_action_tree_prefix_features_and_euler','test_policy_exception_cleanup']
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(ActionContracts(n) for n in names))
    raise SystemExit(0 if result.wasSuccessful() else 1)
