import copy
import unittest

import torch

from test_free_region import FreeRegionContracts
from free_region import run_free_region
from refresh_supervision import install_probe, candidate_regions


class ProbeContracts(FreeRegionContracts):
    def test_probe_isolation_and_same_state(self):
        for dtype in (torch.float32, torch.bfloat16):
            pipe = self.pipeline(dtype)
            plan = self.plan()
            self.run_pipe(pipe, plan)
            expected = [v.clone() for v in pipe.last_free_region_run.step_latents]
            calls, rows = [], []
            original = pipe.transformer.forward
            with install_probe(pipe, plan, lambda *v: calls.append(v), rows.append,
                               probe_steps=(3, 5), budget=4) as receipt:
                self.run_pipe(pipe)
            actual = pipe.last_free_region_run.step_latents
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
            self.assertEqual(pipe.transformer.forward, original)
            self.assertEqual(len([v for v in calls if v[0] == 'completed']), 21)
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(r['velocity_mse'] >= 0 for r in rows))
            self.assertEqual(receipt['completed_steps'], 8)
            for r in rows:
                for a in r['actions']:
                    self.assertEqual(a['additional_tokens'], 4)
                    self.assertFalse(set(a['extra']).intersection(r['selected']))

    def test_probe_cleanup_on_teacher_error(self):
        pipe = self.pipeline()
        original = pipe.transformer.forward
        def fail(event, kind, step, action):
            if event == 'invoked' and kind == 'teacher':
                raise RuntimeError('injected teacher failure')
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with install_probe(pipe, self.plan(), fail, lambda x: None, budget=4):
                self.run_pipe(pipe)
        self.assertEqual(pipe.transformer.forward, original)

    def test_replay_and_full_controls(self):
        for dtype in (torch.float32, torch.bfloat16):
            pipe = self.pipeline(dtype)
            rows = []
            with install_probe(pipe, self.plan(), lambda *v: None, rows.append,
                               teacher_steps=(3, 5), probe_steps=(3, 5), candidate_mode='controls'):
                self.run_pipe(pipe)
            for row in rows:
                actions = {a['action']: a for a in row['actions']}
                self.assertEqual(actions['REPLAY']['update_gain'], 0.)
                self.assertTrue(torch.equal(actions['REPLAY']['update_gain_map'], torch.zeros_like(row['error_map'])))
                self.assertEqual(actions['FULL']['velocity_mse'], 0.)
                self.assertEqual(actions['FULL']['update_mse'], 0.)


if __name__ == '__main__':
    suite = unittest.TestSuite(ProbeContracts(name) for name in
        ('test_probe_isolation_and_same_state', 'test_probe_cleanup_on_teacher_error', 'test_replay_and_full_controls'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
