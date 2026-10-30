import copy
import datetime
import json

import torch

import run_s1 as common


root = common.ROOT
out = root/'data/object_selectivity_seed2_v1_preregistration.json'
noise_path = root/'data/object_selectivity_seed2_noise.pt'
if out.exists() or noise_path.exists():
    raise RuntimeError('do not overwrite frozen repeat')
seed = 2026102802
torch.save(torch.randn(1, 4096, 64, generator=torch.Generator().manual_seed(seed),
                       dtype=torch.bfloat16), noise_path)
base = json.loads((root/'data/object_selectivity_fixed32_v1_preregistration.json').read_text())
cases = copy.deepcopy(base['cases'])
for c in cases:
    c.pop('reference_run')
    c['arms'] = [dict(name='F', kind='native'), dict(name='FIX32', kind='front', margin=32)]
spec = dict(phase='FROZEN_BEFORE_GENERATION', created_at=datetime.datetime.now().astimezone().isoformat(),
    purpose='One additional fixed seed for selected prompts; not new independent sources',
    planned_requests=6, planned_dit=48, cases=cases, seed=seed,
    noise_path=str(noise_path.relative_to(root)), noise_sha256=common.sha(noise_path),
    runner_sha256=common.sha(root/'scripts/run_object_selectivity.py'),
    quality_gate=dict(semantic_required='3/3 at both seeds for F and FIX32',
                      image_MAE_vs_F_max=.015, outside_fixed_box_MAE_vs_F_max=.01),
    independent_validation=False, original_failed_gate_preserved=True)
common.write(out, spec)
print(json.dumps(dict(protocol=str(out), noise_sha256=spec['noise_sha256'])))
