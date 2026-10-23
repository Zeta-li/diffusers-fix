import json
from pathlib import Path

import numpy as np
import torch
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from calibration_collector import CalibrationCollector
from local_executor import LocalTargetExecutor
from moments import save_npz, load_npz


ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(patch_size=2, in_channels=8, out_channels=2, num_layers=2,
    attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24, axes_dims_rope=(4, 6, 6))


def fixture(dtype):
    torch.manual_seed(46103)
    model = QwenImageTransformer2DModel(**CONFIG).to(dtype=dtype).eval()
    model.set_attention_backend('native')
    raw = torch.randn(1, 10, 8).to(dtype)
    text = torch.randn(1, 5, 24).to(dtype)
    mask = torch.tensor([[True, False, True, True, False]])
    sigmas = torch.tensor([1., .9545454545, .9, .8333333333, .75, .6428571429, .5, .3, 0.])
    return model, raw, text, mask, sigmas


def run(dtype):
    model, raw, text, mask, sigmas = fixture(dtype)
    baseline = LocalTargetExecutor(model)
    baseline.begin(6, 4, [1,4], [0,7])
    dense_calls = []
    collector = CalibrationCollector(model)
    original = collector.original_forward
    def counted(*a, **kw):
        dense_calls.append(collector.collect_step)
        return original(*a, **kw)
    collector.original_forward = counted
    samples = collector.begin_collection('tiny_only',6,4,[1,4],sample_count=3)
    assert samples == collector.begin_collection('tiny_only',6,4,[1,4],sample_count=3)
    assert all(not (set(ids) & {1,4}) and len(ids)==3 for ids in samples.values())
    collector.sigmas = sigmas
    z_b, z_c = raw[:,:6].clone(), raw[:,:6].clone()
    expected_h = model.transformer_blocks[0].img_norm1(model.img_in(raw))[0]
    conditions = raw[:,6:]
    raw_predictions = []
    with torch.inference_mode():
        for k in range(8):
            timestep = (sigmas[k:k+1]*1000).to(dtype)/1000
            fields = dict(encoder_hidden_states=text,encoder_hidden_states_mask=mask,
                timestep=timestep,img_shapes=[[(1,2,3),(1,2,2)]],return_dict=True)
            rb = baseline.forward(hidden_states=torch.cat([z_b,conditions],1),**fields).sample[:,:6].clone()
            rc = collector.forward(hidden_states=torch.cat([z_c,conditions],1),**fields).sample[:,:6].clone()
            assert torch.equal(rb,rc),(str(dtype),k,'prediction')
            raw_predictions.append(torch.equal(rb,rc))
            z_b = (z_b.float()+(sigmas[k+1]-sigmas[k])*rb).to(dtype)
            z_c = (z_c.float()+(sigmas[k+1]-sigmas[k])*rc).to(dtype)
            collector.confirm_update(z_c)
            assert torch.equal(z_b,z_c)
    data = collector.moments.as_dict()
    assert len(data['a']) == 2 and len(data['b']) == 12
    selected = expected_h.index_select(0,torch.tensor(samples['0'])).detach().double().numpy()
    np.testing.assert_allclose(data['a']['0:0']['sum_h'],selected.sum(0),rtol=0,atol=0)
    np.testing.assert_allclose(data['a']['0:0']['sum_h2'],(selected*selected).sum(0),rtol=0,atol=0)
    assert dense_calls == list(range(1,7))
    assert collector.local_calls == collector.local_completed == 8
    assert collector.oracle_calls == collector.oracle_completed == 6
    assert all(r['student_only_euler_bitwise_equal'] for r in collector.records)
    assert all(r.get('velocity_age_rng_unchanged') for r in collector.records[1:7])
    output = ROOT/'data'/f'tiny_moments_{str(dtype).split(".")[-1]}.npz'
    save_npz(output,data)
    assert len(load_npz(output)['b'])==12
    collector.close()
    return dict(dtype=str(dtype),status='pass',tiny_local_forwards=16,tiny_dense_forwards=6,
        tiny_total_forwards=22,real_pretrained_forwards=0,student_prediction_bitwise_equal=all(raw_predictions),
        local_steps=8,same_state_shadow_steps=dense_calls,norm1_pre_modulation_checked=True,
        moments_a_records=len(data['a']),moments_b_records=len(data['b']),sample_indices=samples)


def main():
    torch.set_num_threads(4)
    (ROOT/'data').mkdir(exist_ok=True)
    rows=[run(torch.float32),run(torch.bfloat16)]
    result=dict(status='pass',checks=rows,tiny_model_forwards=sum(r['tiny_total_forwards'] for r in rows),
        real_pretrained_forwards=0,scope='Synthetic randomly initialized two-layer Qwen model on CPU; not Train8 or image-quality evidence')
    (ROOT/'data/collector_tiny_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
