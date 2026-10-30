"""TP1/SP1, unquantized adapter bridge; native SGLang integration still pending.

SGLang parallel linears return (output, deferred_bias). Keep that convention,
preserve FP32 unmerged deltas, and allow the teacher/full steps to bypass them.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class TupleStudentLinear(nn.Module):
    def __init__(self, base, lora_A, lora_B):
        super().__init__()
        if base.weight.ndim != 2 or tuple(base.weight.shape) != (lora_B.shape[0], lora_A.shape[1]):
            raise ValueError('Full, unsharded linear weight shape required')
        if lora_A.shape[0] != lora_B.shape[1]:
            raise ValueError('Adapter rank mismatch')
        self.base = base
        self.register_buffer('lora_A', lora_A.detach().to(device=base.weight.device, dtype=torch.float32).clone())
        self.register_buffer('lora_B', lora_B.detach().to(device=base.weight.device, dtype=torch.float32).clone())
        self.enabled = True

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise ValueError('Quantized tuple activations are outside the validated bridge scope')
        result = self.base(x)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError('SGLang-style (output, bias) result required')
        output, bias = result
        if self.enabled:
            output = output + F.linear(F.linear(x.float(), self.lora_A), self.lora_B).to(output.dtype)
        return output, bias


class SGLangStudentAdapters:
    def __init__(self, model, state, *, tp_size, sp_size, quantized=False):
        if (tp_size, sp_size) != (1, 1) or quantized:
            raise ValueError('Bridge supports TP1/SP1 unquantized weights only')
        expected = {}
        for i in range(60):
            expected[f'transformer_blocks.{i}.attn.to_out.0'] = (3072, 3072)
            if (i+1) % 5 == 0:
                expected[f'transformer_blocks.{i}.img_mlp.net.2'] = (3072, 12288)
        if set(state) != {n+'.'+s for n in expected for s in ('lora_A', 'lora_B')}:
            raise ValueError('Expected exactly 72 original-Qwen LOCAL128 adapters')
        # Validate every destination before mutating the model.
        mounts = []
        for name, shape in expected.items():
            parent_name, leaf = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            base = getattr(parent, leaf)
            if isinstance(base, TupleStudentLinear) or tuple(base.weight.shape) != shape:
                raise ValueError('Unexpected or already wrapped destination: '+name)
            a,b=state[name+'.lora_A'],state[name+'.lora_B']
            if tuple(a.shape)!=(8,shape[1]) or tuple(b.shape)!=(shape[0],8):
                raise ValueError('Unexpected rank or shape: '+name)
            mounts.append((name,parent,leaf,base,a,b))
        self.layers = {}
        for name,parent,leaf,base,a,b in mounts:
            layer=TupleStudentLinear(base,a,b);setattr(parent,leaf,layer);self.layers[name]=layer

    @contextmanager
    def mode(self, enabled):
        previous=[x.enabled for x in self.layers.values()]
        try:
            for x in self.layers.values():x.enabled=enabled
            yield
        finally:
            for x,old in zip(self.layers.values(),previous):x.enabled=old
