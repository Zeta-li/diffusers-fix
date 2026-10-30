from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F


class CacheLoRALinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.rank = rank
        self.enabled = True
        self.merged = False
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        base.requires_grad_(False)

    def forward(self, value):
        result = self.base(value)
        if self.enabled and not self.merged:
            result = result + F.linear(F.linear(value.float(), self.lora_A), self.lora_B).to(result.dtype)
        return result

    @torch.no_grad()
    def merge(self):
        if self.merged:
            raise RuntimeError("Adapter already merged")
        self.base.weight.copy_((self.base.weight.float() + self.lora_B @ self.lora_A).to(self.base.weight.dtype))
        self.merged = True


class CacheAdapters:
    def __init__(self, model, rank=8):
        self.model = model
        self.layers = {}
        for index, block in enumerate(model.transformer_blocks):
            path = f"transformer_blocks.{index}.attn.to_out.0"
            block.attn.to_out[0] = CacheLoRALinear(block.attn.to_out[0], rank)
            self.layers[path] = block.attn.to_out[0]
            if (index + 1) % 5 == 0:
                path = f"transformer_blocks.{index}.img_mlp.net.2"
                block.img_mlp.net[2] = CacheLoRALinear(block.img_mlp.net[2], rank)
                self.layers[path] = block.img_mlp.net[2]

    def parameters(self):
        return [p for layer in self.layers.values() for p in (layer.lora_A, layer.lora_B)]

    @contextmanager
    def mode(self, enabled):
        previous = [layer.enabled for layer in self.layers.values()]
        try:
            for layer in self.layers.values():
                if layer.merged and not enabled:
                    raise RuntimeError("Merged student cannot serve as teacher")
                layer.enabled = enabled
            yield
        finally:
            for layer, value in zip(self.layers.values(), previous):
                layer.enabled = value

    def state_dict(self):
        return {name + '.' + key: getattr(layer, key).detach().cpu().clone()
                for name, layer in self.layers.items() for key in ('lora_A', 'lora_B')}

    @torch.no_grad()
    def load_state_dict(self, state):
        expected = {name + '.' + key for name in self.layers for key in ('lora_A', 'lora_B')}
        if set(state) != expected:
            raise ValueError("Adapter checkpoint keys differ")
        for name, layer in self.layers.items():
            for key in ('lora_A', 'lora_B'):
                getattr(layer, key).copy_(state[name + '.' + key])
