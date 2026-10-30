import importlib.metadata
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

torch.set_num_threads(1)
torch.manual_seed(20261028)
original = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False)).to(torch.bfloat16)
model = get_peft_model(original, LoraConfig(r=2, lora_alpha=2, target_modules=["0"], inference_mode=True))
for name, param in model.named_parameters():
    if "lora_B" in name:
        param.data.fill_(0.03125)
value = torch.randn(2, 8, dtype=torch.bfloat16)
with torch.no_grad():
    live = model(value)
    fused = model.merge_and_unload(safe_merge=True)
    merged = fused(value)
assert isinstance(fused[0], torch.nn.Linear)
assert torch.isfinite(live).all() and torch.isfinite(merged).all()
error = (live.float() - merged.float()).abs().max().item()
assert error < 0.02
result = dict(status="PASS", device="cpu", dtype="bfloat16", nonzero_lora_B=True,
              adapter_injection_and_safe_merge_completed=True, max_abs_bf16_merge_difference=error,
              cpu_linear_forward_calls=2, cpu_tiny_dit_calls=0, real_weight_gpu_dit_calls=0,
              packages={n:importlib.metadata.version(n) for n in ["torch","peft","torchao","diffusers"]})
path = Path(__file__).resolve().parents[1] / "data/adapter_environment_check.json"
path.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
