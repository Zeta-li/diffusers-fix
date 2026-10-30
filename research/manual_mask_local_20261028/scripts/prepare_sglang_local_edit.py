"""Export a reversible bridge package and verify CPU arithmetic. No GPU DiT."""
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file, load_file

import run_s1 as common
from sglang_student_adapter import TupleStudentLinear

ROOT=common.ROOT
PIN='227189488449d97d89d8da2fd5c1ee4bf1e2f955'


class TupleLinear(nn.Module):
    def __init__(self, deferred):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(12,16,dtype=torch.bfloat16))
        self.bias=nn.Parameter(torch.randn(12,dtype=torch.bfloat16))
        self.deferred=deferred

    def forward(self,x):
        return F.linear(x,self.weight,None if self.deferred else self.bias), self.bias if self.deferred else None


def main():
    torch.set_num_threads(1);torch.manual_seed(2026103041)
    checkout=Path('/workspace/sglang-localedit')
    head=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    assert head==PIN
    ckpt=ROOT/'runs/budget_expansion_train_v1/checkpoints/student_inference.pt'
    assert common.sha(ckpt)=='216794f27aacf696af7835612234dcdf04cb7dcc6dbfbc9ecffcfa3ef55332f7'
    state=torch.load(ckpt,weights_only=True,map_location='cpu')['adapters']
    assert len(state)==144 and all(x.dtype==torch.float32 for x in state.values())
    destination=ROOT/'exports/sglang_local_edit_v1';assert not destination.exists();destination.mkdir(parents=True)
    weights=destination/'student_unmerged_fp32.safetensors'
    save_file({k:v.contiguous() for k,v in state.items()},str(weights),metadata={'base':'original Qwen-Image-Edit','rank':'8','alpha_effective':'8','merge':'forbidden; full steps disable student'})
    restored=load_file(str(weights));assert set(restored)==set(state) and all(torch.equal(v,restored[k]) for k,v in state.items())
    records=[]
    for name in sorted(k[:-7] for k in state if k.endswith('.lora_A')):
        a,b=state[name+'.lora_A'],state[name+'.lora_B']
        assert a.shape[0]==b.shape[1]==8 and b.shape[0]==3072
        assert a.shape[1]==(12288 if '.img_mlp.' in name else 3072)
        records.append(dict(module=name,A_shape=list(a.shape),B_shape=list(b.shape),dtype='float32'))
    assert len(records)==72 and sum(x.numel() for x in state.values())==4423680
    checks=[]
    for deferred in (False,True):
        base=TupleLinear(deferred);a=torch.randn(8,16);b=torch.randn(12,8);wrapper=TupleStudentLinear(base,a,b)
        x=torch.randn(1,7,16,dtype=torch.bfloat16)
        y,bias=base(x);expected=y+F.linear(F.linear(x.float(),a),b).to(y.dtype)
        actual,actual_bias=wrapper(x);assert torch.equal(actual,expected) and actual_bias is bias
        wrapper.enabled=False;actual,actual_bias=wrapper(x);assert torch.equal(actual,y) and actual_bias is bias
        checks.append(dict(deferred_bias=deferred,enabled_exact=True,disabled_exact=True,bias_identity=True))
    files=['runtime/models/dits/qwen_image.py','runtime/pipelines/qwen_image.py','configs/models/dits/qwenimage.py']
    manifest=dict(status='EXPORT_AND_CPU_ARITHMETIC_PASS_NATIVE_INTEGRATION_PENDING',sglang_commit=head,
        sglang_source_sha256={p:common.sha(checkout/'python/sglang/multimodal_gen'/p) for p in files},
        checkpoint_sha256=common.sha(ckpt),export_sha256=common.sha(weights),export_tensors=144,export_parameters=4423680,
        rank=8,adapter_scale=1,unmerged=True,layers=records,CPU_checks=checks,
        scheduler_contract='reuse frozen sigmas, mu=ln(3); native default linear shift differs',
        timestep_contract='native model divides raw timestep by1000; do not apply Diffusers caller scaling twice',
        restrictions=['TP1/SP1','BF16 base,FP32 delta','no quantization','no graph/fused norm-RoPE initially','no upstream cache combined initially'],
        native_model_import_tested=False,native_full_generation_tested=False,native_sparse_executor_implemented=False,
        GPU_DiT=0,training_updates=0)
    common.write(destination/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('layers','sglang_source_sha256')},indent=2))


if __name__=='__main__':main()
