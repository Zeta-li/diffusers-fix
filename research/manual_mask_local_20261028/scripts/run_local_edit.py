import argparse
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
import torch

import run_s1 as common
from cache_aware_adapters import CacheAdapters
from run_free_pilot import load_pipeline
from local_edit_runtime import LocalEditRequest, GeometryRefreshSelector, resolve_request, export_backend_plan, install_local_edit


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',type=Path,required=True);p.add_argument('--prompt',required=True)
    p.add_argument('--roi',type=float,nargs=4,required=True);p.add_argument('--profile',choices=['F','B','R4','R25','R6','SUP'],default='R25')
    p.add_argument('--request-key');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=2026102201);p.add_argument('--noise-file',type=Path)
    p.add_argument('--export-plan-only',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    image=Image.open(args.image).convert('RGB')
    if image.width!=image.height:raise ValueError('This validated entry point requires a square source image')
    key=args.request_key or hashlib.sha256(image.tobytes()+args.prompt.encode()+json.dumps(args.roi).encode()).hexdigest()[:24]
    request=LocalEditRequest(key,args.prompt,image.size,tuple(args.roi))
    checkpoint=common.ROOT/'runs/budget_expansion_train_v1/checkpoints/student_inference.pt'
    head=common.ROOT/'runs/refresh_policy_v1/checkpoints/supervised.pt'
    selector=GeometryRefreshSelector(head) if args.profile=='SUP' else None
    plan,meta=resolve_request(request,args.profile,selector)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.export_plan_only:
        common.write(args.output,export_backend_plan(plan,meta));print(str(args.output));return
    if args.output.suffix.lower()!='.png':raise ValueError('Use a PNG output path')
    if args.output.with_suffix('.json').exists():raise FileExistsError(args.output.with_suffix('.json'))
    assert common.sha(checkpoint)=='216794f27aacf696af7835612234dcdf04cb7dcc6dbfbc9ecffcfa3ef55332f7'
    pipe=load_pipeline();pipe.vae.disable_tiling()
    adapters=CacheAdapters(pipe.transformer)
    adapters.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu')['adapters'])
    noise=torch.load(args.noise_file,weights_only=True,map_location='cuda') if args.noise_file else torch.randn((1,4096,64),generator=torch.Generator(device='cpu').manual_seed(args.seed),dtype=torch.bfloat16).to('cuda')
    assert noise.shape==(1,4096,64) and noise.dtype==torch.bfloat16
    torch.cuda.synchronize();start=time.perf_counter()
    with install_local_edit(pipe,adapters,request,args.profile,selector) as record:
        result=pipe(image=image,prompt=args.prompt,true_cfg_scale=1,num_inference_steps=8,height=1024,width=1024,latents=noise)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    output=result.images[0];output.save(args.output)
    common.write(args.output.with_suffix('.json'),dict(record=record,seed=args.seed,noise_file=None if args.noise_file is None else str(args.noise_file),
        model_resident_request_seconds=elapsed,timing_note='single request;not a paired benchmark',RGB_sha256=hashlib.sha256(output.tobytes()).hexdigest()))
    print(str(args.output))


if __name__=='__main__':main()
