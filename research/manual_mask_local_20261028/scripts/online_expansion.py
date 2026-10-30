"""Run the expansion head after two full steps, before any partial execution."""

import copy
import torch
import torch.nn.functional as F
from diffusers import QwenImageEditPipeline
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit as native_module

from free_region import FreeRegionRunRecord, _digest, build_region_plan
from legacy_local_executor import LocalTargetExecutor


@torch.no_grad()
def run_online_region(pipe,image,prompt,initial_noise,case,predictor,trace_latents=True,
                      prompt_embeds=None,prompt_embeds_mask=None):
    if type(pipe) is not QwenImageEditPipeline:
        raise TypeError("native pipeline required")
    if getattr(pipe,"_free_region_request_active",False):
        raise RuntimeError("nested request")
    width,height,_=native_module.calculate_dimensions(1024*1024,image.width/image.height)
    multiple=2*pipe.vae_scale_factor
    width,height=width//multiple*multiple,height//multiple*multiple
    grid=[height//multiple,width//multiple];n=grid[0]*grid[1]
    if initial_noise.shape!=(1,n,pipe.transformer.config.in_channels):
        raise ValueError("noise geometry mismatch")
    if image.mode!="RGB" or list(image.size)!=case["source_wh"]:
        raise ValueError("source geometry mismatch")
    original=pipe.transformer.forward
    base=LocalTargetExecutor(pipe.transformer)
    record=FreeRegionRunRecord(None,height,width,_digest(initial_noise))
    pipe.last_free_region_run=record
    pipe._free_region_request_active=True
    online_info=None
    def dispatch(*args,**fields):
        nonlocal online_info
        k=base.step
        if args or not 0<=k<8:
            raise RuntimeError("invalid request state")
        hidden=fields["hidden_states"]
        if k==0:
            if not torch.equal(hidden[:,:n],initial_noise):
                raise RuntimeError("initial state mismatch")
            base.begin(n,hidden.shape[1]-n,list(range(n)),[0,7],native=False)
            record.sigmas=pipe.scheduler.sigmas.float().cpu().tolist()
            record.reference_condition_sha256=_digest(hidden[:,n:])
            record.text_sha256=_digest(fields["encoder_hidden_states"])
            mask=fields.get("encoder_hidden_states_mask")
            record.text_mask_sha256=None if mask is None else _digest(mask)
        result=base.forward(**fields)
        record.dit_calls+=1
        if k==1:
            clean=hidden[:,:n].float()-pipe.scheduler.sigmas[k]*result[0][:,:n].float()
            text_pool=F.adaptive_avg_pool1d(fields["encoder_hidden_states"].float().mean(1).unsqueeze(1),64).squeeze(1)
            box,fallback=predictor(clean,hidden[:,n:],text_pool,case,grid)
            plan=build_region_plan(case["bbox"],case["source_wh"],grid,case["case_id"],halo=0,
                target_envelope_xyxy=box,kind="all_selected" if fallback else "front",
                packed_channels=initial_noise.shape[-1])
            if plan["target_tokens_per_step"][:2]!=[n,n]:
                raise RuntimeError("head requires exactly two full warmup steps")
            for step,ids in enumerate(plan["selected_indices_per_step"]):
                base.indices_cpu[step]=torch.tensor(ids,dtype=torch.long)
                base.indices[step]=base.indices_cpu[step].to(initial_noise.device)
            record.plan=copy.deepcopy(plan)
            online_info=dict(prediction_step=1,features_from_future=False,
                             prediction=copy.deepcopy(getattr(predictor,"last",None)))
        return result
    def callback(p,index,t,values):
        state=values["latents"]
        if state.shape!=initial_noise.shape or not bool(torch.isfinite(state).all()):
            raise RuntimeError("invalid full scheduler state")
        record.steps.append(dict(step=index,timestep=float(t),state_sha256=_digest(state),
            sigma_current=record.sigmas[index],sigma_next=record.sigmas[index+1],
            all_target_latents_scheduler_updated=True))
        if trace_latents:record.step_latents.append(state.detach().cpu().clone())
        return {}
    pipe.transformer.forward=dispatch
    try:
        result=pipe(image=image,prompt=prompt,prompt_embeds=prompt_embeds,prompt_embeds_mask=prompt_embeds_mask,
            true_cfg_scale=1,num_inference_steps=8,num_images_per_prompt=1,height=height,width=width,
            latents=initial_noise.clone(),output_type="latent",callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"])
        if base.step!=8 or record.dit_calls!=8 or record.plan is None:
            raise RuntimeError("incomplete online selection")
        record.completed=True
        return result
    finally:
        record.executor_record=dict(completed=record.completed,steps=base.step,plan=record.plan,
            trace=copy.deepcopy(base.trace),age_trace=copy.deepcopy(base.age_trace),
            online_head=online_info,raw_cache_bytes_before_clear=base.cache_bytes())
        pipe.transformer.forward=original
        pipe._free_region_request_active=False
        base.clear()
