import copy
import math
from contextlib import contextmanager

import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import (
    apply_rotary_emb_qwen, compute_text_seq_len_from_mask, dispatch_attention_fn,
)
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit as native_module

from budget_profiles import validate_budget_plan


class BudgetCacheExecutor:
    def __init__(self, model, plan, mode="all"):
        if mode not in ("source", "all"):
            raise ValueError("Unknown context cache mode")
        validate_budget_plan(plan)
        self.model, self.plan, self.mode = model, copy.deepcopy(plan), mode
        self.step = 0
        self.layer_cache = {}
        self.velocity = None
        self.trace = []
        self.state_spec = None
        self.cached_geometry = None
        self.context_stamp = -1
        self.target_stamps = [-1] * plan["target_count"]
        device = next(model.parameters()).device
        self.selected = [torch.tensor(ids, dtype=torch.long, device=device)
                         for ids in plan["selected_indices_per_step"]]
        self.full_steps = set(plan["full_steps_zero_based"])

    def cache_bytes(self):
        size = sum(x.numel() * x.element_size() for kv in self.layer_cache.values() for x in kv)
        return size + (0 if self.velocity is None else self.velocity.numel() * self.velocity.element_size())

    def clear(self):
        self.layer_cache.clear()
        self.velocity = None
        self.cached_geometry = None
        self.selected.clear()

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None,
                timestep=None, img_shapes=None, guidance=None, attention_kwargs=None,
                controlnet_block_samples=None, additional_t_cond=None, return_dict=True):
        model = self.model
        n = self.plan["target_count"]
        if self.step >= 8 or hidden_states.shape[0] != 1 or hidden_states.shape[2] != self.plan["packed_channels"]:
            raise RuntimeError("Invalid request shape or step")
        c = hidden_states.shape[1] - n
        if c <= 0 or math.prod(img_shapes[0][0]) != n or sum(math.prod(s) for s in img_shapes[0][1:]) != c:
            raise ValueError("Unexpected image token geometry")
        if model.zero_cond_t or guidance is not None or controlnet_block_samples is not None or additional_t_cond is not None or attention_kwargs:
            raise ValueError("Unsupported model options")
        spec = (tuple(hidden_states.shape), str(hidden_states.dtype), str(hidden_states.device),
                tuple(encoder_hidden_states.shape), repr(img_shapes), encoder_hidden_states.data_ptr(),
                None if encoder_hidden_states_mask is None else encoder_hidden_states_mask.data_ptr())
        if self.state_spec is not None and spec != self.state_spec:
            raise RuntimeError("Request changed without cache reset")
        self.state_spec = spec
        anchor = self.step in self.full_steps
        text_fresh = anchor or self.mode == "source"
        selected = self.selected[self.step]
        image_ids = torch.cat([selected, torch.arange(n, n + c, device=selected.device)]) if anchor else selected
        image = model.img_in(hidden_states.index_select(1, image_ids))
        text = model.txt_in(model.txt_norm(encoder_hidden_states)) if text_fresh else None
        if self.cached_geometry is None:
            length, _, mask = compute_text_seq_len_from_mask(text, encoder_hidden_states_mask)
            rope = model.pos_embed(img_shapes, max_txt_seq_len=length, device=image.device)
            key_mask = None
            if mask is not None:
                image_mask = torch.ones((1, n + c), dtype=torch.bool, device=image.device)
                key_mask = torch.cat([mask, image_mask], dim=1)[:, None, None, :]
            self.cached_geometry = (length, rope, key_mask)
        length, rope, key_mask = self.cached_geometry
        image_rope = rope[0].index_select(0, image_ids)
        temb = model.time_text_embed(timestep.to(image.dtype), image, additional_t_cond)
        cache_image_ids = length + image_ids
        active = selected.numel()
        if anchor:
            self.context_stamp = self.step
        for i in self.plan["selected_indices_per_step"][self.step]:
            self.target_stamps[i] = self.step
        for layer_index, block in enumerate(model.transformer_blocks):
            im1, im2 = block.img_mod(temb).chunk(2, dim=-1)
            im, ig1 = block._modulate(block.img_norm1(image), im1)
            attn = block.attn
            head_dim = attn.inner_dim // attn.heads
            iq = attn.to_q(im).unflatten(-1, (-1, head_dim))
            ik = attn.to_k(im).unflatten(-1, (-1, head_dim))
            iv = attn.to_v(im).unflatten(-1, (-1, head_dim))
            if attn.norm_q is not None:
                iq = attn.norm_q(iq)
            if attn.norm_k is not None:
                ik = attn.norm_k(ik)
            iq = apply_rotary_emb_qwen(iq, image_rope, use_real=False)
            ik = apply_rotary_emb_qwen(ik, image_rope, use_real=False)
            if layer_index not in self.layer_cache:
                if not anchor:
                    raise RuntimeError("Missing anchor cache")
                shape = (1, length + n + c, ik.shape[2], ik.shape[3])
                self.layer_cache[layer_index] = (torch.empty(shape, device=ik.device, dtype=ik.dtype),
                                                 torch.empty(shape, device=iv.device, dtype=iv.dtype))
            ck, cv = self.layer_cache[layer_index]
            ck.index_copy_(1, cache_image_ids, ik)
            cv.index_copy_(1, cache_image_ids, iv)
            if text_fresh:
                tm1, tm2 = block.txt_mod(temb).chunk(2, dim=-1)
                tx, tg1 = block._modulate(block.txt_norm1(text), tm1)
                tq = attn.add_q_proj(tx).unflatten(-1, (-1, head_dim))
                tk = attn.add_k_proj(tx).unflatten(-1, (-1, head_dim))
                tv = attn.add_v_proj(tx).unflatten(-1, (-1, head_dim))
                if attn.norm_added_q is not None:
                    tq = attn.norm_added_q(tq)
                if attn.norm_added_k is not None:
                    tk = attn.norm_added_k(tk)
                tq = apply_rotary_emb_qwen(tq, rope[1], use_real=False)
                tk = apply_rotary_emb_qwen(tk, rope[1], use_real=False)
                ck[:, :length].copy_(tk)
                cv[:, :length].copy_(tv)
                query = torch.cat([tq, iq], dim=1)
            else:
                query = iq
            joint = dispatch_attention_fn(query, ck, cv, attn_mask=key_mask, dropout_p=0.0,
                is_causal=False, backend=attn.processor._attention_backend,
                parallel_config=None).flatten(2, 3).to(query.dtype)
            image_attention = joint[:, length:] if text_fresh else joint
            ia = attn.to_out[0](image_attention.contiguous())
            if len(attn.to_out) > 1:
                ia = attn.to_out[1](ia)
            image = image + ig1 * ia
            im2v, ig2 = block._modulate(block.img_norm2(image), im2)
            image = image + ig2 * block.img_mlp(im2v)
            if text_fresh:
                text = text + tg1 * attn.to_add_out(joint[:, :length].contiguous())
                tx2v, tg2 = block._modulate(block.txt_norm2(text), tm2)
                text = text + tg2 * block.txt_mlp(tx2v)
            if layer_index in (0, len(model.transformer_blocks) // 2, len(model.transformer_blocks) - 1):
                self.trace.append(dict(step=self.step + 1, layer=layer_index, anchor=anchor,
                    selected_target=active, fresh_condition=c if anchor else 0,
                    fresh_text=length if text_fresh else 0, text_tokens=length,
                    image_qkv_tokens=im.shape[1], image_mlp_tokens=im2v.shape[1],
                    joint_query_tokens=query.shape[1], joint_kv_tokens=ck.shape[1],
                    context_age=self.step - self.context_stamp,
                    max_target_age=max(self.step - stamp for stamp in self.target_stamps)))
        output = model.proj_out(model.norm_out(image, temb))
        if self.velocity is None:
            self.velocity = torch.empty((1, n + c, output.shape[-1]), dtype=output.dtype, device=output.device)
        self.velocity.index_copy_(1, image_ids, output)
        self.step += 1
        if return_dict:
            return Transformer2DModelOutput(sample=self.velocity)
        return (self.velocity,)


@contextmanager
def install_budget_executor(pipe, plan, mode="all"):
    model = pipe.transformer
    if getattr(model, "_controlled_local_bridge_active", False):
        raise RuntimeError("Nested executor")
    base = BudgetCacheExecutor(model, plan, mode)
    original = model.forward
    model.forward = base.forward
    model._controlled_local_bridge_active = True
    succeeded = False
    try:
        yield base
        if base.step != 8:
            raise RuntimeError("Incomplete request")
        succeeded = True
    finally:
        base.bridge_record = dict(completed=succeeded, steps=base.step, mode=mode,
            trace=copy.deepcopy(base.trace), raw_cache_bytes_before_clear=base.cache_bytes())
        model.forward = original
        model._controlled_local_bridge_active = False
        base.clear()


def budget_request(pipe, image, prompt, noise, plan, mode="all", callback=None,
                   *, output_type="pil", prompt_embeds=None, prompt_embeds_mask=None):
    width, height, _ = native_module.calculate_dimensions(1024 * 1024, image.width / image.height)
    multiple = 2 * pipe.vae_scale_factor
    width, height = width // multiple * multiple, height // multiple * multiple
    if plan["source_wh"] != list(image.size) or plan["grid_hw"] != [height // multiple, width // multiple]:
        raise ValueError("Plan and request geometry differ")
    kwargs = {}
    if callback is not None:
        kwargs.update(callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
    with install_budget_executor(pipe, plan, mode) as executor:
        result = pipe(image=image, prompt=prompt, prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask, true_cfg_scale=1, num_inference_steps=8,
            num_images_per_prompt=1, height=height, width=width, latents=noise.detach().clone(),
            output_type=output_type, **kwargs)
    return result, executor.bridge_record
