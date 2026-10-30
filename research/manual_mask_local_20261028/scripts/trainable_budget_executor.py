import functools

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import (
    apply_rotary_emb_qwen, compute_text_seq_len_from_mask, dispatch_attention_fn,
)

from budget_cache_executor import BudgetCacheExecutor


def run_cached_block(block, image, text, temb, old_key, old_value, *, image_ids,
                     image_rope, text_rope, length, total, key_mask, text_fresh):
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
    if old_key is None:
        shape = (1, total, ik.shape[2], ik.shape[3])
        old_key = torch.empty(shape, device=ik.device, dtype=ik.dtype)
        old_value = torch.empty_like(old_key)
    if torch.is_grad_enabled():
        key = old_key.index_copy(1, image_ids, ik)
        value = old_value.index_copy(1, image_ids, iv)
    else:
        key, value = old_key, old_value
        key.index_copy_(1, image_ids, ik)
        value.index_copy_(1, image_ids, iv)
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
        tq = apply_rotary_emb_qwen(tq, text_rope, use_real=False)
        tk = apply_rotary_emb_qwen(tk, text_rope, use_real=False)
        if torch.is_grad_enabled():
            key = torch.cat([tk, key[:, length:]], dim=1)
            value = torch.cat([tv, value[:, length:]], dim=1)
        else:
            key[:, :length].copy_(tk)
            value[:, :length].copy_(tv)
        query = torch.cat([tq, iq], dim=1)
    else:
        query = iq
    joint = dispatch_attention_fn(query, key, value, attn_mask=key_mask, dropout_p=0.0,
        is_causal=False, backend=attn.processor._attention_backend, parallel_config=None)
    joint = joint.flatten(2, 3).to(query.dtype)
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
    return image, text, key.detach(), value.detach()


class TrainableBudgetExecutor(BudgetCacheExecutor):
    def __init__(self, model, plan, mode="all", spatial=None, checkpoint_blocks=True):
        super().__init__(model, plan, mode="all" if mode == "fresh" else mode)
        self.fresh_context = mode == "fresh"
        self.spatial = spatial
        self.checkpoint_blocks = checkpoint_blocks
        self.last_active = None
        self.last_raw_velocity = None
        self.last_residual = None

    def detach_window(self):
        self.layer_cache = {k: tuple(x.detach() for x in kv) for k, kv in self.layer_cache.items()}
        if self.velocity is not None:
            self.velocity = self.velocity.detach()

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None,
                timestep=None, img_shapes=None, guidance=None, attention_kwargs=None,
                controlnet_block_samples=None, additional_t_cond=None, return_dict=True):
        model, n = self.model, self.plan["target_count"]
        if self.step >= 8 or hidden_states.shape[0] != 1:
            raise ValueError("Invalid training request")
        if model.zero_cond_t or guidance is not None or controlnet_block_samples is not None or additional_t_cond is not None or attention_kwargs:
            raise ValueError("Unsupported training options")
        c = hidden_states.shape[1] - n
        if c != n or hidden_states.shape[-1] != self.plan['packed_channels']:
            raise ValueError("Expected aligned source and target token grids")
        anchor = self.step in self.full_steps
        text_fresh = anchor or self.mode == 'source' or self.fresh_context
        condition_fresh = anchor or self.fresh_context
        selected = self.selected[self.step]
        image_ids = torch.cat([selected, torch.arange(n, n + c, device=selected.device)]) if condition_fresh else selected
        image = model.img_in(hidden_states.index_select(1, image_ids))
        text = model.txt_in(model.txt_norm(encoder_hidden_states)) if text_fresh else None
        if self.cached_geometry is None:
            length, _, mask = compute_text_seq_len_from_mask(text, encoder_hidden_states_mask)
            rope = model.pos_embed(img_shapes, max_txt_seq_len=length, device=image.device)
            key_mask = None
            if mask is not None:
                key_mask = torch.cat([mask, torch.ones((1, n + c), dtype=torch.bool, device=image.device)], dim=1)[:, None, None]
            self.cached_geometry = length, rope, key_mask
        length, rope, key_mask = self.cached_geometry
        image_rope = rope[0].index_select(0, image_ids)
        temb = model.time_text_embed(timestep.to(image.dtype), image, additional_t_cond)
        cache_ids = length + image_ids
        for layer_index, block in enumerate(model.transformer_blocks):
            old_key, old_value = self.layer_cache.get(layer_index, (None, None))
            call = functools.partial(run_cached_block, block, image_ids=cache_ids,
                image_rope=image_rope, text_rope=rope[1], length=length, total=length + n + c,
                key_mask=key_mask, text_fresh=text_fresh)
            arguments = (image, text, temb, old_key, old_value)
            if torch.is_grad_enabled() and self.checkpoint_blocks:
                image, text, key, value = checkpoint(call, *arguments, use_reentrant=False, preserve_rng_state=False)
            else:
                image, text, key, value = call(*arguments)
            self.layer_cache[layer_index] = (key, value)
        fresh = model.proj_out(model.norm_out(image, temb))
        previous = self.velocity
        if previous is None:
            previous = torch.zeros((1, n + c, fresh.shape[-1]), device=fresh.device, dtype=fresh.dtype)
        raw = previous.index_copy(1, image_ids, fresh)
        active = torch.zeros((1, n, 1), device=fresh.device)
        active[:, selected] = 1
        ages = torch.tensor([self.step - stamp for stamp in self.target_stamps], device=fresh.device).reshape(1, n, 1)
        self.velocity = raw
        self.last_raw_velocity = raw[:, :n]
        self.last_active = active
        self.last_residual = None
        if self.spatial is not None and not anchor:
            delta = (raw[:, :n].float() - previous[:, :n].float()) * active
            core = torch.zeros_like(active)
            core[:, self.plan['core_indices']] = 1
            valid = torch.ones(encoder_hidden_states.shape[:2], device=fresh.device) if encoder_hidden_states_mask is None else encoder_hidden_states_mask.float()
            pooled = (encoder_hidden_states.float() * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True)
            text_pool = F.adaptive_avg_pool1d(pooled[:, None], 64)[:, 0]
            residual = self.spatial(hidden_states[:, :n], hidden_states[:, n:], raw[:, :n],
                delta, active, ages, timestep, text_pool, core, self.plan['grid_hw'])
            self.last_residual = residual
            corrected = (raw[:, :n].float() + residual).to(raw.dtype)
            output = torch.cat([corrected, raw[:, n:]], dim=1)
        else:
            output = raw
        for i in self.plan['selected_indices_per_step'][self.step]:
            self.target_stamps[i] = self.step
        if condition_fresh:
            self.context_stamp = self.step
        self.trace.append(dict(step=self.step, anchor=anchor, selected=selected.numel(),
                               condition_fresh=condition_fresh, text_fresh=text_fresh))
        self.step += 1
        return Transformer2DModelOutput(sample=output) if return_dict else (output,)
