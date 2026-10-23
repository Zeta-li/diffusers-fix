import math

import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import (
    apply_rotary_emb_qwen,
    compute_text_seq_len_from_mask,
    dispatch_attention_fn,
)


class LocalTargetExecutor:
    def __init__(self, model):
        self.model = model
        self.original_forward = model.forward
        self.native = True
        self.clear()

    def install(self):
        self.model.forward = self.forward

    def clear(self):
        self.layer_cache = {}
        self.velocity = None
        self.step = 0
        self.trace = []
        self.indices = {}
        self.indices_cpu = {}
        self.state_spec = None
        self.age = None

    def begin(self, target_count, condition_count, indices, full_steps, native=False, reverse_full=False, per_step_indices=None):
        self.clear()
        self.native = native
        self.target_count = target_count
        self.condition_count = condition_count
        self.full_steps = set(full_steps)
        self.reverse_full = reverse_full
        device = next(self.model.parameters()).device
        self.all_targets_cpu = torch.arange(target_count)
        self.all_targets = self.all_targets_cpu.to(device=device)
        if reverse_full:
            self.all_targets = self.all_targets.flip(0)
            self.all_targets_cpu = self.all_targets_cpu.flip(0)
        self.conditions = torch.arange(target_count, target_count + condition_count, device=device)
        mapping = {step: indices for step in range(8)} if per_step_indices is None else per_step_indices
        for step, value in mapping.items():
            value = torch.as_tensor(value, dtype=torch.long)
            if value.numel() == 0 or value.numel() != torch.unique(value).numel() or value.min() < 0 or value.max() >= target_count:
                raise ValueError("Invalid target indices")
            self.indices_cpu[step] = value.cpu()
            self.indices[step] = value.to(device=device)
        self.age = torch.full((target_count,), -1, dtype=torch.long)

    def cache_bytes(self):
        total = sum(tensor.numel() * tensor.element_size() for pair in self.layer_cache.values() for tensor in pair)
        if self.velocity is not None:
            total += self.velocity.numel() * self.velocity.element_size()
        return total

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None, timestep=None,
                img_shapes=None, guidance=None, attention_kwargs=None, controlnet_block_samples=None,
                additional_t_cond=None, return_dict=True):
        if self.native:
            self.step += 1
            return self.original_forward(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask, timestep=timestep, img_shapes=img_shapes,
                guidance=guidance, attention_kwargs=attention_kwargs, controlnet_block_samples=controlnet_block_samples,
                additional_t_cond=additional_t_cond, return_dict=return_dict)
        model = self.model
        if self.step >= 8:
            raise RuntimeError("New requests must call begin to reset caches")
        spec = (tuple(hidden_states.shape), str(hidden_states.dtype), str(hidden_states.device), tuple(encoder_hidden_states.shape), repr(img_shapes), encoder_hidden_states.data_ptr(), encoder_hidden_states_mask.data_ptr() if encoder_hidden_states_mask is not None else None)
        if self.state_spec is not None and spec != self.state_spec:
            raise RuntimeError("Request geometry or state changed without cache reset")
        self.state_spec = spec
        if model.zero_cond_t or guidance is not None or controlnet_block_samples is not None or additional_t_cond is not None or attention_kwargs:
            raise ValueError("Unsupported model options in local pilot")
        if hidden_states.shape[0] != 1 or hidden_states.shape[1] != self.target_count + self.condition_count:
            raise ValueError("Local pilot requires fixed batch-one image geometry")
        if math.prod(img_shapes[0][0]) != self.target_count or sum(math.prod(s) for s in img_shapes[0][1:]) != self.condition_count:
            raise ValueError("Image geometry disagrees with target/condition split")
        selected = self.all_targets if self.step in self.full_steps else self.indices[self.step]
        if self.step == 0 and selected.numel() != self.target_count:
            raise RuntimeError("First step must initialize all target caches")
        image_indices = torch.cat([selected, self.conditions])
        active = selected.numel()
        image = model.img_in(hidden_states.index_select(1, image_indices))
        text = model.txt_in(model.txt_norm(encoder_hidden_states))
        text_length, _, mask = compute_text_seq_len_from_mask(text, encoder_hidden_states_mask)
        temb = model.time_text_embed(timestep.to(image.dtype), image, additional_t_cond)
        full_rope = model.pos_embed(img_shapes, max_txt_seq_len=text_length, device=image.device)
        image_rope = full_rope[0].index_select(0, image_indices)
        text_rope = full_rope[1]
        key_mask = None
        if mask is not None:
            image_mask = torch.ones((1, self.target_count + self.condition_count), dtype=torch.bool, device=image.device)
            key_mask = torch.cat([mask, image_mask], dim=1)[:, None, None, :]
        selected_cpu = self.all_targets_cpu if self.step in self.full_steps else self.indices_cpu[self.step]
        previous_age = self.age.index_select(0, selected_cpu)
        newly_activated = int(((previous_age < self.step - 1) & (previous_age >= 0)).sum().item()) if self.step else 0
        for layer_index, block in enumerate(model.transformer_blocks):
            im1, im2 = block.img_mod(temb).chunk(2, dim=-1)
            tm1, tm2 = block.txt_mod(temb).chunk(2, dim=-1)
            im, ig1 = block._modulate(block.img_norm1(image), im1)
            tx, tg1 = block._modulate(block.txt_norm1(text), tm1)
            attn = block.attn
            head_dim = attn.inner_dim // attn.heads
            iq = attn.to_q(im).unflatten(-1, (-1, head_dim))
            ik = attn.to_k(im).unflatten(-1, (-1, head_dim))
            iv = attn.to_v(im).unflatten(-1, (-1, head_dim))
            tq = attn.add_q_proj(tx).unflatten(-1, (-1, head_dim))
            tk = attn.add_k_proj(tx).unflatten(-1, (-1, head_dim))
            tv = attn.add_v_proj(tx).unflatten(-1, (-1, head_dim))
            if attn.norm_q is not None:
                iq = attn.norm_q(iq)
            if attn.norm_k is not None:
                ik = attn.norm_k(ik)
            if attn.norm_added_q is not None:
                tq = attn.norm_added_q(tq)
            if attn.norm_added_k is not None:
                tk = attn.norm_added_k(tk)
            iq = apply_rotary_emb_qwen(iq, image_rope, use_real=False)
            ik = apply_rotary_emb_qwen(ik, image_rope, use_real=False)
            tq = apply_rotary_emb_qwen(tq, text_rope, use_real=False)
            tk = apply_rotary_emb_qwen(tk, text_rope, use_real=False)
            if layer_index not in self.layer_cache:
                if self.step > 0 and active != self.target_count:
                    raise RuntimeError("Missing initialized layer cache on partial step")
                shape = (1, self.target_count, ik.shape[2], ik.shape[3])
                self.layer_cache[layer_index] = (torch.empty(shape, device=ik.device, dtype=ik.dtype), torch.empty(shape, device=iv.device, dtype=iv.dtype))
            ck, cv = self.layer_cache[layer_index]
            ck.index_copy_(1, selected, ik[:, :active])
            cv.index_copy_(1, selected, iv[:, :active])
            query = torch.cat([tq, iq], dim=1)
            key = torch.cat([tk, ck, ik[:, active:]], dim=1)
            value = torch.cat([tv, cv, iv[:, active:]], dim=1)
            joint = dispatch_attention_fn(query, key, value, attn_mask=key_mask, dropout_p=0.0,
                is_causal=False, backend=attn.processor._attention_backend, parallel_config=None).flatten(2, 3).to(query.dtype)
            ia = attn.to_out[0](joint[:, text_length:].contiguous())
            if len(attn.to_out) > 1:
                ia = attn.to_out[1](ia)
            ta = attn.to_add_out(joint[:, :text_length].contiguous())
            image = image + ig1 * ia
            text = text + tg1 * ta
            im2v, ig2 = block._modulate(block.img_norm2(image), im2)
            tx2v, tg2 = block._modulate(block.txt_norm2(text), tm2)
            image = image + ig2 * block.img_mlp(im2v)
            text = text + tg2 * block.txt_mlp(tx2v)
            if layer_index in (0, len(model.transformer_blocks) // 2, len(model.transformer_blocks) - 1):
                self.trace.append(dict(step=self.step + 1, layer=layer_index, selected_target=active,
                    condition=self.condition_count, image_qkv_tokens=im.shape[1], image_mlp_tokens=im2v.shape[1],
                    joint_query_tokens=query.shape[1], joint_kv_tokens=key.shape[1], text_tokens=text_length,
                    cache_kv_bytes=self.cache_bytes(), newly_activated_from_current_latent=newly_activated))
        output = model.proj_out(model.norm_out(image, temb))
        if self.velocity is None:
            self.velocity = torch.empty((1, self.target_count + self.condition_count, output.shape[-1]), dtype=output.dtype, device=output.device)
        self.velocity.index_copy_(1, image_indices, output)
        self.age.index_fill_(0, selected_cpu, self.step)
        self.step += 1
        if return_dict:
            return Transformer2DModelOutput(sample=self.velocity)
        return (self.velocity,)
