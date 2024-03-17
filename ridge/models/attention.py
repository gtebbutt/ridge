from typing import List, Dict, Tuple, Callable, Optional, Union, Any

import torch
from torch import nn
import torch.nn.functional as F

from .embeddings import MultiAxisRotaryPositionEmbed


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        norm_num_groups: Optional[int] = None,
        out_bias: bool = True,
        eps: float = 1e-5,
        use_rotary_pos_embed: bool = False,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.scale = dim_head**-0.5
        self.heads = heads

        if use_rotary_pos_embed:
            self.rotary_emb = MultiAxisRotaryPositionEmbed(embed_dim=self.query_dim // self.heads)
        else:
            self.rotary_emb = None

        if norm_num_groups is not None:
            # DiT only
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.query_dim, bias=out_bias))

        self.processor = RotaryAttnProcessor()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        height_patches: int = None,
        width_patches: int = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        return self.processor(
            attn=self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            height_patches=height_patches,
            width_patches=width_patches,
            **cross_attention_kwargs,
        )


class RotaryAttnProcessor:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("RotaryAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        *,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        height_patches: int = None,
        width_patches: int = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            if attention_mask.shape[0] < batch_size * attn.heads:
                attention_mask = attention_mask.repeat_interleave(attn.heads, dim=0)
            # scaled_dot_product_attention expects attention_mask shape to be (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            # DiT only
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)

        if attn.rotary_emb is not None:
            query, key = attn.rotary_emb(query, key, (height_patches, width_patches))

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)

        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)

        return hidden_states
