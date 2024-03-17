import math
from functools import reduce
from typing import List, Dict, Tuple, Callable, Optional, Union, Any

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from einops import rearrange
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings, PixArtAlphaCombinedTimestepSizeEmbeddings, get_2d_sincos_pos_embed


# Diffusers combines the patchifying and position embedding into the PatchEmbed class - they're separated here so they can be used independently
class PatchEmbed(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )

    def forward(self, latent):
        latent = self.proj(latent)
        latent = latent.flatten(2).transpose(1, 2)  # (b c h w) -> (b n c)
        return latent


# This is very similar to the diffusers PatchEmbed, to retain backwards compatibility with vanilla DiT and Pixart models; it's split out as a separate class because it isn't used in Ridge models
class PositionEmbed(nn.Module):
    def __init__(
        self,
        *,
        # These aren't actual output height and width, they're the model's nominal values assuming square output - the output values are worked out based on the latent shape in the forward function
        height: int,
        width: int,
        patch_size: int,
        embed_dim: int,
        interpolation_scale: float = 1.0,
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale
        self.embed_dim = embed_dim
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, int(num_patches**0.5), base_size=self.base_size, interpolation_scale=self.interpolation_scale
        )
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

    # Strength parameter is used by the training script to incrementally reduce the impact of the fixed position embedding when converting existing weights to rotary embedding
    def forward(
        self,
        latent: torch.Tensor,
        # Important to pass height and width explicitly, unlike the diffusers version, because the input has already been flattened to a sequence here
        height: int,
        width: int,
        strength: float = 1.0
    ):
        if not (0.0 <= strength <= 1.0):
            raise ValueError(f"Invalid PositionEmbed strength {strength}, values must be between 0 and 1")
        
        if strength == 0:
            return latent

        # See diffusers implementation and/or Pixart paper for info on this
        if self.height != height or self.width != width:
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=self.embed_dim,
                grid_size=(height, width),
                base_size=self.base_size,
                interpolation_scale=self.interpolation_scale,
            )
            pos_embed = torch.from_numpy(pos_embed)
            pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
        else:
            pos_embed = self.pos_embed

        return (latent + (pos_embed * strength)).to(latent.dtype)


def multiply_as_complex(a, b):
    # Elementwise multiplication of complex numbers represented using an extra dimension, to avoid breaking torch.compile by using actual complex dtypes. Assumes final dimension of tensor is length 2 with format (real, imaginary)
    if a.shape[-1] != b.shape[-1] or a.shape[-1] != 2:
        raise ValueError(f"Expected both tensors to have (real, complex) as their final dimension, but got shapes {a.shape} and {b.shape}")

    return torch.stack([
        (a[..., -1] * b[..., -1]) - (a[..., -2] * b[..., -2]),
        (a[..., -2] * b[..., -1]) + (a[..., -1] * b[..., -2])
    ], dim=-1)


# Generalised rotary position embedding in N axes, extended from the original single-axis RoPE https://arxiv.org/abs/2104.09864 and the two axis version proposed in FiT https://arxiv.org/abs/2402.12376
# NB: "Dimensions" as used in the RoFormer paper refers to the embedding dimension, whereas FiT uses "1D RoPE" and "2D RoPE" to refer to the number of spatial dimensions they can keep track of; this class uses "dimension" to refer exclusively to embed_dim, and axes to refer to the input shape (one axis for linear, two for area, three for volumetric, etc.)
class MultiAxisRotaryPositionEmbed(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        axes: int = 2,
        theta: float = 10000.0,
    ):
        super().__init__()

        # embed_dim will be equal to the attention_head_dim used in the transformer config
        self.embed_dim = embed_dim
        self.axes = axes
        self.theta = theta

    # Linear implementations generally cache this in a buffer, but doing so for more axes triggers torch recompilation every time the shape changes, and adds potentially significant memory overhead unless you know up front you're only using a single aspect ratio
    def calculate_freqs(self, shape: List[int]):
        if not len(shape) == self.axes:
            raise ValueError(f"Expected a {self.axes}-axis shape, but got {shape}")

        # One tensor for each axis, each containing the cartesian coordinates on that axis for every point in the space; meshgrid returns int64 coordinates by default
        axis_positions = torch.meshgrid([torch.arange(l) for l in shape], indexing="ij")

        # Explicitly using fp32 here, since autocast is disabled in the forward function
        # Factor of two per axis to account for the fact we'll be repeating this for each axis, and then using the overall result twice to get the sin and cos components in the forward function
        base_freqs = 1.0 / (self.theta ** (torch.arange(0, self.embed_dim, 2 * self.axes).to(dtype=torch.float32) / self.embed_dim))

        freqs = [torch.outer(pos.flatten().to(torch.float32), base_freqs) for pos in reversed(axis_positions)]
        
        # Einops accepts a list of tensors and treats it as if the values were already stacked into a single tensor, which makes things easier here
        # Using torch.cat(..., dim=-1) directly would be equivalent to "a n d -> n (a d)", which gives the shape we want but not the correct ordering of elements within the tensor; the alternative is to stack along a new final dimension and then reshape, but rearrange makes things more readable. Ordering matters here because we're emulating the way torch handles complex polar values
        # Returns shape (n, self.embed_dim / 2)
        return rearrange(freqs, "a n d -> n (d a)")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        shape: List[int],
    ):
        # Query and key shape are both (b n h d) where h is num_attention_heads and d is attention_head_dim (which should be equal to embed_dim); for DiT and Pixart models num_attention_heads=16 and attention_head_dim=72
        batch_size = query.shape[0]
        sequence_length = query.shape[1]
        input_dtype = query.dtype

        # Always run these calculations in fp32, even if autocast is enabled - longer sequences get unstable if the sin and cos values are handled in reduced precision
        with torch.cuda.amp.autocast(enabled=False):
            freqs = self.calculate_freqs(shape) # (n d/2)

            # Equivalent to torch.polar but using the last dimension for (real, imaginary) rather than using complex dtypes, as they aren't compatible with torch.compile
            freqs = torch.stack([freqs.cos(), freqs.sin()], dim=-1)

            # If sequence_length is longer than the multiple of all of our axes, that means it's been zero-padded and we need to do the same to the embeddings
            pad_length = sequence_length - reduce(lambda a, b: a * b, shape)
            freqs = F.pad(
                freqs.transpose(0, 2),
                (0, pad_length),
                mode="constant",
                value=0,
            ).transpose(0, 2)

            # Add singleton dimensions for batch and attention head count
            freqs = freqs.unsqueeze(0).unsqueeze(2).to(query.device)

            # Reshape into complex polar form, and cast to fp32 to maintain precision when multiplying
            query = query.reshape(*query.shape[:-1], -1, 2).to(torch.float32)
            key = key.reshape(*key.shape[:-1], -1, 2).to(torch.float32)

            query = multiply_as_complex(query, freqs).flatten(-2)
            key = multiply_as_complex(key, freqs).flatten(-2)

        return query.to(input_dtype), key.to(input_dtype)


# Handles the timestep embedding but doesn't calculate scale & shift or apply the LayerNorm - makes it a lot easier to generalise the transformer block code between DiT-style and Pixart-style models
class AdaLNEmbed(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        use_additional_conditions: bool = False
    ):
        super().__init__()

        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
            )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ):
        added_cond_kwargs = {} if added_cond_kwargs is None else added_cond_kwargs
        if isinstance(self.emb, PixArtAlphaCombinedTimestepSizeEmbeddings):
            added_cond_kwargs["batch_size"] = batch_size
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype, **added_cond_kwargs)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep
