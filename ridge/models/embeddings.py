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
        # This is equivalent to training_interpolation_scale in MultiAxisRotaryPositionEmbed, naming here is just kept to match the original version
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
        # Important to pass shapes explicitly, unlike the diffusers version, because the input has already been flattened to a sequence here
        shapes: List[List[int]],
        strength: float = 1.0,
    ):
        if not (0.0 <= strength <= 1.0):
            raise ValueError(f"Invalid PositionEmbed strength {strength}, values must be between 0 and 1")

        if len(set(shapes)) != 1:
            raise NotImplementedError(f"Heterogeneous batches are not yet supported")
        else:
            height, width = shapes[0]

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

        # Zero-pad pos_embed to match padded latent, if required
        if pos_embed.shape[1] < latent.shape[1]:
            padding_length = latent.shape[1] - pos_embed.shape[1]
            pos_embed = F.pad(
                pos_embed,
                (0, 0,
                 0, padding_length),
                mode="constant",
                value=0,
            )

        return (latent + (pos_embed * strength)).to(latent.dtype)


def multiply_as_complex(a, b):
    # Elementwise multiplication of complex numbers represented using an extra dimension, to avoid breaking torch.compile by using actual complex dtypes. Assumes final dimension of tensor is length 2 with format (real, imaginary)
    if a.shape[-1] != b.shape[-1] or a.shape[-1] != 2:
        raise ValueError(f"Expected both tensors to have (real, imaginary) as their final dimension, but got shapes {a.shape} and {b.shape}")

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
        max_trained_sequence_length: int,
        interpolation_type: Optional[str] = None,
        dynamic_interpolation: bool = False,
        training_interpolation_scale: float = 1.0,
        theta: float = 10000.0,
    ):
        super().__init__()

        # embed_dim will be equal to the attention_head_dim used in the transformer config
        self.embed_dim = embed_dim
        self.axes = axes
        self.theta = theta

        # The different types of scaling and when they get applied can sometimes be confusing; this one is used at training time to more quickly fine tune an existing model to a larger (or smaller) target sequence length by mapping new positions into the range that the model already understands, and then remains fixed at that value to maintain the base scaling for inference with those weights
        # Good visualisation and writeup at https://blog.eleuther.ai/yarn/#position-interpolation
        self.training_interpolation_scale = training_interpolation_scale

        # This is used to more precisely scale embeddings for each individual sequence at inference time if dynamic_interpolation is enabled. At training time it should be set to the maximum sequence length in the new training dataset, even if fine tuning from an existing checkpoint with a shorter trained sequence length
        self.max_trained_sequence_length = max_trained_sequence_length

        valid_interpolation_types = {None, "linear", "ntk", "yarn"}
        if interpolation_type not in valid_interpolation_types:
            raise ValueError(f"Invalid interpolation type: {interpolation_type}, expected one of {', '.join(valid_interpolation_types)}")

        self.interpolation_type = interpolation_type
        self.dynamic_interpolation = dynamic_interpolation

    # Linear implementations generally cache this in a buffer, but doing so for more axes triggers torch recompilation every time the shape changes, and adds potentially significant memory overhead unless you know up front you're only using a single aspect ratio
    def calculate_freqs(
        self,
        shape: List[int],
        tensor_sequence_length: int,
        # Functionally equivalent to alpha in the original dynamic NTK release, but also accounts for separate fine tuning with NTK scaling; values somewhere around 2.0 to 4.0 are generally reasonable
        ntk_alpha: Optional[float] = None,
    ):
        if not len(shape) == self.axes:
            raise ValueError(f"Expected a {self.axes}-axis shape, but got {shape}")

        if self.training:
            if self.dynamic_interpolation:
                raise RuntimeError(f"Dynamic interpolation should not be enabled during training: max_trained_sequence_length should be set equal to the longest sequence expected in the training dataset")
            
            if ntk_alpha is not None:
                raise ValueError(f"ntk_alpha is an inference parameter and should not be set during training; training_interpolation_scale should be used as the alpha value for training - see comments in ridge.models.embeddings for further details")

        if ntk_alpha is None:
            if self.dynamic_interpolation:
                # 1.0 means no further scaling
                ntk_alpha = 1.0
        else:
            if not self.dynamic_interpolation:
                raise ValueError(f"ntk_alpha is only compatible with dynamic interpolation. To statically scale the whole embedding space (NTK-aware interpolation), use training_interpolation_scale as the alpha value")

        # Using the shape value ensures that any interpolation is based on the actual input size, not on the padded sequence length within the batched tensor
        input_sequence_length = reduce(lambda a, b: a * b, shape)

        if self.interpolation_type is None:
            interpolation_scale = 1.0
        else:
            # This class currently assumes uniform scaling across all axes, which gives greater overall flexibility and works well for real-world image generation use, but it can easily be extended to apply per-axis by tracking the static interpolation scale (and/or max trained length) for each axis rather than for the sequence as a whole and calculating a separate interpolation scale for each; may be useful to do so if extrapolating to aspect ratios very significantly beyond those in the training data

            # All interpolation types will, at minimum, apply a constant scaling factor (which may be 1.0 in the case of a model that hasn't been fine tuned above its original max sequence length)
            interpolation_scale = self.training_interpolation_scale

            # Dynamic scaling further increases the interpolation factor based on the input sequence length if it's longer than the trained maximum
            if self.dynamic_interpolation and input_sequence_length > self.max_trained_sequence_length:
                interpolation_scale = interpolation_scale * input_sequence_length / self.max_trained_sequence_length
            else:
                # Only apply the inference time NTK alpha scaling to longer sequences - this matches how the dynamic NTK formula is originally described, although the caching in the reference implementation overwrites the shorter sequence embeddings the first time a longer sequence is used, meaning the actual behaviour of that version can be inconsistent
                # This will also ensure the correct calculation when dynamic mode is disabled, without breaking the input validation above
                ntk_alpha = 1.0

        if self.interpolation_type == "ntk":
            exponent = (self.embed_dim / (self.embed_dim - 2))
            # This slightly extends /u/emozilla's original dynamic NTK implementation by defining alpha=(self.training_interpolation_scale*ntk_alpha)
            # It allows a static training alpha to be set for much faster fine tuning of pretrained models, which is then locked in as the baseline and scaled at inference time by ntk_alpha for better long-range performance
            theta = self.theta * ((interpolation_scale * ntk_alpha) - ((self.training_interpolation_scale * ntk_alpha) - 1)) ** exponent
        else:
            theta = self.theta

        # One tensor for each axis, each containing the cartesian coordinates on that axis for every point in the space; meshgrid returns int64 coordinates by default
        axis_positions = torch.meshgrid([torch.arange(l) for l in shape], indexing="ij")
        axis_positions = [pos.to(torch.float32) * self.training_interpolation_scale for pos in axis_positions]

        if self.interpolation_type == "linear":
            axis_positions = axis_positions / interpolation_scale

        # Explicitly using fp32 here, since autocast is disabled in the forward function
        # Factor of two per axis to account for the fact we'll be repeating this for each axis, and then using the overall result twice to get the sin and cos components in the forward function
        base_freqs = 1.0 / (theta ** (torch.arange(0, self.embed_dim, 2 * self.axes).to(dtype=torch.float32) / self.embed_dim))

        freqs = [torch.outer(pos.flatten(), base_freqs) for pos in axis_positions]

        # Einops accepts a list of tensors and treats it as if the values were already stacked into a single tensor, which makes things easier here
        # Using torch.cat(..., dim=-1) directly would be equivalent to "a n d -> n (a d)", which gives the shape we want but not the correct ordering of elements within the tensor; the alternative is to stack along a new final dimension and then reshape, but rearrange makes things more readable. Ordering matters here because we're emulating the way torch handles complex polar values
        # Returns shape (n, self.embed_dim / 2)
        freqs = rearrange(freqs, "a n d -> n (d a)")

        # Equivalent to torch.polar but using the last dimension for (real, imaginary) rather than using complex dtypes, as they aren't compatible with torch.compile
        freqs = torch.stack([freqs.cos(), freqs.sin()], dim=-1)

        # If sequence_length is longer than the multiple of all of our axes, that means it's been zero-padded and we need to do the same to the embeddings
        padding_length = tensor_sequence_length - input_sequence_length
        return F.pad(
            freqs,
            (0, 0,
            0, 0,
            0, padding_length),
            mode="constant",
            value=0,
        )

    
    # Wrapper to ensure that calling code can easily create the frequencies tensor in an identical way to the forward function (e.g. for caching)
    def calculate_batch_freqs(
        self,
        shapes: List[List[int]],
        tensor_sequence_length: int,
        ntk_alpha: Optional[float] = None,
    ):
        freqs = []
        # Calculate freqs for each shape in the batch separately, to allow for heterogeneous batches
        for shape in shapes:
            freqs.append(self.calculate_freqs(shape, tensor_sequence_length, ntk_alpha))

        # Stack along batch axis and add singleton dimension for attention head count
        return torch.stack(freqs).unsqueeze(2)


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        shapes: List[List[int]],
        ntk_alpha: Optional[int] = None,
        freqs: Optional[torch.Tensor] = None,
    ):
        # Query and key shape are both (b n h d) where h is num_attention_heads and d is attention_head_dim (which should be equal to embed_dim); for DiT and Pixart models num_attention_heads=16 and attention_head_dim=72
        sequence_length = query.shape[1]
        input_dtype = query.dtype

        # Always run these calculations in fp32, even if autocast is enabled - longer sequences get unstable if the sin and cos values are handled in reduced precision
        with torch.cuda.amp.autocast(enabled=False):
            # There are a lot of cases where the same `freqs` tensor will be reused multiple times (at minimum for the multiple steps using the same latents within a given diffusion run), but trying to use a buffer to cache the value inside this class is inflexible and error-prone, and can break assumptions made by torch.compile
            # Optionally passing the tensor into this function as an argument allows the value to be easily calculated and cached by calling code (e.g. at the pipeline level) without imposing any assumptions or potentially unwanted state on the model as a whole
            # If internal caching in a buffer is needed for any reason, this also makes it trivial to implement using a wrapper class
            if freqs is None:
                freqs = self.calculate_batch_freqs(shapes, sequence_length, ntk_alpha)

            freqs = freqs.to(query.device)

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
        use_additional_conditions: bool = False,
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
