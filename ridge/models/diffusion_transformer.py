import re
from types import SimpleNamespace
from collections import OrderedDict
from typing import List, Dict, Tuple, Callable, Optional, Union, Any

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection

from .attention import Attention
from .embeddings import PatchEmbed, PositionEmbed, AdaLNEmbed


class DiffusionTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str,
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str,
        norm_eps: float,
        use_rotary_pos_embed: bool = False,
        max_trained_sequence_length: Optional[int] = None,
    ):
        super().__init__()

        if norm_type == "ada_norm_zero":
            # DiT uses adaptive norm per-block, rather than once at the top of the network
            self.adaln_embed = AdaLNEmbed(dim, num_embeddings=num_embeds_ada_norm)
        else:
            self.adaln_embed = None

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            use_rotary_pos_embed=use_rotary_pos_embed,
            max_trained_sequence_length=max_trained_sequence_length,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        if cross_attention_dim:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                bias=attention_bias,
                use_rotary_pos_embed=use_rotary_pos_embed,
                max_trained_sequence_length=max_trained_sequence_length,
            )
        else:
            self.attn2 = None

        self.ff = FeedForward(
            dim=dim,
            activation_fn=activation_fn,
        )

        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        shapes_patches: Optional[List[Tuple[int, int]]] = None,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0]

        if self.adaln_embed is not None:
            # DiT models only - Pixart-style models apply this once at the transformer level
            timestep, _ = self.adaln_embed(
                timestep, {"class_labels": class_labels}, hidden_dtype=hidden_states.dtype
            )

            # Diffusers skips the reshape here, meaning there are then checks throughout to treat the dimensions differently between Pixart and DiT - adding the reshape at the top is functionally identical but easier to keep track of
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = timestep.reshape(batch_size, 6, -1).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.squeeze(1)

        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
            shapes_patches=shapes_patches,
            **cross_attention_kwargs,
        )

        attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        if self.attn2 is not None:
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                shapes_patches=shapes_patches,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp * ff_output
        hidden_states = ff_output + hidden_states

        return hidden_states


class DiffusionTransformerModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        *,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: int,
        patch_size: int,
        activation_fn: str,
        num_embeds_ada_norm: Optional[int] = None,
        norm_type: str,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-6,
        caption_channels: int = None,
        # It won't make sense to enable both of these in normal use, but it's important for training to allow one to be phased out in favour of the other over the course of a few thousand steps
        use_absolute_pos_embed: bool = True,
        use_rotary_pos_embed: bool = False,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.patch_size = patch_size

        # Note that these dimensions are in latent space, so will be 8x smaller than pixel values (assuming default VAE scale factor)
        # Currently assumes square patches in two dimensions, but can easily be extended
        max_trained_sequence_length = (sample_size ** 2) // (self.patch_size ** 2)

        # This just does the patching, rather than treating patching and position embedding as a single operation
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
        )

        if use_absolute_pos_embed:
            # Assumes the default sample size is 512x512px with an 8x VAE reduction (i.e. 64x64 latents)
            interpolation_scale = sample_size // 64
            interpolation_scale = max(interpolation_scale, 1)

            self.pos_embed = PositionEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                embed_dim=inner_dim,
                interpolation_scale=interpolation_scale,
            )
        else:
            self.pos_embed = None

        self.transformer_blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    use_rotary_pos_embed=use_rotary_pos_embed,
                    max_trained_sequence_length=max_trained_sequence_length,
                )
                for d in range(num_layers)
            ]
        )

        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        if norm_type == "ada_norm_zero":
            self.timestep_proj_out = nn.Linear(inner_dim, 2 * inner_dim)
            self.adaln_embed = None
            self.caption_projection = None
        else:
            self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
            self.adaln_embed = AdaLNEmbed(inner_dim, use_additional_conditions=self.config.sample_size == 128)
            self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

        self.gradient_checkpointing = False

    # The additional keys this is converting aren't deprecated per se, just different to the ones diffusers uses, but extending this function is by far the easiest way to ensure they're patched in the right place
    def _convert_deprecated_attention_blocks(self, state_dict: OrderedDict) -> None:
        super()._convert_deprecated_attention_blocks(state_dict)

        original_keys = list(state_dict.keys())

        def update_key(old_str, new_str):
            new_key = k.replace(old_str, new_str)
            state_dict[new_key] = state_dict.pop(k)

        is_dit = any("class_embedder" in k for k in original_keys)
        dit_norm1_prefix = re.compile("^transformer_blocks\.\d*\.norm1\.")

        for k in original_keys:
            if "pos_embed.proj" in k:
                update_key("pos_embed", "patch_embed")

            if is_dit:
                if dit_norm1_prefix.match(k):
                    # There's also a LayerNorm that's been moved from norm1.norm to norm1, but that has no learnable parameters, so doesn't have any state keys to worry about
                    update_key("norm1", "adaln_embed")
                
                if "norm3" in k:
                    update_key("norm3", "norm2")
                
                if "proj_out_1" in k:
                    update_key("proj_out_1", "timestep_proj_out")
                
                if "proj_out_2" in k:
                    update_key("proj_out_2", "proj_out")
            else:
                if "adaln_single" in k:
                    update_key("adaln_single", "adaln_embed")

    # The Conv2d used by patch_embed to downsample each patch to a single float has learnable parameters, and the padding needs to be done before the data reaches the model otherwise it can't be batched into a single tensor (as required for autograd to work properly)
    # Since the Conv2d call treats each patch as independent and non-overlapping, and then the rest of the model expects a flattened sequence anyway, it's functionally identical whether the input is passed in its original ratio or rearranged into a single column of blocks with height h*p and width p
    def flatten_and_pad(
        self,
        *,
        batch: List[torch.Tensor],
        attention_masks: Optional[List[Optional[torch.Tensor]]] = None,
        target_sequence_length_patches: Optional[int] = None,
    ):
        if attention_masks is None: 
            attention_masks = [None] * len(batch)

        if len(attention_masks) != len(batch):
            raise ValueError(f"Expected one attention mask per item, but got {len(attention_masks)} for batch size {len(batch)}")

        # Important that shapes are captured before flattening, so that the positional embedding for non-square input doesn't get thrown off
        # Don't forget that dimensions are in latent space, not pixel space
        shapes_patches = []
        for item in batch:
            # Expect each item to have a batch dimension, so that embedding functions don't need to worry about whether they're getting batched or unbatched input, but ensure it's still only a single item
            if item.ndim != 4 or item.shape[0] != 1 or item.shape[1] != self.config.in_channels:
                raise ValueError(f"Expected input shape (1 {self.config.in_channels} h w) but got {item.shape}")
            shapes_patches.append((item.shape[-2] // self.patch_size, item.shape[-1] // self.patch_size))

        if target_sequence_length_patches is None:
            target_sequence_length_patches = max([h * w for h, w in shapes_patches])

        for i in range(len(batch)):
            # Unfolds the h and w dimensions to give a six dimensional tensor of shape (b c h/p p w/p p), and then flattens into a single column of patches to give (b c h*w/p p)
            batch[i] = batch[i].unfold(2, size=self.patch_size, step=self.patch_size).unfold(3, size=self.patch_size, step=self.patch_size).flatten(start_dim=2, end_dim=4)

            sequence_length_patches = batch[i].shape[-2] // self.patch_size

            if sequence_length_patches > target_sequence_length_patches:
                raise ValueError(f"Tensor shape {batch[i].shape} will result in a sequence of length {sequence_length_patches}, which is greater than target {target_sequence_length_patches}")

            if attention_masks[i] is None:
                # Mask is a simple 2D tensor of (b n), i.e. one value per patch in the sequence, no need to worry about channels etc. 
                attention_masks[i] = torch.ones((1, sequence_length_patches), device=batch[i].device, dtype=batch[i].dtype)

            if attention_masks[i].shape[-1] < target_sequence_length_patches:
                padding_length = target_sequence_length_patches - attention_masks[i].shape[-1]
                attention_masks[i] = F.pad(
                    attention_masks[i],
                    (0, padding_length),
                    mode="constant",
                    value=0,
                )

            if sequence_length_patches < target_sequence_length_patches:
                padding_length = (target_sequence_length_patches * self.patch_size) - (sequence_length_patches * self.patch_size)
                batch[i] = F.pad(
                    batch[i],
                    # padding_before, padding_after for each dimension, counting from the last
                    (0, 0, # Width is already exactly p
                    0, padding_length),
                    mode="constant",
                    value=0,
                )

        return torch.cat(batch, dim=0), shapes_patches, torch.cat(attention_masks, dim=0)

    def unpad_and_reshape(
        self,
        latents: torch.Tensor,
        shapes_patches: List[Tuple[int, int]],
    ):
        num_channels = latents.shape[1]
        latents = list(latents.unbind(dim=0))
        for i in range(len(latents)):
            height_patches, width_patches = shapes_patches[i]
            sequence_length = height_patches * width_patches * self.patch_size
            latents[i] = latents[i][:, :sequence_length]
            latents[i] = latents[i].unfold(1, size=self.patch_size, step=self.patch_size)
            latents[i] = rearrange(latents[i], "c (h w) pw ph -> c (h ph) (w pw)", c=num_channels, h=height_patches)
            # Add a batch axis back for consistency with input format
            latents[i] = latents[i].unsqueeze(0)

        return latents

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        absolute_pos_embed_strength: float = 1.0,
        # (h w) shape, in patches, for each item in the batch - important to know if they've been padded, and/or to properly encode positions of images that are the same total number of patches but a different aspect ratio
        shapes_patches: List[Tuple[int, int]] = None,
        return_dict: bool = True,
    ):
        batch_size = hidden_states.shape[0]

        # These will generally not be the actual image dimensions, assuming flatten_and_pad was called, but it's important to store whatever dimensions the tensor started with before patchifying so it can be recovered correctly at the other end
        output_width = hidden_states.shape[-1]
        output_height = hidden_states.shape[-2]

        # Compress each patch into a single float and flatten into a linear sequence
        hidden_states = self.patch_embed(hidden_states) # (b n c)


        if self.pos_embed is not None:
            hidden_states = self.pos_embed(hidden_states, shapes_patches, absolute_pos_embed_strength)

        if attention_mask is not None and attention_mask.ndim == 2:
            # Converts a bitmask to a usable bias that can be added directly to the attention scores
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # This will use the text encoder sequence length (if applicable), so doesn't need padding to input_sequence_length
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if self.adaln_embed is not None:
            # Pixart-style models only - DiT applies this at the block level
            timestep, embedded_timestep = self.adaln_embed(
                timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    shapes_patches,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    shapes_patches=shapes_patches,
                )

        if self.adaln_embed is None:
            _, embedded_timestep = self.transformer_blocks[0].adaln_embed(timestep, {"class_labels": class_labels}, hidden_dtype=hidden_states.dtype)
            # Reshaping here converts a (batch_size, 2304) tensor to (batch_size, 2, 1152) before chunking - that way the dimensions match those expected in DiT models
            shift, scale = self.timestep_proj_out(F.silu(embedded_timestep)).reshape((batch_size, 2, -1)).chunk(2, dim=1)
        else:
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # Unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, output_height // self.patch_size, output_width // self.patch_size, self.patch_size, self.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        hidden_states = hidden_states.reshape(
            shape=(-1, self.out_channels, output_height, output_width)
        )

        if return_dict:
            # Some diffusers pipelines are hardcoded to expect a .sample parameter on the return value, and SimpleNamespace is the easiest way to return an object with dot notation access (rather than dict-style square bracket notation)
            return SimpleNamespace(sample=hidden_states)
        else:
            return (hidden_states,)
