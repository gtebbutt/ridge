import math
from typing import List, Dict, Tuple, Callable, Optional, Union, Any

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import retrieve_timesteps
from transformers import T5EncoderModel, T5Tokenizer

from ..models.diffusion_transformer import DiffusionTransformerModel


def round_to(x, base, fn=round):
    return base * fn(x / base)


# Combines and simplifies the core functionality from diffusers versins of both DiTPipeline and PixartAlphaPipeline, and adds support for DiffusionTransformerModel's extra features
class RidgePipeline(DiffusionPipeline):
    _optional_components = ["tokenizer", "text_encoder"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        *,
        transformer: DiffusionTransformerModel,
        vae: AutoencoderKL,
        scheduler: KarrasDiffusionSchedulers,
        tokenizer: Optional[T5Tokenizer],
        text_encoder: Optional[T5EncoderModel],
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer, vae=vae, scheduler=scheduler, tokenizer=tokenizer, text_encoder=text_encoder,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # Identical to the version used in the majority of diffusers pipelines, but it looks like it's not part of the base DiffusionPipeline class
    def prepare_latents(
        self,
        *,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
        latents: torch.Tensor = None,
    ):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @staticmethod
    def resize_and_crop(
        batch: torch.Tensor,
        target_height: int,
        target_width: int,
        crop_type: str = "center"
    ):
        input_height, input_width = batch.shape[-2], batch.shape[-1]

        # Return values if resize and/or crop aren't needed
        resize_height = input_height
        resize_width = input_width
        crop_top = 0
        crop_left = 0

        if input_height != target_height or input_width != target_width:
            input_ratio = input_width / input_height
            target_ratio = target_width / target_height

            if target_ratio > input_ratio:
                # If the target size is narrower than the original, resize to target_width at the original aspect and the top/bottom will be cropped
                resize_width = target_width
                # Always round up, crop will trim off the extra single pixel if required
                resize_height = int(math.ceil(resize_width / input_ratio))
            else:
                # If the patch target size is wider, do the opposite and the left/right edges will be cropped
                resize_height = target_height
                resize_width = int(math.ceil(resize_height * input_ratio))

            batch = TF.resize(batch, (resize_height, resize_width), interpolation=TF.InterpolationMode.BICUBIC)

            if crop_type is not None and (batch.shape[-2] != target_height or batch.shape[-1] != target_width):
                if crop_type == "center":
                    crop_top = (resize_height - target_height) // 2
                    crop_left = (resize_width - target_width) // 2
                elif crop_type == "random":
                    crop_top, crop_left, _, _ = transforms.RandomCrop.get_params(batch, output_size=(target_height, target_width))
                else:
                    raise ValueError(f"Unexpected crop type: {crop_type}")

                batch = TF.crop(batch, crop_top, crop_left, target_height, target_width)

        # Return the intermediate values in case they're needed for training use
        return batch, resize_height, resize_width, crop_top, crop_left

    @staticmethod
    def prepare_class_labels(
        class_labels: List[int],
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        device: torch.device,
    ):
        class_labels = torch.tensor(class_labels * num_images_per_prompt, device=device)
        if do_classifier_free_guidance:
            class_nulls = torch.tensor([1000] * len(class_labels) * num_images_per_prompt, device=device)
            # Doesn't specifically matter whether this is nulls first or labels first, just has to match the order of noise_pred_uncond, noise_pred_cond after the prediction is run
            class_labels = torch.cat([class_nulls, class_labels], dim=0)

        return class_labels

    # This differs from the diffusers version in that it just handles encoding, with a separate prepare_prompt_embeds function for cfg duplication etc. - makes it easier to call separately during training
    def encode_prompts(
        self,
        prompts: Union[str, List[str]],
        max_sequence_length: int,
        device: torch.device,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        prompts = [p.lower().strip() for p in prompts]
        prompt_tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_attention_mask = prompt_tokens.attention_mask.to(device)
        prompt_embeds = self.text_encoder(prompt_tokens.input_ids.to(device), attention_mask=prompt_attention_mask)[0]

        return prompt_embeds, prompt_attention_mask

    @staticmethod
    def prepare_prompt_embeds(
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        null_prompt_embeds: torch.Tensor,
        null_prompt_attention_mask: torch.Tensor,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        if prompt_attention_mask is not None and prompt_attention_mask.shape[0] != prompt_embeds.shape[0]:
            raise ValueError(f"prompt_embeds and prompt_attention_mask must have the same batch size, but got shapes {prompt_embeds.shape} and {prompt_attention_mask.shape}")

        batch_size = prompt_embeds.shape[0]

        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)

        if prompt_attention_mask is not None:
            prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            if null_prompt_embeds is None:
                raise ValueError(f"null_prompt_embeds must be passed if do_classifier_free_guidance is enabled")
            elif null_prompt_embeds.shape[1] != prompt_embeds.shape[1]:
                raise ValueError(f"null_prompt_embeds should be padded to the same sequence length as prompt_embeds, but got shapes {null_prompt_embeds.shape} and {prompt_embeds.shape}")
            
            if null_prompt_attention_mask is not None and null_prompt_attention_mask.shape[0] != null_prompt_embeds.shape[0]:
                raise ValueError(f"null_prompt_embeds and null_prompt_attention_mask must have the same batch size, but got shapes {null_prompt_embeds.shape} and {null_prompt_attention_mask.shape}")

            if null_prompt_embeds.shape[0] == 1:
                null_prompt_embeds = null_prompt_embeds.repeat(prompt_embeds.shape[0], 1, 1)
            elif null_prompt_embeds.shape[0] == batch_size:
                null_prompt_embeds = null_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
            else:
                raise ValueError(f"null_prompt_embeds is expected to have batch size 1, or the same batch size as prompt_embeds, but got shapes {null_prompt_embeds.shape} and {prompt_embeds.shape}")
            
            if null_prompt_attention_mask is not None:
                # We've already confirmed the batch size was originally equal to null_prompt_embeds, so this will always be an integer multiple
                null_prompt_attention_mask = null_prompt_attention_mask.repeat(null_prompt_embeds.shape[0] // null_prompt_attention_mask.shape[0], 1)

            # As in prepare_class_labels, the order just needs to match the order that noise_pred is unpacked in - so nulls first, in this case
            prompt_embeds = torch.cat([null_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([null_prompt_attention_mask, prompt_attention_mask], dim=0)
        
        return prompt_embeds, prompt_attention_mask

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompts: Optional[Union[str, List[str]]] = None,
        class_labels: Optional[Union[int, List[int]]] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: Optional[int] = 1,
        # List of (h w) tuples, in pixels, one per class label
        # NB: if one batch item is significantly larger (in terms of overall pixel count, h*w) than the rest, it's generally more efficient to run it separately - by necessity all inputs will be padded to the size of the largest, which can be wasteful if there's a big gap between sizes
        sizes: Optional[List[Tuple[int, int]]] = None,
        prompt_max_sequence_length: int = 120,
        ntk_alpha: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        # Used during training validation
        absolute_pos_embed_strength: float = 1.0,
    ):
        device = self._execution_device
        
        if prompts is None and class_labels is None:
            raise ValueError(f"Must specify one of prompts or class_labels")
        elif prompts is not None and class_labels is not None:
            raise ValueError(f"Must specify only one of prompts or class_labels, but not both")

        if isinstance(class_labels, int):
            class_labels = [class_labels]
        
        if isinstance(prompts, str):
            prompts = [prompts]

        if prompts is not None:
            batch_size = len(prompts)
        else:
            batch_size = len(class_labels)

        if sizes is None:
            sizes = [(self.transformer.config.sample_size * self.vae_scale_factor, self.transformer.config.sample_size * self.vae_scale_factor)] * batch_size

        if len(sizes) != batch_size:
            raise ValueError(f"Expected number of image sizes ({len(sizes)}) to match number of class labels or prompts ({batch_size})")

        # Ensure each image will map to an integer number of patches after VAE encoding and convolutional downsampling, rounding if necessary
        # Unlike aspect ratio binning, this is guaranteed to be within (patch_size * vae_scale_factor)/2 pixels of the requested size on each side - the 1 to 2% trim is pretty much imperceptible in normal use
        patch_size_pixels = self.vae_scale_factor * self.transformer.patch_size
        sizes_aligned = [(round_to(h, patch_size_pixels), round_to(w, patch_size_pixels)) for h, w in sizes]

        latent_channels = self.transformer.config.in_channels
        do_classifier_free_guidance = guidance_scale > 1.0

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # Latents may be different sizes, so they need to be created individually before batching
        latents = []
        for _ in range(num_images_per_prompt):
            for h, w in sizes_aligned:
                latents.append(self.prepare_latents(
                    batch_size=1,
                    num_channels_latents=latent_channels,
                    height=h,
                    width=w,
                    dtype=self.transformer.dtype,
                    device=device,
                    generator=generator,
                ))

        # Pad latents to match the largest size in the batch and return as a single tensor; will also be reshaped as part of the process, but that doesn't affect the inference loop
        latents, shapes_patches, attention_mask = self.transformer.flatten_and_pad(batch=latents)
        attention_mask = torch.cat([attention_mask] * 2) if do_classifier_free_guidance else latents

        model_kwargs = {}

        if prompts is not None:
            prompt_embeds, prompt_attention_mask = self.encode_prompts(prompts, prompt_max_sequence_length, device)

            if do_classifier_free_guidance:
                null_prompt_embeds, null_prompt_attention_mask = self.encode_prompts("", prompt_max_sequence_length, device)
            else:
                null_prompt_embeds, null_prompt_attention_mask = (None, None)

            prompt_embeds, prompt_attention_mask = self.prepare_prompt_embeds(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                null_prompt_embeds=null_prompt_embeds,
                null_prompt_attention_mask=null_prompt_attention_mask,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )

            model_kwargs["encoder_hidden_states"] = prompt_embeds
            model_kwargs["encoder_attention_mask"] = prompt_attention_mask
        else:
            model_kwargs["class_labels"] = self.prepare_class_labels(class_labels, num_images_per_prompt, do_classifier_free_guidance, device)

        # TODO: Set actual values if using unmodified Pixart 1024 weights, for precise backwards compatibility
        # Resolution and aspect_ratio args are requried by PixArtAlphaCombinedTimestepSizeEmbeddings, even if they're set to None
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input.device)
                current_timestep = current_timestep.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    shapes_patches=shapes_patches,
                    attention_mask=attention_mask,
                    timestep=current_timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    ntk_alpha=ntk_alpha,
                    absolute_pos_embed_strength=absolute_pos_embed_strength,
                    return_dict=False,
                    **model_kwargs,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                else:
                    noise_pred = noise_pred

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                progress_bar.update()

        if output_type == "latent":
            # These will still be padded and reshaped, ready to pass directly back to a model if required
            output = latents
        else:
            # Unpack into a list of correctly shaped tensors and process separately
            latents = self.transformer.unpad_and_reshape(latents, shapes_patches)

            output = []
            for i, latent in enumerate(latents):
                image = self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False)[0]
                h, w = sizes[i % num_images_per_prompt]
                image = self.resize_and_crop(image, h, w)[0]
                # This is returned as a list since it's treated as a one-item batch
                image = self.image_processor.postprocess(image, output_type=output_type)
                output.extend(image)

        self.maybe_free_model_hooks()

        return (output,)
