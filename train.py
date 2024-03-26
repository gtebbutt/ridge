# Basic training script for initial proof of concept on classified inputs

import os
import random
import argparse
import functools
from typing import List, Dict, Tuple, Callable, Optional, Union, Any

import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from diffusers.schedulers import DDPMScheduler
from einops._torch_specific import allow_ops_in_compiled_graph

from ridge.models.diffusion_transformer import DiffusionTransformerModel
from ridge.data.datasets import EncodedTensorDataset
from ridge.pipelines.default import RidgePipeline


logger = get_logger(__name__, log_level="INFO")


@torch.inference_mode()
def run_validation(
    *,
    base_model_path: str,
    accelerator: Accelerator,
    transformer: DiffusionTransformerModel,
    output_dir: str,
    output_suffix: str,
    seed: int,
    num_images_per_prompt: int,
    weight_dtype: torch.dtype,
    model_save_path: Optional[str] = None,
):
    if accelerator.is_main_process:
        logger.info("Running validation...")

        pipe = RidgePipeline.from_pretrained(
            base_model_path,
            transformer=accelerator.unwrap_model(transformer),
            torch_dtype=weight_dtype,
        )

        # Currently only used on the final validation, after training is complete, to save a final copy of everything in one place
        if model_save_path is not None:
            pipe.save_pretrained(os.path.join(output_dir, model_save_path))

        pipe.to(accelerator.device)

        validation_folder = os.path.join(output_dir, "samples", f"samples-{output_suffix}")
        os.makedirs(validation_folder, exist_ok=True)

        # Nothing inherently special about this choice of labels, it just matches the ones used in the original facebookresearch/DiT repo for easier comparison
        class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

        # These are also more or less arbitrary, as a decent range of shapes and sizes to test the boundaries of a nominally 256x256 model
        sizes = [
            (256, 256),
            (384, 384),
            (512, 512),
            (192, 336),
            (128, 512),
            (1024, 64),
        ]

        # Runs every size and label separately, rather than needing to keep track of what does/doesn't work with any given transformer config
        for size in sizes:
            for label in class_labels:
                generator = torch.Generator(device=accelerator.device).manual_seed(seed)

                # Enabling autocast when in fp32 mode can cause VAE issues
                with torch.cuda.amp.autocast(enabled=(weight_dtype != torch.float32)):
                    images = pipe(
                        class_labels=[label],
                        sizes=[size],
                        generator=generator,
                        num_images_per_prompt=num_images_per_prompt,
                    )[0]
                
                h, w = size
                for i, image in enumerate(images):
                    image.save(os.path.join(validation_folder, f"sample-{w}x{h}-{label}-{i}.png"))


def training_loop(
    *,
    args,
    accelerator: Accelerator,
    model: DiffusionTransformerModel,
    noise_scheduler: DDPMScheduler,
    optimizer: torch.optim.AdamW,
    dataloader: torch.utils.data.DataLoader,
):
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    global_step = 0
    steps_per_epoch = len(dataloader)
    max_steps = steps_per_epoch * args.num_epochs

    model.train()

    progress_bar = tqdm(range(global_step, max_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    error_count = 0

    for epoch in range(args.num_epochs):
        for epoch_step, batch in enumerate(dataloader):
            # Validation first, so it always runs on step zero before the weights are modified - particularly useful to get baseline output when fine tuning from an existing model
            if global_step % int(args.validation_epochs * steps_per_epoch) == 0:
                run_validation(
                    base_model_path=args.base_model_path,
                    accelerator=accelerator,
                    transformer=model,
                    output_dir=args.output_dir,
                    output_suffix=str(global_step),
                    seed=args.seed,
                    num_images_per_prompt=args.validation_images_per_prompt,
                    weight_dtype=weight_dtype,
                )

            if not batch:
                logger.error(f"Skipping step {global_step} due to a batch loading error")

                # Increment the global step to avoid throwing things out of sync, but log the error count as well so we know how many actual steps occurred - even when loading the dataset over the network, error count should be negligible compared to step count
                global_step += 1
                error_count += 1

            model_kwargs = {}

            if batch.get("class_labels") is not None:
                model_kwargs["class_labels"] = batch["class_labels"].to(device=accelerator.device, dtype=torch.int)
            else:
                model_kwargs["prompt_embeds"] = batch["prompt_embeds"].to(device=accelerator.device, dtype=weight_dtype)
                model_kwargs["prompt_attention_mask"] = batch["prompt_attention_mask"].to(device=accelerator.device, dtype=weight_dtype)

            # Since we're creating duplicate tensors here, it's done on the same device as the model_input from the batch (should always be the CPU, given how the dataloader is set up) and only the parts being passed to the model are moved to the GPU
            model_input = batch["model_input"]
            noise = torch.randn_like(model_input)
            timestep = torch.randint(0, noise_scheduler.config.num_train_timesteps, (model_input.shape[0],), device=model_input.device)
            timestep = timestep.long()
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timestep)

            timestep = timestep.expand(noisy_model_input.shape[0]).to(device=accelerator.device)
            noisy_model_input.to(device=accelerator.device, dtype=weight_dtype)

            noisy_model_input, shapes_patches, attention_mask = model.flatten_and_pad(batch=[l.unsqueeze(0) for l in noisy_model_input])

            with accelerator.accumulate(model):
                optimizer.zero_grad()

                noise_pred = model(
                    hidden_states=noisy_model_input,
                    shapes_patches=shapes_patches,
                    attention_mask=attention_mask,
                    timestep=timestep,
                    return_dict=False,
                    **model_kwargs,
                )[0]

                if model.config.out_channels // 2 == model.config.in_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                # NB: This currently only supports homogeneous batches, making padding/unpadding largely redundant - 
                noise_pred = torch.cat(model.unpad_and_reshape(noise_pred, shapes_patches), dim=0)
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                
                optimizer.step()

            accelerator.wait_for_everyone()

            # The actual training step has finished at this point, anything below is happeing at the subsequent step index
            global_step += 1
            progress_bar.update(1)
            
            if global_step % int(args.checkpointing_epochs * steps_per_epoch) == 0:
                if accelerator.is_main_process:
                    # This is currently just a snapshot of the model config and weights, rather than a full accelerator state checkpoint
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}", "transformer")
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        run_validation(
            base_model_path=args.base_model_path,
            accelerator=accelerator,
            transformer=model,
            output_dir=args.output_dir,
            output_suffix="final",
            seed=args.seed,
            num_images_per_prompt=args.validation_images_per_prompt,
            weight_dtype=weight_dtype,
            # Saves the full pipeline state to `args.output_dir/final` before validating
            model_save_path="final",
        )

    accelerator.end_training()


def collate_fn(examples, proportion_null_inputs):
    if len(examples) == 0:
        # Allows the training loop to cleanly recognise and skip batch loading errors
        return {}
    
    # VAE input already has a batch axis, so using torch.cat rather than torch.stack
    model_input = torch.cat([example["model_input"] for example in examples])
    class_labels = torch.cat([
        torch.tensor([int(example["metadata"]["label"])])
        if random.random() > proportion_null_inputs else
        torch.tensor([1000])
        for example in examples
    ])

    return {
        "model_input": model_input,
        "class_labels": class_labels,
    }


def main(args):
    # Prevent einops from causing unncessary graph breaks
    allow_ops_in_compiled_graph()

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        dynamo_backend=args.dynamo_backend,
        log_with=None,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs")),
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.base_model_path,
        subfolder="scheduler",
        variance_type=args.variance_type
    )

    transformer_base_path = args.base_model_path if args.transformer_path is None else args.transformer_path

    if args.train_from_scratch:
        diffusion_transformer = DiffusionTransformerModel.from_config(DiffusionTransformerModel.load_config(transformer_base_path, subfolder="transformer"))
    else:
        diffusion_transformer = DiffusionTransformerModel.from_pretrained(transformer_base_path, subfolder="transformer")

    # For now this just follows what the torch the defaults would do anyway, but it's useful to have here as a placeholder if we do need to freeze certain layers later
    diffusion_transformer.requires_grad_(True)
    params_to_optimize = list(filter(lambda p: p.requires_grad, diffusion_transformer.parameters()))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        weight_decay=args.adam_weight_decay,
    )

    dataset = EncodedTensorDataset(
        csv_path=args.csv_path,
        columns_to_load={
            args.image_latent_column: {"dir": args.image_latent_dir, "rename_to": "model_input"}
        },
        use_class_labels=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=functools.partial(collate_fn, proportion_null_inputs=args.proportion_null_inputs),
        num_workers=args.dataloader_num_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
    )

    # Can be passed in any order, the prepared objects are just returned in input order
    diffusion_transformer, optimizer, dataloader = accelerator.prepare(diffusion_transformer, optimizer, dataloader)

    training_loop(
        args=args,
        accelerator=accelerator,
        model=diffusion_transformer,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        dataloader=dataloader,
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="",
        allow_abbrev=False,
    )

    parser.add_argument("--base_model_path", type=str, help="Diffusers-style model path to load transformer, VAE, noise scheduler config, and validation pipeline config from")
    parser.add_argument("--transformer_path", type=str, default=None, help="Optional path to main transformer config and weights, to use instead of the transformer from --base_model_path")
    parser.add_argument("--train_from_scratch", action="store_true", help="Load main transformer config only (from either --base_model_path or --transformer_path if present) without loading pretrained weights")
    parser.add_argument("--csv_path", type=str, default="metadata.csv", help="Metadata csv for the dataloader")
    parser.add_argument("--image_latent_column", type=str, default="image_latent_filename", help="Column header containing the filenames for the VAE encoded image latents")
    parser.add_argument("--image_latent_dir", type=str, default="./preprocessed/vae", help="Folder path for VAE latents, if not already included in filenames column")
    parser.add_argument("--seed", type=int, default=42, help="Currently only used during validation")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataloader_num_workers", type=int, default=12)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    # Checkpointing and validation are defined relative to epochs rather than steps, since epochs don't introduce any dependency on batch size, GPU count, etc. - but accepts float values because waiting an entire epoch when trying out early tests is often too long
    parser.add_argument("--checkpointing_epochs", type=float, default=0.2)
    parser.add_argument("--validation_epochs", type=float, default=0.2)
    parser.add_argument("--validation_images_per_prompt", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="train-out")
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--dynamo_backend", type=str, default="inductor")
    parser.add_argument("--variance_type", type=str, default="learned_range")
    parser.add_argument("--adam_weight_decay", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--proportion_null_inputs", type=float, default=0.1, help="Input dropout, for CFG support. Applies whether input type is text prompts or class labels")

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
