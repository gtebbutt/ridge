import os
import csv
import argparse
from typing import Optional

import torch
import torchvision.transforms.v2.functional as TF

from PIL import Image
from diffusers import AutoencoderKL

from ridge.utils import load_csv, round_to
from ridge.pipelines.default import RidgePipeline


# Resizes an image to fit within max_pixel_count, with very slight cropping to account for patch alignment if necessary. Unless overridden, this will maintain original aspect ratio and will not upscale images that are already smaller than the maximum (both to within +/- one patch on each axis)
# With a reasonably large max_pixel_count this will alter most images by less than 3%, which is close to negligible when viewed
@torch.no_grad()
def transform_image(
    *,
    image: Image.Image,
    # Working in pixel space, for finest granularity, even though the final sequence lengths will be in patches of latent space (i.e. 16x smaller in each dimension when accounting for VAE scaling and patching, assuming default patch and VAE settings)
    max_pixel_count: int,
    vae_scale_factor: int,
    patch_size: int,
    # If override_width and override_height are set, the image will be scaled (maintaining aspect ratio, and upscaling if necessary) and then cropped to precisely those dimensions, but will still be validated for patch alignment and sequence length - this is only used for comparison testing against existing models, and isn't needed for general training
    override_width: Optional[int] = None,
    override_height: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
):
    if not isinstance(image, Image.Image):
        raise ValueError(f"Expected PIL image, but got {type(image)}")

    if not image.mode == "RGB":
        raise ValueError(f"Expected three channel RGB input, but got {image.mode}")

    if (override_width is not None and override_height is None) or (override_width is None and override_height is not None):
        raise ValueError(f"If either override_width or override_height is specified, the other must also be set")

    original_height = image.height
    original_width = image.width

    patch_size_pixels = vae_scale_factor * patch_size
    # Currently assumes two axes - can easily update the exponent if generalising
    max_sequence_length = max_pixel_count / (patch_size_pixels ** 2)

    if int(max_sequence_length) != max_sequence_length:
        raise ValueError(f"max_pixel_count {max_pixel_count} should be an integer multiple of patch_size_pixels^2 (patch_size_pixels = {patch_size_pixels})")

    max_sequence_length = int(max_sequence_length)

    # This will return a uint8 tensor - torchvision recommends keeping in that format for transformations; note that v1 transforms would automatically scale from uint8 to fp32, v2 as used here does not
    image = TF.pil_to_tensor(image)

    # Ensure shape is (c h w) as expected
    if not list(image.shape) == [3, original_height, original_width]:
        raise ValueError(f"Expected shape {[3, original_height, original_width]} but got {image.shape}")

    aspect_ratio = original_width / original_height

    # If overrides are set, use those directly
    if override_width or override_height:
        target_width = override_width
        target_height = override_height
    else:
        # Since aspect ratio is x / y, and max pixel count is x * y, it's trivial to figure this out with the known values
        target_width = (max_pixel_count * aspect_ratio) ** 0.5

        # Don't scale up smaller images
        target_width = min(target_width, original_width)

        # Calculate height based on the unrounded target_width, to minimise any ratio difference
        target_height = target_width / aspect_ratio

        # target_width and target_height are currently the largest values that'll fit within our pixel limit at the original aspect ratio, not accounting for patching (and not even rounded to whole pixels) - that gives an upper limit, now round each to the nearest patch, even if that's slightly larger and/or slightly off ratio
        target_width = round_to(target_width, patch_size_pixels)
        target_height = round_to(target_height, patch_size_pixels)

        # Depending on aspect ratio and how close each target value was to the patch boundary before rounding, the target size may now have overshot the maximum - that's expected, as part of the process of getting as close as possible to both the maximum patch count and the original aspect ratio while maintaining alignment to patch boundaries
        if target_width * target_height > max_pixel_count:
            short_side = min(target_width, target_height)
            long_side = max(target_width, target_height)

            # See if trimming only the short side by one patch will be enough (minimum possible crop)
            if (short_side - patch_size_pixels) * long_side <= max_pixel_count:
                short_side = short_side - patch_size_pixels
            # If not, see if trimming only the long side by one patch will be enough
            elif short_side * (long_side - patch_size_pixels) <= max_pixel_count:
                long_side = long_side - patch_size_pixels
            # If that's still not enough, trim both sides by one patch (which should always fit since the target rounding was only within a boundary of one patch on each side)
            else:
                short_side = short_side - patch_size_pixels
                long_side = long_side - patch_size_pixels
            
            if target_height > target_width:
                target_height = long_side
                target_width = short_side
            else:
                target_height = short_side
                target_width = long_side

    # Double check nothing unexpected has happened with float rounding, or with invalid override values being passed in
    if target_width % patch_size_pixels != 0 or target_height % patch_size_pixels != 0:
        raise ValueError(f"Calculated size {target_width}x{target_height} is not an integer multiple of {patch_size_pixels}")

    # Check height and width equality separately, in case we're resizing a very marginal image by a few pixels from its original ratio on only one axis
    if target_width != original_width or target_height != original_height:
        if target_width * target_height > max_pixel_count:
            raise ValueError(f"Total pixel count for ({target_width}, {target_height}) image is greater than maximum value {max_pixel_count}")

        # The target values should be reasonably close to the actual maximum, or to the original size if that's smaller and we're just resizing to align with patches - the difference can be slightly more significant at extreme aspect ratios, but anything substantially lower suggests an error in the calculations
        if target_width * target_height <= min(original_width * original_height, max_sequence_length) * 0.9:
            raise ValueError(f"Total pixel count for ({target_width}, {target_height}) image is more than 10% smaller than {max_pixel_count}")

    # This will automatically resize to cover and then crop precisely to target if necessary
    image, resize_height, resize_width, crop_top, crop_left = RidgePipeline.resize_and_crop(image, target_height, target_width, "random")

    # Scale here means scaling the values from uint8 to target dtype (generally fp32), not scaling the image dimensions
    image = TF.to_dtype(image, dtype=dtype, scale=True)
    image = TF.normalize(image, [0.5], [0.5])

    # Ensure shape is still (c h` w`) after transforms
    assert list(image.shape) == [3, target_height, target_width], f"Expected shape {[3, target_height, target_width]} after transformations, but got {image.shape}"
    
    # Adds batch dimension, expected by VAE
    image = image.unsqueeze(0)

    return {
        "image": image,
        "original_height": original_height,
        "original_width": original_width,
        "resized_height": resize_height,
        "resized_width": resize_width,
        "height": target_height,
        "width": target_width,
        "crop_top": crop_top,
        "crop_left": crop_left,
    }


@torch.no_grad()
def encode_image(
    vae: AutoencoderKL,
    image: torch.FloatTensor,
):
    return vae.encode(image.to(device=vae.device, dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor


@torch.no_grad()
def main(args):
    # Always uses fp32, to avoid instability. Image latents can be converted to lower precision later if required
    vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float32).to(args.device)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    rows = load_csv(args.input_csv_path, [args.id_column, args.image_filename_column])
    output = []

    os.makedirs(args.output_dir, exist_ok=True)

    for i, row in enumerate(rows):
        image_filename = row[args.image_filename_column]
        row_id = row[args.id_column]

        latent_filename = f"{row_id}_latent.pt"
        with Image.open(os.path.join(args.image_dir, image_filename)) as im:
            transformed = transform_image(
                image=im,
                max_pixel_count=args.max_pixel_count,
                vae_scale_factor=vae_scale_factor,
                patch_size=args.patch_size,
                override_width=args.override_width,
                override_height=args.override_height,
                dtype=vae.dtype,
            )

        image_latent = encode_image(vae, transformed["image"])

        del transformed["image"]
        row.update(transformed)

        # Important to clone the tensors when saving, otherwise torch will save the entire associated memory area rather than just the active view
        torch.save(image_latent.detach().clone(), os.path.join(args.output_dir, latent_filename))

        row[args.image_latent_column] = latent_filename

        output.append(row)

        if i % args.save_interval == 0:
            with open(args.output_csv_path, "w",  encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=output[0].keys())
                writer.writeheader()
                writer.writerows(output)

            print(f"Completed {i} / {len(rows)} rows")

    print(f"Completed all rows")


def get_args():
    parser = argparse.ArgumentParser(
        description="",
        allow_abbrev=False,
    )

    parser.add_argument("--vae_path", type=str, help="VAE to use when encoding the images")
    parser.add_argument("--max_pixel_count", type=int, default=1048576, help="Maximum image size in pixels, defaults to 1,048,576 (1024^2)")
    parser.add_argument("--input_csv_path", type=str, default="metadata.csv", help="Metadata csv containing the image filenames to be encoded")
    parser.add_argument("--id_column", type=str, default="id", help="Column header containing the row IDs, to use for file naming")
    parser.add_argument("--image_filename_column", type=str, default="image_filename", help="Column header containing the input images to encode")
    parser.add_argument("--image_dir", type=str, default="./images", help="Folder path for the input images, if not already included in the filenames column")
    parser.add_argument("--output_csv_path", type=str, default="metadata-latents.csv", help="Output path for csv with --image_latent_column and image metadata added")
    parser.add_argument("--image_latent_column", type=str, default="image_latent_filename", help="Column header to write the filenames of the image latents")
    parser.add_argument("--output_dir", type=str, default="./preprocessed/vae", help="Output folder for encoded tensors")
    parser.add_argument("--save_interval", type=int, default=100, help="Update the output CSV after processing every N rows, for partial output even if the script is stopped before completing")
    parser.add_argument("--patch_size", type=int, default=2, help="Patch size in latent space (can be found in transformer/config.json)")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use")

    parser.add_argument("--override_width", type=int, default=None, help="Force image width to this value. You probably don't want to use this")
    parser.add_argument("--override_height", type=int, default=None, help="Force image height to this value. You probably don't want to use this")

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
