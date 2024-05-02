import os
import csv
import argparse

import torch

from tqdm import tqdm

from ridge.utils import load_csv
from ridge.pipelines.default import RidgePipeline


@torch.no_grad()
def main(args):
    pipe = RidgePipeline.from_pretrained(
        args.model_path,
        # Only need to load the text handling components
        transformer=None,
        scheduler=None,
        vae=None,
    )

    pipe.to(args.device)

    rows = load_csv(args.input_csv_path, [args.id_column, args.text_column])
    output = []

    os.makedirs(args.output_dir, exist_ok=True)

    for i, row in enumerate(tqdm(rows)):
        prompt = row[args.text_column]
        row_id = row[args.id_column]
        embed, attention_mask = pipe.encode_prompts(prompt, args.prompt_max_sequence_length, args.device)

        embed_filename = f"{row_id}_embed.pt"
        attention_mask_filename = f"{row_id}_mask.pt"

        # Important to clone the tensors when saving, otherwise torch will save the entire associated memory area rather than just the active view
        torch.save(embed.detach().clone(), os.path.join(args.output_dir, embed_filename))
        torch.save(attention_mask.detach().clone(), os.path.join(args.output_dir, attention_mask_filename))

        row[args.text_embedding_column] = embed_filename
        row[args.text_attention_mask_column] = attention_mask_filename

        output.append(row)

        if i % args.save_interval == 0 or i + 1 == len(rows):
            with open(args.output_csv_path, "w",  encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=output[0].keys())
                writer.writeheader()
                writer.writerows(output)

    print(f"Completed all rows")


def get_args():
    parser = argparse.ArgumentParser(
        description="",
        allow_abbrev=False,
    )

    parser.add_argument("--model_path", type=str, help="Diffusers-style model path to load tokenizer and text encoder from")
    parser.add_argument("--input_csv_path", type=str, default="metadata.csv", help="Metadata csv containing the prompts to be encoded")
    parser.add_argument("--id_column", type=str, default="id", help="Column header containing the row IDs, to use for file naming")
    parser.add_argument("--text_column", type=str, default="text", help="Column header containing the input text to encode")
    parser.add_argument("--output_csv_path", type=str, default="metadata-encoded.csv", help="Output path for csv with --text_embedding_column and --text_attention_mask_column added")
    parser.add_argument("--text_embedding_column", type=str, default="text_embedding_filename", help="Column header to write the filenames of the encoded prompts")
    parser.add_argument("--text_attention_mask_column", type=str, default="text_attention_mask_filename", help="Column header to write the filenames of the prompt attention masks")
    parser.add_argument("--output_dir", type=str, default="./preprocessed/text", help="Output folder for encoded tensors")
    parser.add_argument("--save_interval", type=int, default=50000, help="Update the output CSV after processing every N rows, for partial output even if the script is stopped before completing")
    parser.add_argument("--prompt_max_sequence_length", type=int, default=120, help="Maximum (token) sequence length to use when encoding - defines padding used for output tensors")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use")

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
