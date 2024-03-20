import os
import csv
import math
from typing import Dict, List, Iterator

import torch
from torch.utils.data import Dataset, Sampler


# Simple loader for already encoded data (VAE latents, text embeddings, etc.), allows for a clean separation between data prep and actual training
class EncodedTensorDataset(Dataset):
    def __init__(
        self,
        *,
        csv_path: str,
        columns_to_load: Dict[str, Dict[str, str]],
        use_class_labels: bool = False,
    ):
        self.columns_to_load = columns_to_load
        self.use_class_labels = use_class_labels

        # columns_to_load specifies the columns with tensor filenames to be loaded into memory; all columns from the csv file will still be read and returned under the `metadata` key
        for column_name, loading_info in self.columns_to_load.items():
            if loading_info.get("dir") is None:
                raise ValueError(f"No directory provided in {loading_info} for column {column_name}")

            # Each loadable column must have a dir for the actual files, and can have an optional rename_to if the output column name differs from the input (e.g. loading from "image_latent_filename" but returning the actual tensor under "image_latent")
            expected_keys = {"dir", "rename_to"}

            if not set(loading_info.keys()).issubset(expected_keys):
                raise ValueError(f"Expected keys of column info {loading_info} for column {column_name} to be a subset of {expected_keys}")

        # The metadata is generally just filenames, so not usually a significant overhead to read the whole list into memory
        with open(csv_path) as f:
            reader = csv.DictReader(f)

            if not set(self.columns_to_load.keys()).issubset(set(reader.fieldnames)):
                raise ValueError(f"CSV file {csv_path} is missing expected key(s): {set(reader.fieldnames) - set(self.columns_to_load.keys())}")

            self.data_rows = [r for r in reader]

    def __len__(self):
        return len(self.data_rows)

    def __getitem__(self, idx):
        row = self.data_rows[idx]

        output = {}

        for column_name, loading_info in self.columns_to_load.items():
            key = loading_info.get("rename_to", column_name)

            # Always load into CPU memory - loading directly to GPU (including implicitly doing so by omitting map_location if a tensor was originally saved from a GPU) can cause CUDA context deadlocks if done inside a torch DataLoader
            output[key] = torch.load(os.path.join(loading_info["dir"], row[column_name]), map_location="cpu")

        output["metadata"] = row

        return output
