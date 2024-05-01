import sys
import csv
import json
import platform
import subprocess
from typing import List

import torch


def round_to(x, base, fn=round):
    return base * fn(x / base)


def load_csv(
    csv_path: str,
    required_columns: List[str] = [],
):
    with open(csv_path) as f:
        reader = csv.DictReader(f)

        if not set(required_columns).issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV file {csv_path} is missing expected key(s): {set(reader.fieldnames) - set(required_columns)}")

        data_rows = [r for r in reader]
    
    return data_rows


def to_json(obj):
    # Output format matches diffusers ConfigMixin.to_json_file()
    return f"{json.dumps(obj, indent=2, sort_keys=True)}\n"


def get_commit_hash():
    ch = None
    try:
        ch = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception as e:
        print(f"Couldn't get commit hash: {e}")
    return ch


def get_system_info():
    gpus = [
        {
            "name": torch.cuda.get_device_properties(i).name,
            "total_memory": torch.cuda.get_device_properties(i).total_memory,
        }
        for i in range(torch.cuda.device_count())
    ]

    return {
        "python": {
            "version": sys.version,
            "hex": sys.hexversion,
        },
        "commit_hash": get_commit_hash(),
        "gpu_info": gpus,
        "platform_info": platform.uname()._asdict(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
