import csv
from typing import List


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
