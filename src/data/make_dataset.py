import pandas as pd
import numpy as np
import kagglehub
import os

from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from kagglehub import KaggleDatasetAdapter

def load_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")


def load_params(filepath: str) -> float:
    try:
        yaml = YAML(typ='safe')
        with open(filepath) as f:
            params = yaml.load(f)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}: {e}")

def split_data(data: pd.DataFrame, test_size: float):
    try:
        return train_test_split(data, test_size=test_size, random_state=42)
    except Exception as e:
        raise ValueError(f"Error splitting data: {e}")


def save_data(df: pd.DataFrame, file_path: str):
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {file_path}: {e}")

from pathlib import Path


def find_data_file() -> str:
    candidates = [
        Path("data/data.csv"),
        Path("data/external/data.csv"),
    ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    external_path = Path("data/external")
    if external_path.is_dir():
        csv_files = sorted(external_path.rglob("*.csv"))
        if csv_files:
            return str(csv_files[0])

    raise FileNotFoundError(
        "Input data file not found. Add a CSV to 'data/data.csv' or 'data/external/*.csv', "
        "or configure make_dataset.py to use a valid path."
    )


def main():
    try:
        data_file_path = find_data_file()
        param_file_path = "params.yaml"
        raw_data_path = Path("data") / "raw"

        data = load_data(data_file_path)
        test_size = load_params(param_file_path)
        train_data, test_data = split_data(data, test_size)

        raw_data_path.mkdir(parents=True, exist_ok=True)
        save_data(train_data, str(raw_data_path / "train.csv"))
        save_data(test_data, str(raw_data_path / "test.csv"))
    except Exception as e:
        raise Exception(f"Error in data collection process: {e}")


if __name__ == "__main__":
    main()
