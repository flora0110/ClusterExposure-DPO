import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
from pathlib import Path
import json
from src.utils.io_utils import safe_write_json


def sample_and_save(input_path, size, save_path):
    """
    Sample a subset of the dataset and save to a new file.

    Args:
        input_path (str or Path): Path to the input JSON file.
        size (int): Number of samples to draw.
        save_path (str or Path): Path to save the sampled dataset.
    """
    input_path = Path(input_path)
    save_path = Path(save_path)
    with open(input_path, "r") as f:
        data = json.load(f)
    sampled = random.sample(data, min(size, len(data)))
    safe_write_json(save_path, sampled)
    print(f"[✓] Saved {len(sampled)} samples to {save_path}")


def sample_dataset(train_path, valid_path, test_path, output_dir,
                   train_size=1024, valid_size=256, test_size=1000):
    """
    Sample train/valid/test sets from the full dataset and save to output directory.

    Args:
        train_path (str): Path to the full training data.
        valid_path (str): Path to the full validation data.
        test_path (str): Path to the full test data.
        output_dir (str): Directory to save sampled datasets.
        train_size (int): Number of train samples.
        valid_size (int): Number of valid samples.
        test_size (int): Number of test samples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_and_save(train_path, train_size, output_dir / "train.json")
    sample_and_save(valid_path, valid_size, output_dir / "valid.json")
    sample_and_save(test_path, test_size, output_dir / "test.json")
