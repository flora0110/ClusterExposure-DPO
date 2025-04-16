import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset.dataset_sample import sample_dataset

if __name__ == "__main__":
    sample_dataset(
        train_path="data/raw/Goodreads/train.json",
        valid_path="data/raw/Goodreads/valid.json",
        test_path="data/raw/Goodreads/test.json",
        output_dir="data/sampled/Goodreads",
        train_size=1024,
        valid_size=256,
        test_size=1000
    )
