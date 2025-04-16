import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.io_utils import safe_write_json

sample_data = {"foo": "bar"}
safe_write_json("data/sampled/Goodreads/train_sample.json", sample_data)
