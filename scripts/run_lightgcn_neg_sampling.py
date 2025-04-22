#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.dataset.lightgcn_neg_sampling import lightgcn_neg_sampling

def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.
    
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/lightgcn_neg_sampling_config.yml")
    config = load_config(config_path)
    lightgcn_neg_sampling(config)

if __name__ == "__main__":
    main()
