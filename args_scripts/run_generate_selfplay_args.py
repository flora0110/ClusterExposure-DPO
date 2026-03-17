#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.dataset.selfplay_generate_negatives_SPRecSet import generate_selfplay_negatives
import fire
def load_config(config_path):
    """
    Load the YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main(config_path):
    # Assume the config file is located in ../configs/selfplay_config.yml relative to this script.
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    config = load_config(config_path)
    generate_selfplay_negatives(config)

if __name__ == "__main__":
    fire.Fire(main)