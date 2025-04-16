#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.models.dpo_trainer import train_dpo

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

if __name__ == "__main__":
    # Set config file path relative to this script
    config_path = os.path.join(os.path.dirname(__file__), "../configs/dpo_config.yml")
    config = load_config(config_path)
    train_dpo(config)
