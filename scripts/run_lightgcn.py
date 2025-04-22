#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.cf.lightgcn_trainer import lightgcn_trainer

def load_config(config_path):
    """
    Load YAML configuration.

    Args:
        config_path (str): Path to the config YAML.
    
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/lightgcn_config.yml")
    config = load_config(config_path)

    lightgcn_trainer(config)

if __name__ == "__main__":
    main()
