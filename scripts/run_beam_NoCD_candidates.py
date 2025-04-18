#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.dataset.beam_nocd_generate_candidates import beam_nocd_generate_candidtate

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
    config_path = os.path.join(os.path.dirname(__file__), "../configs/beam_nocd_config.yml")
    config = load_config(config_path)

    beam_nocd_generate_candidtate(config)

if __name__ == "__main__":
    main()
