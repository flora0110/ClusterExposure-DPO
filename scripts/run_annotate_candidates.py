#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.dataset.annotate_candidates_with_metadata import annotate_candidates_with_metadata

def load_config(config_path: str) -> dict:
    """
    Load the YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML config file.
    
    Returns:
        dict: The configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/annotate_candidates_config.yml")
    config = load_config(config_path)
    annotate_candidates_with_metadata(config)

if __name__ == "__main__":
    main()
