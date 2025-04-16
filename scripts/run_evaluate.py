#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.evaluation.evaluate import evaluate_metrics

def load_config(config_path):
    """
    Load the YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/eval_config.yml")
    config = load_config(config_path)
    evaluate_metrics(config)

if __name__ == "__main__":
    main()
