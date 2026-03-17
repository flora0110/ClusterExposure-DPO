#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
# from src.models.dpo_trainer import train_dpo
from src.models.dpo_trainer_CHES_set import train_dpo
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
    # 初始化 argparse
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    config = load_config(config_path)
    train_dpo(config)


if __name__ == "__main__":
    # 使用 fire 啟動 main 函式
    fire.Fire(main)