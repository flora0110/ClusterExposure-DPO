#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.models.sft_trainer import train_sft

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "../configs/sft_config.yml")
    config = load_config(config_path)
    train_sft(config)
