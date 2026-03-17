#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
# from src.models.sft_follow_SPRec_set_trainer import train_sft
from re_run_src.models.sft_trainer import train_sft
import fire

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main(config_path):
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    train_sft(config)


if __name__ == "__main__":
    fire.Fire(main)
