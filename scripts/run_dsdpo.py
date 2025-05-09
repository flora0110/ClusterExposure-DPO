#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.models.dsdpo_trainer import train_dsdpo


STRATEGIES = {
    # "balanced_popularity",
    # "clusterin_high_clusterout_low",
    "dpc_0.05_2.0",
}

D= "Div"
num_return_sequences= "5"
diversity_penalty= "1.0"
BASE_OUTPUT = f"/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/model/sigmoid_DS-DPO/t_1.0_{D}_{num_return_sequences}_{diversity_penalty}"
BASE_CONFIG = "/scratch/user/chuanhsin0110/ClusterExposure-DPO/configs/dsdpo_config.yml"


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
    with open(BASE_CONFIG) as f:
        base_cfg = yaml.safe_load(f)

    cfg = dict(base_cfg)
    cfg["output_dir"] = os.path.join(BASE_OUTPUT, f'epoch_{cfg["num_train_epochs"]}_{cfg["distance_type"]}_{cfg["min_beta"]}_{cfg["max_beta"]}')

    # write a temp config

    # train_sdpo
    train_dsdpo(cfg)