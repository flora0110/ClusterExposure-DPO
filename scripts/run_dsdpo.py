#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.models.sdpo_trainer import train_sdpo


STRATEGIES = {
    # "balanced_popularity",
    # "clusterin_high_clusterout_low",
    "dc_0.05_2.0",
}

D= "Div"
num_return_sequences= "10"
diversity_penalty= "1.0"
BASE_OUTPUT = f"/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/model/DS-DPO/{D}_{num_return_sequences}_{diversity_penalty}"
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

    for folder in STRATEGIES:
        cfg = dict(base_cfg)
        cfg["output_dir"] = os.path.join(BASE_OUTPUT, folder)

        # write a temp config
        tmp_cfg_path = f"/scratch/user/chuanhsin0110/ClusterExposure-DPO/configs/tmp/dsdpo_{folder}_config.yml"
        with open(tmp_cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        print(f"\n>>> Launching DPO for {folder} → folder `{cfg['output_dir']}`")

        # train_sdpo
        train_sdpo(cfg)