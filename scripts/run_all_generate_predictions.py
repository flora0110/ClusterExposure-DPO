#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.inference.generate_predict_batch import generate_predictions

STRATEGIES = {
    # "clusterout_low",
    # "clusterout_high",
    # "clusterin_high",
    # "clusterin_low",
    # "low_exposure",
    # "high_exposure",
    # "clusterin_high_clusterout_low",
    "balanced_popularity",
    "clusterin_low_clusterout_low",
}

D= "Div"
num_return_sequences= "10"
diversity_penalty= "1.0"
BASE_OUTPUT = f"/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/predictions/ClusterExposure_model/{D}_{num_return_sequences}_{diversity_penalty}"
BASE_INPUT = f"/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/model/ClusterExposure_model/{D}_{num_return_sequences}_{diversity_penalty}"
BASE_CONFIG = "/scratch/user/chuanhsin0110/ClusterExposure-DPO/configs/predict_config.yml"


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    with open(BASE_CONFIG) as f:
        base_cfg = yaml.safe_load(f)

    for folder in STRATEGIES:
        cfg = dict(base_cfg)
        cfg["output_dir"] = os.path.join(BASE_OUTPUT, folder)

        cfg["finetuned_path"] = os.path.join(BASE_INPUT, folder)
        # cfg["finetuned_path"] = os.path.join(cfg["finetuned_path"], "final_model")

        # write a temp config
        tmp_cfg_path = f"/scratch/user/chuanhsin0110/ClusterExposure-DPO/configs/tmp/predict_{folder}_config.yml"
        with open(tmp_cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        print(f"\n>>> Launching DPO for {folder} → folder `{cfg['output_dir']}`")

        config = load_config(tmp_cfg_path)
        generate_predictions(config)
