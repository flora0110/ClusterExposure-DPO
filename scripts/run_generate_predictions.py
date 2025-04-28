#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.inference.generate_predict_batch import generate_predictions

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "../configs/predict_config.yml")

    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)
    cfg = dict(base_cfg)

    if cfg["tuned_model"] == "DS-DPO" or cfg["tuned_model"] == "sigmoid_DS-DPO":
        cfg["output_dir"] = os.path.join(cfg["base_output_dir"], cfg["tuned_model"], cfg["method"], f'epoch_{cfg["num_train_epochs"]}_{cfg["distance_type"]}_{cfg["min_beta"]}_{cfg["max_beta"]}')
        cfg["finetuned_path"] = os.path.join(cfg["base_finetuned_path"], cfg["tuned_model"], cfg["method"], f'epoch_{cfg["num_train_epochs"]}_{cfg["distance_type"]}_{cfg["min_beta"]}_{cfg["max_beta"]}')
    
    elif cfg["tuned_model"] == "S-DPO":
        cfg["output_dir"] = os.path.join(cfg["base_output_dir"], cfg["tuned_model"], cfg["method"],  f'epoch_{cfg["num_train_epochs"]}_beta_{cfg["beta"]}')
        cfg["finetuned_path"] = os.path.join(cfg["base_finetuned_path"], cfg["tuned_model"], cfg["method"], f'epoch_{cfg["num_train_epochs"]}_beta_{cfg["beta"]}')

    # cfg["output_dir"] = os.path.join(cfg["base_output_dir"], cfg["tuned_model"], cfg["method"], f'{cfg["distance_type"]}_{cfg["min_beta"]}_{cfg["max_beta"]}')
    # cfg["finetuned_path"] = os.path.join(cfg["base_finetuned_path"], cfg["tuned_model"], cfg["method"], f'{cfg["distance_type"]}_{cfg["min_beta"]}_{cfg["max_beta"]}')
    # cfg["output_dir"] = os.path.join(cfg["base_output_dir"], cfg["tuned_model"], cfg["method"])
    # cfg["finetuned_path"] = os.path.join(cfg["base_finetuned_path"], cfg["tuned_model"], cfg["method"])
    generate_predictions(cfg)


    # config_path = os.path.join(os.path.dirname(__file__), "../configs/predict_config.yml")
    # config = load_config(config_path)
    # generate_predictions(config)
