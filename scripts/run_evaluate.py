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

    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)
    cfg = dict(base_cfg)
    if cfg["tuned_model"] == "sigmoid_DS-DPO":
        cfg["output_file"] = os.path.join(cfg["base_output_dir"], cfg["tuned_model"],  f't_{cfg["temperature"]}_{cfg["method"]}', f'epoch_{cfg["num_train_epochs"]}_{cfg["distance_type"]}_{cfg["min_beta"]}_{cfg["max_beta"]}', cfg["output_filename"])
        cfg["predictions_file"] = os.path.join(cfg["base_predictions_dir"], cfg["tuned_model"],  f't_{cfg["temperature"]}_{cfg["method"]}', f'epoch_{cfg["num_train_epochs"]}_{cfg["distance_type"]}_{cfg["min_beta"]}_{cfg["max_beta"]}', cfg["predictions_filename"])
        cfg["model_name"] = cfg["tuned_model"]
        cfg["sample_method"] = f't_{cfg["temperature"]}_{cfg["method"]}_epoch_{cfg["num_train_epochs"]}_{cfg["distance_type"]}_{cfg["min_beta"]}_{cfg["max_beta"]}'
   
    elif cfg["tuned_model"] == "DS-DPO":
        cfg["output_file"] = os.path.join(cfg["base_output_dir"], cfg["tuned_model"],  f'{cfg["method"]}', f'epoch_{cfg["num_train_epochs"]}_{cfg["distance_type"]}_{cfg["min_beta"]}_{cfg["max_beta"]}', cfg["output_filename"])
        cfg["predictions_file"] = os.path.join(cfg["base_predictions_dir"], cfg["tuned_model"],  f'{cfg["method"]}', f'epoch_{cfg["num_train_epochs"]}_{cfg["distance_type"]}_{cfg["min_beta"]}_{cfg["max_beta"]}', cfg["predictions_filename"])
        cfg["model_name"] = cfg["tuned_model"]
        cfg["sample_method"] = f'{cfg["method"]}_epoch_{cfg["num_train_epochs"]}_{cfg["distance_type"]}_{cfg["min_beta"]}_{cfg["max_beta"]}'
    
    elif cfg["tuned_model"] == "S-DPO":
        cfg["output_file"] = os.path.join(cfg["base_output_dir"], cfg["tuned_model"], cfg["method"], f'epoch_{cfg["num_train_epochs"]}_beta_{cfg["beta"]}',  cfg["output_filename"])
        cfg["predictions_file"] = os.path.join(cfg["base_predictions_dir"], cfg["tuned_model"], cfg["method"], f'epoch_{cfg["num_train_epochs"]}_beta_{cfg["beta"]}', cfg["predictions_filename"])
        cfg["model_name"] = cfg["tuned_model"]
        cfg["sample_method"] = f'{cfg["method"]}_epoch_{cfg["num_train_epochs"]}_{cfg["distance_type"]}_beta{cfg["beta"]}'
    evaluate_metrics(cfg)
    
    # config_path = os.path.join(os.path.dirname(__file__), "../configs/eval_config.yml")
    # config = load_config(config_path)
    # evaluate_metrics(config)

if __name__ == "__main__":
    main()
