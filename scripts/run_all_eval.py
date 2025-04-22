#!/usr/bin/env python3
import sys
import os
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.evaluate import evaluate_metrics

STRATEGIES = {
    # "clusterout_low",
    # "clusterout_high",
    # "clusterin_high",
    # "clusterin_low",
    # "low_exposure",
    # "high_exposure",
    # "clusterin_high_clusterout_low",
    # "balanced_popularity",
    # "clusterin_low_clusterout_low",
    # "out_past_farthest",
         "past_farthest", "past_nearest",
        "chosen_farthest", "chosen_nearest",
        "in_chosen_farthest", "in_chosen_nearest",
        "out_chosen_farthest", "out_chosen_nearest",
        "in_past_farthest", "in_past_nearest",
        # "out_past_farthest", 
        "out_past_nearest",
}

D = "Div"
NUM_RETURN = "10"
DIVERSITY = "1.0"

BASE_PRED_DIR = (
    f"/scratch/user/chuanhsin0110/ClusterExposure-DPO"
    f"/experiments/predictions/lightgcn/{D}_{NUM_RETURN}_{DIVERSITY}/"
)
BASE_METRICS_DIR = (
    f"/scratch/user/chuanhsin0110/ClusterExposure-DPO"
    f"/experiments/metrics/lightgcn/{D}_{NUM_RETURN}_{DIVERSITY}"
)
BASE_EVAL_CFG = "/scratch/user/chuanhsin0110/ClusterExposure-DPO/configs/eval_config.yml"
TMP_CFG_DIR = "/scratch/user/chuanhsin0110/ClusterExposure-DPO/configs/tmp"

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    base_cfg = load_config(BASE_EVAL_CFG)

    os.makedirs(TMP_CFG_DIR, exist_ok=True)

    for strat in STRATEGIES:
        cfg = dict(base_cfg)

        pred_file = os.path.join(BASE_PRED_DIR, strat, "raw_results_1000.json")
        out_file = os.path.join(BASE_METRICS_DIR, strat, "eval_result.json")

        cfg["predictions_file"] = pred_file
        cfg["output_file"]      = out_file
        cfg["sample_method"]    = strat
        cfg["model_name"]    = "lightgcn"

        tmp_path = os.path.join(TMP_CFG_DIR, f"eval_{strat}_config.yml")
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"\n>>> Evaluating strategy `{strat}`")
        print(f"    predictions_file: {pred_file}")
        print(f"    output_file:      {out_file}")

        evaluate_metrics(cfg)

if __name__ == "__main__":
    main()


