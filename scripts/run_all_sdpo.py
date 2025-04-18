#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import yaml
# from src.models.dpo_trainer import train_dpo


STRATEGIES = {
    # "balanced_popularity":       "neg_sampling_balanced_popularity",
    # "clusterin_high":            "neg_sampling_clusterin_high_exposure",
    # "clusterin_low":             "neg_sampling_clusterin_low_exposure",
    # "low_exposure":              "neg_sampling_low_exposure",
    # "high_exposure":             "neg_sampling_high_exposure",
    "clusterout_low":            "neg_sampling_clusterout_low_exposure",
    # "clusterout_high":           "neg_sampling_clusterout_high_exposure",
    # "clusterin_high_clusterout_low": "clusterin_high_clusterout_low",
    # "clusterin_low_clusterout_low": "clusterin_low_clusterout_low",

}

