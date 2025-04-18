import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import random
import re
from collections import Counter
from src.utils.io_utils import safe_load_json, safe_write_json, prepare_output_dir

def format_prompt(instruction, input_text):
    """
    Format a prompt string with an instruction and optional input.

    Args:
        instruction (str): The instruction text.
        input_text (str): Supplementary input text.

    Returns:
        str: The formatted prompt.
    """
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text.strip():
        prompt += f"### Input:\n{input_text}\n"
    prompt += "### Response:"
    return prompt

# -------------------------
# Negative Sampling Strategy Functions
# -------------------------

def sample_balanced_popularity(rejected_details: list) -> list:
    """
    Balanced Popularity-based Negative Sampling:
    Sort rejected_details by exposure_count in descending order,
    split the list into two halves, and randomly select one candidate from each half.
    
    Args:
        rejected_details (list): List of dictionaries with keys including "rejected" and "exposure_count".
    
    Returns:
        list: A list of selected candidate strings (may contain two elements).
    """
    if not rejected_details:
        return []
    sorted_details = sorted(rejected_details, key=lambda x: x.get("exposure_count", 0), reverse=True)
    n = len(sorted_details)
    if n == 1:
        return [sorted_details[0]["rejected"]]
    split_idx = n // 2
    first_half = sorted_details[:split_idx] if split_idx > 0 else sorted_details
    second_half = sorted_details[split_idx:] if split_idx < n else sorted_details
    sample1 = random.choice(first_half)["rejected"] if first_half else None
    sample2 = random.choice(second_half)["rejected"] if second_half else None
    samples = []
    if sample1 is not None:
        samples.append(sample1)
    if sample2 is not None and sample2 not in samples:
        samples.append(sample2)
    return samples

def sample_cluster_in_high_exposure(rejected_details: list, interest_clusters: list) -> str:
    """
    ClusterIn-HighExposure Negative Sampling:
    Select candidates from rejected_details that have a non-empty intersection with interest_clusters,
    and return the candidate with the highest exposure_count.
    
    Args:
        rejected_details (list): List of candidate dictionaries.
        interest_clusters (list): List of genres representing user's interests.
    
    Returns:
        str: The selected candidate's "rejected" field, or an empty string if none.
    """
    candidates = [item for item in rejected_details if set(item.get("genre", [])) & set(interest_clusters)]
    if not candidates:
        return ""
    candidate = max(candidates, key=lambda x: x.get("exposure_count", 0))
    return candidate["rejected"]

def sample_cluster_in_low_exposure(rejected_details: list, interest_clusters: list) -> str:
    """
    ClusterIn-LowExposure Negative Sampling:
    Select candidates from rejected_details that have a non-empty intersection with interest_clusters,
    and return the candidate with the lowest exposure_count.
    
    Args:
        rejected_details (list): List of candidate dictionaries.
        interest_clusters (list): List of genres representing user's interests.
    
    Returns:
        str: The selected candidate's "rejected" field, or an empty string if none.
    """
    candidates = [item for item in rejected_details if set(item.get("genre", [])) & set(interest_clusters)]
    if not candidates:
        return ""
    candidate = min(candidates, key=lambda x: x.get("exposure_count", float('inf')))
    return candidate["rejected"]

def sample_low_exposure(rejected_details: list) -> str:
    """
    Negative Sampling based on Low Exposure:
    Return the candidate with the minimum exposure_count.
    
    Args:
        rejected_details (list): List of candidate dictionaries.
    
    Returns:
        str: The selected candidate's "rejected" field.
    """
    candidate = min(rejected_details, key=lambda x: x.get("exposure_count", float('inf')))
    return candidate["rejected"]

def sample_high_exposure(rejected_details: list) -> str:
    """
    Negative Sampling based on High Exposure:
    Return the candidate with the maximum exposure_count.
    
    Args:
        rejected_details (list): List of candidate dictionaries.
    
    Returns:
        str: The selected candidate's "rejected" field.
    """
    candidate = max(rejected_details, key=lambda x: x.get("exposure_count", 0))
    return candidate["rejected"]

def sample_cluster_out_low_exposure(rejected_details: list, interest_clusters: list) -> str:
    """
    ClusterOut-LowExposure Negative Sampling:
    From candidates with no intersection with interest_clusters (if any), select the one with the lowest exposure_count.
    If none, select from candidates with the least overlap.
    
    Args:
        rejected_details (list): List of candidate dictionaries.
        interest_clusters (list): User's interest genres.
    
    Returns:
        str: The selected candidate's "rejected" field.
    """
    candidates_no_intersection = []
    all_candidates = []
    interest_set = set(interest_clusters)
    for item in rejected_details:
        item_genres = set(item.get("genre", []))
        intersection_count = len(item_genres & interest_set)
        all_candidates.append((item, intersection_count))
        if intersection_count == 0:
            candidates_no_intersection.append(item)
    if candidates_no_intersection:
        candidate = min(candidates_no_intersection, key=lambda x: x.get("exposure_count", float('inf')))
    else:
        min_intersection = min(intersection for _, intersection in all_candidates)
        candidates_least_overlap = [item for item, intersection in all_candidates if intersection == min_intersection]
        candidate = min(candidates_least_overlap, key=lambda x: x.get("exposure_count", float('inf')))
    return candidate["rejected"]

def sample_cluster_out_high_exposure(rejected_details: list, interest_clusters: list) -> str:
    """
    ClusterOut-HighExposure Negative Sampling:
    From candidates with no intersection with interest_clusters (if any), select the one with the highest exposure_count.
    If none, select from candidates with the highest overlap.
    
    Args:
        rejected_details (list): List of candidate dictionaries.
        interest_clusters (list): User's interest genres.
    
    Returns:
        str: The selected candidate's "rejected" field.
    """
    candidates_no_intersection = []
    all_candidates = []
    interest_set = set(interest_clusters)
    for item in rejected_details:
        item_genres = set(item.get("genre", []))
        intersection_count = len(item_genres & interest_set)
        all_candidates.append((item, intersection_count))
        if intersection_count == 0:
            candidates_no_intersection.append(item)
    if candidates_no_intersection:
        candidate = max(candidates_no_intersection, key=lambda x: x.get("exposure_count", 0))
    else:
        max_intersection = max(intersection for _, intersection in all_candidates)
        candidates_most_overlap = [item for item, intersection in all_candidates if intersection == max_intersection]
        candidate = max(candidates_most_overlap, key=lambda x: x.get("exposure_count", 0))
    return candidate["rejected"]

def sample_clusterin_high_clusterout_low(rejected_details: list, interest_clusters: list) -> list:
    """
    Clustering-Exposure Balanced Sampling:
    Combine two strategies: from candidates overlapping with interest_clusters, take the one with highest exposure;
    from candidates not overlapping, take the one with lowest exposure.
    
    Args:
        rejected_details (list): List of candidate dictionaries.
        interest_clusters (list): User's interest genres.
    
    Returns:
        list: A list containing the selected candidate(s).
    """
    sample_in = sample_cluster_in_high_exposure(rejected_details, interest_clusters)
    sample_out = sample_cluster_out_low_exposure(rejected_details, interest_clusters)
    samples = []
    if sample_in:
        samples.append(sample_in)
    if sample_out:
        samples.append(sample_out)
    return samples


def sample_clusterin_low_clusterout_low(rejected_details: list, interest_clusters: list) -> list:
    """
    Clustering-Exposure Balanced Sampling:
    Combine two strategies: from candidates overlapping with interest_clusters, take the one with highest exposure;
    from candidates not overlapping, take the one with lowest exposure.
    
    Args:
        rejected_details (list): List of candidate dictionaries.
        interest_clusters (list): User's interest genres.
    
    Returns:
        list: A list containing the selected candidate(s).
    """
    sample_in = sample_cluster_in_low_exposure(rejected_details, interest_clusters)
    sample_out = sample_cluster_out_low_exposure(rejected_details, interest_clusters)
    samples = []
    if sample_in:
        samples.append(sample_in)
    if sample_out:
        samples.append(sample_out)
    return samples

# -------------------------
# Main Processing Function
# -------------------------

def process_negative_sampling(config: dict, data_type: str) -> dict:
    """
    Process annotated candidate data and generate negative sampling outputs
    for eight strategies.
    
    The function performs the following:
      1. Load the annotated candidate data (processed from the self-play step).
      2. For each record, extract the user interest clusters (assumed to exist in the record).
      3. For each negative sampling strategy, apply the corresponding function on the record's "rejected_details".
      4. Collect results for each strategy in separate lists.
      5. Return a dictionary mapping strategy names to the list of processed records.
      
    Args:
        config (dict): Configuration dictionary with keys:
            - input_file: Path to the annotated candidate JSON file.
            - data_type: "train" or "valid" (used for record naming).
            - (others can be added if needed)
    
    Returns:
        dict: A mapping where keys are strategy identifiers and values are lists of records.
    """
    input_dir = config["input_dir"]  # 請在 config 中指定，例如：/scratch/…/experiments/data/beam_cd_candidates
    param_tag = f"{config['D']}_{config['num_return_sequences']}_{config['diversity_penalty']}"
    input_dir = prepare_output_dir(input_dir, param_tag, allow_existing=True)
    input_path = os.path.join(input_dir, config[f"input_{data_type}_filename"])
    # valid_input_path = os.path.join(input_dir, config["input_valid_filename"])

    data = safe_load_json(input_path)
    # valid_data = safe_load_json(valid_input_path)
    data_strategies = {}
    # valid_data_strategies = {}

    
    # 初始化八種策略的資料儲存列表
    data_strategies["balanced_popularity"] = []
    data_strategies["clusterin_high"] = []
    data_strategies["clusterin_low"] = []
    data_strategies["low_exposure"] = []
    data_strategies["high_exposure"] = []
    data_strategies["clusterout_low"] = []
    data_strategies["clusterout_high"] = []
    data_strategies["clusterin_high_clusterout_low"] = []
    data_strategies["clusterin_low_clusterout_low"] = []

    # valid_data_strategies["balanced_popularity"] = []
    # valid_data_strategies["clusterin_high"] = []
    # valid_data_strategies["clusterin_low"] = []
    # valid_data_strategies["low_exposure"] = []
    # valid_data_strategies["high_exposure"] = []
    # valid_data_strategies["clusterout_low"] = []
    # valid_data_strategies["clusterout_high"] = []
    # valid_data_strategies["clusterin_high_clusterout_low"] = []
    # valid_data_strategies["clusterin_low_clusterout_low"] = []
    
    # 遍歷每筆記錄，根據該記錄的 interest_clusters 與 rejected_details，獲取各策略結果
    for record in data:
        prompt = format_prompt(record["instruction"], record["input"])
        chosen = record["chosen"]
        interest_clusters = record.get("interest_clusters", [])
        rejected_details = record.get("rejected_details", [])
        
        # 對每種策略計算負樣本結果
        res_balanced = sample_balanced_popularity(rejected_details)
        res_clusterin_high = sample_cluster_in_high_exposure(rejected_details, interest_clusters)
        res_clusterin_low = sample_cluster_in_low_exposure(rejected_details, interest_clusters)
        res_low = sample_low_exposure(rejected_details)
        res_high = sample_high_exposure(rejected_details)
        res_clusterout_low = sample_cluster_out_low_exposure(rejected_details, interest_clusters)
        res_clusterout_high = sample_cluster_out_high_exposure(rejected_details, interest_clusters)
        res_clusterin_high_clusterout_low = sample_clusterin_high_clusterout_low(rejected_details, interest_clusters)
        res_clusterin_low_clusterout_low = sample_clusterin_low_clusterout_low(rejected_details, interest_clusters)
        
        # 建立基礎結果結構（每筆記錄均包含 prompt 和 chosen）
        base_record = {
            "prompt": prompt,
            "chosen": chosen
        }
        # 為每個策略建立記錄（若結果為空，記錄為空字串或空列表）
        data_strategies["balanced_popularity"].append({**base_record, "rejected": res_balanced})
        data_strategies["clusterin_high"].append({**base_record, "rejected": res_clusterin_high})
        data_strategies["clusterin_low"].append({**base_record, "rejected": res_clusterin_low})
        data_strategies["low_exposure"].append({**base_record, "rejected": res_low})
        data_strategies["high_exposure"].append({**base_record, "rejected": res_high})
        data_strategies["clusterout_low"].append({**base_record, "rejected": res_clusterout_low})
        data_strategies["clusterout_high"].append({**base_record, "rejected": res_clusterout_high})
        data_strategies["clusterin_high_clusterout_low"].append({**base_record, "rejected": res_clusterin_high_clusterout_low})
        data_strategies["clusterin_low_clusterout_low"].append({**base_record, "rejected": res_clusterin_low_clusterout_low})

    # # 遍歷每筆記錄，根據該記錄的 interest_clusters 與 rejected_details，獲取各策略結果
    # for record in valid_data:
    #     prompt = format_prompt(record["instruction"], record["input"])
    #     chosen = record["chosen"]
    #     interest_clusters = record.get("interest_clusters", [])
    #     rejected_details = record.get("rejected_details", [])
        
    #     # 對每種策略計算負樣本結果
    #     res_balanced = sample_balanced_popularity(rejected_details)
    #     res_clusterin_high = sample_cluster_in_high_exposure(rejected_details, interest_clusters)
    #     res_clusterin_low = sample_cluster_in_low_exposure(rejected_details, interest_clusters)
    #     res_low = sample_low_exposure(rejected_details)
    #     res_high = sample_high_exposure(rejected_details)
    #     res_clusterout_low = sample_cluster_out_low_exposure(rejected_details, interest_clusters)
    #     res_clusterout_high = sample_cluster_out_high_exposure(rejected_details, interest_clusters)
    #     res_clusterin_high_clusterout_low = sample_clusterin_high_clusterout_low(rejected_details, interest_clusters)
    #     res_clusterin_low_clusterout_low = sample_clusterin_low_clusterout_low(rejected_details, interest_clusters)
        
    #     # 建立基礎結果結構（每筆記錄均包含 prompt 和 chosen）
    #     base_record = {
    #         "prompt": prompt,
    #         "chosen": chosen
    #     }
    #     # 為每個策略建立記錄（若結果為空，記錄為空字串或空列表）
    #     valid_data_strategies["balanced_popularity"].append({**base_record, "rejected": res_balanced})
    #     valid_data_strategies["clusterin_high"].append({**base_record, "rejected": res_clusterin_high})
    #     valid_data_strategies["clusterin_low"].append({**base_record, "rejected": res_clusterin_low})
    #     valid_data_strategies["low_exposure"].append({**base_record, "rejected": res_low})
    #     valid_data_strategies["high_exposure"].append({**base_record, "rejected": res_high})
    #     valid_data_strategies["clusterout_low"].append({**base_record, "rejected": res_clusterout_low})
    #     valid_data_strategies["clusterout_high"].append({**base_record, "rejected": res_clusterout_high})
    #     valid_data_strategies["clusterin_high_clusterout_low"].append({**base_record, "rejected": res_clusterin_high_clusterout_low})
    #     valid_data_strategies["clusterin_low_clusterout_low"].append({**base_record, "rejected": res_clusterin_low_clusterout_low})
    
    return data_strategies #, valid_data_strategies

def cluster_exposure_neg_sampling(config):
    # Prepare output directory based on configuration
    # 例如：先用 prepare_output_dir 建立一個層級，然後依據策略名稱再建立子資料夾
    base_output_dir = config["output_dir_base"]  # 請在 config 中指定，例如：/scratch/…/experiments/data/beam_cd_candidates
    param_tag = f"{config['D']}_{config['num_return_sequences']}_{config['diversity_penalty']}"
    final_output_dir = os.path.join(base_output_dir, param_tag)
    final_output_dir = prepare_output_dir(final_output_dir, None, allow_existing=True)
    
    # Load annotated candidate file from configuration
    # config["input_file"] = config["input_file"]  # 例如：annotated_candidates.json
    # 選擇 data type (train/valid) 為檔案名稱中的一部分
    # data_type = config.get("data_type", "train")
        
    data_type = "train"

    # Process negative sampling across all strategies
    results = process_negative_sampling(config, data_type)

    # For each strategy, compute output file path and save the results
    for strat, records in results.items():
        strat_dir = os.path.join(final_output_dir, strat)
        if not os.path.exists(strat_dir):
            os.makedirs(strat_dir, exist_ok=True)
        out_file = os.path.join(strat_dir, f"{data_type}.json")
        safe_write_json(out_file, records)
        print(f"Saved {strat} results to {out_file}")


    data_type = "valid"

    # Process negative sampling across all strategies
    results = process_negative_sampling(config, data_type)

    # For each strategy, compute output file path and save the results
    for strat, records in results.items():
        strat_dir = os.path.join(final_output_dir, strat)
        if not os.path.exists(strat_dir):
            os.makedirs(strat_dir, exist_ok=True)
        out_file = os.path.join(strat_dir, f"{data_type}.json")
        safe_write_json(out_file, records)
        print(f"Saved {strat} results to {out_file}")

# if __name__ == "__main__":
#     # for standalone testing; in practice, use runner script
#     config = {
#         "input_file": "./eval/Goodreads/annotated_candidates.json",
#         "data_type": "train"
#     }
#     results = process_negative_sampling(config)
#     # For example，直接存取 high exposure sampling結果：
#     safe_write_json("./analysis/annotated_candidates_high_exposure.json", results["high_exposure"])
