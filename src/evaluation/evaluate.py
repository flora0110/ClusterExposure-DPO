import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import re
import math
import json
import torch
import pandas as pd
from tqdm import tqdm
from collections import Counter
from src.utils.io_utils import safe_load_json, safe_write_json, prepare_output_dir

def read_json(json_file: str) -> dict:
    """
    Read a JSON file and return its content.
    
    Args:
        json_file (str): The path to the JSON file.
    
    Returns:
        dict: Parsed JSON content.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def batch(list_obj, batch_size=1):
    """
    Yield batches of a list.
    
    Args:
        list_obj (list): The list to be divided.
        batch_size (int): Number of items per batch.
    
    Yields:
        list: A sublist containing at most batch_size elements.
    """
    chunk_size = (len(list_obj) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list_obj[batch_size * i: batch_size * (i + 1)]

def sum_of_first_i_keys(sorted_dic, i):
    """
    Sum the values of the first i keys in a sorted dictionary.
    
    Args:
        sorted_dic (dict): The dictionary sorted by value.
        i (int): Number of keys to sum.
    
    Returns:
        float: The sum of the first i values.
    """
    keys = list(sorted_dic.values())[:i]
    return sum(keys)

def gh(category: str, test_data, eval_dir: str) -> list:
    """
    Compute and normalize genre distribution based on the input test data.
    
    Args:
        category (str): The category name (e.g., "Goodreads").
        test_data (list): A list of test data entries.
        eval_dir (str): Directory containing evaluation files for the category.
    
    Returns:
        list: A list of normalized genre values.
    """
    notin_count = 0
    in_count = 0
    name2genre = read_json(os.path.join(eval_dir, category, "name2genre.json"))
    genre_dict = read_json(os.path.join(eval_dir, category, "genre_dict.json"))
    for data in tqdm(test_data, desc="Processing category data..."):
        input_text = data['prompt']
        names = re.findall(r'"([^"]+)"', input_text)
        for name in names:
            if name in name2genre:
                in_count += 1
                genres = name2genre[name]
            else:
                notin_count += 1
                continue
            select_genres = []
            for genre in genres:
                if genre in genre_dict:
                    select_genres.append(genre)
            if len(select_genres) > 0:
                for genre in select_genres:
                    genre_dict[genre] += 1 / len(select_genres)
    gh_values = [genre_dict[x] for x in genre_dict]
    gh_normalize = [x / sum(gh_values) for x in gh_values]
    print(f"InCount: {in_count}\nNotinCount: {notin_count}")
    return gh_normalize

def update_csv(dataset_name: str,
               model_name: str,
               sample_method: str,
               metrics_dict: dict,
               csv_file: str):
    """
    Update (or create) a CSV of evaluation metrics, keyed by Dataset, Model, and SampleMethod.
    If a row with the same (Dataset, Model, SampleMethod) exists, its metric columns will be overwritten.
    Otherwise a new row is appended.
    """
    # Make sure directory exists
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    key_cols = ["Dataset", "Model", "SampleMethod"]
    metric_cols = list(metrics_dict.keys())

    if not os.path.exists(csv_file):
        # Build empty DataFrame with all needed columns
        df = pd.DataFrame(columns=key_cols + metric_cols)
        # New row
        new_row = {"Dataset": dataset_name,
                   "Model": model_name,
                   "SampleMethod": sample_method}
        new_row.update(metrics_dict)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.read_csv(csv_file)
        # Ensure the SampleMethod column exists
        if "SampleMethod" not in df.columns:
            df["SampleMethod"] = None

        # Condition: same dataset, model, and sample_method
        cond = (
            (df["Dataset"] == dataset_name) &
            (df["Model"] == model_name) &
            (df["SampleMethod"] == sample_method)
        )

        if not cond.any():
            # Append new row
            new_row = {col: None for col in df.columns}
            new_row.update({
                "Dataset": dataset_name,
                "Model": model_name,
                "SampleMethod": sample_method,
                **metrics_dict
            })
            # Add any missing metric columns
            for m in metrics_dict:
                if m not in df.columns:
                    df[m] = None
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # Update existing row
            for metric, value in metrics_dict.items():
                if metric not in df.columns:
                    df[metric] = None
                df.loc[cond, metric] = value

    # Write back
    df.to_csv(csv_file, index=False)
    print(f"CSV updated: {csv_file}")

def evaluate_metrics(config):
    """
    Main evaluation function to compute metrics from prediction results.
    
    This function performs the following steps:
      - Loads evaluation files (e.g., id2name, name2id, embeddings, name2genre, genre_dict) from a specified directory.
      - Loads prediction results from a JSON file.
      - Uses a SentenceTransformer model to encode predicted text.
      - Computes pairwise distances and ranks.
      - Calculates evaluation metrics such as NDCG, HR, diversity, fairness metrics (DGU, MGU, ORRatio),
        and the ratio of predictions not found in the genre mapping.
      - Saves evaluation results as a JSON file and optionally updates a CSV file.
    
    Args:
        config (dict): Configuration dictionary with keys:
            - eval_dir: Directory containing evaluation files.
            - category: Category name (e.g., "Goodreads").
            - predictions_file: Path to the JSON file with predictions.
            - output_file: Path to output evaluation JSON.
            - exp_csv: (Optional) Path to CSV file for logging metrics.
            - topk: Top-k value to use for ranking metrics.
            - sbert_model_path: Path to the SentenceTransformer model.
            - model_name: (For record) Model name used in evaluation.
            - sample_method: (For record) Negative sampling method.
    
    Returns:
        None.
    """
    category = config["category"]
    eval_dir = config["eval_dir"]

    id2name = safe_load_json(os.path.join(eval_dir, category, "id2name.json"))
    name2id = safe_load_json(os.path.join(eval_dir, category, "name2id.json"))
    embeddings = torch.load(os.path.join(eval_dir, category, "embeddings.pt"))
    name2genre = safe_load_json(os.path.join(eval_dir, category, "name2genre.json"))
    genre_dict = safe_load_json(os.path.join(eval_dir, category, "genre_dict.json"))

    test_data = safe_load_json(config["predictions_file"])

    # Load SentenceTransformer model for encoding prediction text.
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer(config["sbert_model_path"])
    print("SentenceTransformer model loaded successfully.")

    embeddings = torch.tensor(embeddings).cuda()
    text = []
    for entry in tqdm(test_data, desc="Extracting prediction names"):
        if len(entry["prediction"]) > 0:
            match = re.search(r'"([^"]+)"', entry['prediction'])
            if match:
                name = match.group(1)
                text.append(name)
            else:
                # Fallback: take the first line of prediction
                text.append(entry['prediction'].split('\n', 1)[0])
        else:
            text.append("NAN")
            print("Empty prediction!")
    
    pred_not_in_count = sum(1 for name in text if name not in name2genre)
    pred_not_in_ratio = pred_not_in_count / len(text)
    print(f"Prediction not in name2genre ratio: {pred_not_in_ratio:.4f}")

    # Encode prediction text using SentenceTransformer.
    predict_embeddings = []
    for batch_text in batch(text, 8):
        encoded = sbert_model.encode(batch_text)
        predict_embeddings.append(torch.tensor(encoded))
    predict_embeddings = torch.cat(predict_embeddings, dim=0).cuda()
    print("Prediction embeddings size:", predict_embeddings.size())
    
    # Compute pairwise distances between prediction embeddings and the precomputed embeddings.
    dist = torch.cdist(predict_embeddings, embeddings, p=2)
    batch_size_ = 1
    num_batches = (dist.size(0) + batch_size_ - 1) // batch_size_
    print(f"Number of batches for ranking: {num_batches}")
    rank_list = []
    for i in tqdm(range(num_batches), desc="Processing Batches"):
        start_idx = i * batch_size_
        end_idx = min((i + 1) * batch_size_, dist.size(0))
        batch_dist = dist[start_idx:end_idx]
        batch_rank = batch_dist.argsort(dim=-1).argsort(dim=-1)
        torch.cuda.empty_cache()
        rank_list.append(batch_rank)
    rank_list = torch.cat(rank_list, dim=0)
    print(f"Rank list length: {len(rank_list)}")
    
    topk = int(config["topk"])
    S_ndcg = 0
    S_hr = 0
    diversity_set = set()
    diversity_dic = {}
    total = len(test_data)
    for i in tqdm(range(len(test_data)), desc="Calculating Metrics"):
        rank = rank_list[i]
        target_name = test_data[i]['ground_truth'].strip().strip('"')
        if target_name in name2id:
            target_id = name2id[target_name]
        else:
            continue
        rankId = rank[target_id]
        if rankId < topk:
            S_ndcg += (1 / math.log(rankId + 2))
            S_hr += 1
        for j in range(topk):
            topi_id = torch.argwhere(rank == j).item()
            topi_name = id2name[str(topi_id)]
            if topi_name in name2genre:
                topi_genre = name2genre[topi_name]
                select_genres = [genre for genre in topi_genre if genre in genre_dict]
                if len(select_genres) > 0:
                    for genre in select_genres:
                        genre_dict[genre] += 1 / len(select_genres)
            diversity_set.add(torch.argwhere(rank == j).item())
            diversity_dic[torch.argwhere(rank == j).item()] = diversity_dic.get(torch.argwhere(rank == j).item(), 0) + 1
    NDCG = S_ndcg / len(test_data) / (1 / math.log(2))
    HR = S_hr / len(test_data)
    diversity = len(diversity_set)
    
    gh_genre = gh(category, test_data, eval_dir)
    gp_genre = [genre_dict[x] for x in genre_dict]
    gp_genre = [x / sum(gp_genre) for x in gp_genre]
    dis_genre = [gp_genre[i] - gh_genre[i] for i in range(len(gh_genre))]
    DGU_genre = max(dis_genre) - min(dis_genre)
    dis_abs_genre = [abs(x) for x in dis_genre]
    MGU_genre = sum(dis_abs_genre) / len(dis_abs_genre)

    eval_dic = {
        "method_name": config.get("model_name", "Unknown"),
        "sample_method": config.get("sample_method", "Unknown"),
        "topK": topk,
        "Dis_genre": dis_abs_genre,
        "NDCG": NDCG,
        "HR": HR,
        "diversity": diversity,
        "DivRatio": diversity / (total * topk),
        "DGU": DGU_genre,
        "MGU": MGU_genre,
        "Predict_NotIn_Ratio": pred_not_in_ratio
    }
    sorted_dic = dict(sorted(diversity_dic.items(), key=lambda item: item[1], reverse=True))
    eval_dic["ORRatio"] = sum_of_first_i_keys(sorted_dic, 3) / (topk * total)
    print(f"ORRatio: {eval_dic['ORRatio']}")
    
    output_file = config["output_file"]
    output_dir = os.path.dirname(output_file)
    prepare_output_dir(output_dir, None, allow_existing=True)
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append(eval_dic)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, separators=(',', ': '), indent=2)
    
    # Optionally update CSV if provided in config.
    if config.get("exp_csv"):
        metric_dic = {
            f"MGU@{topk}": eval_dic["MGU"],
            f"DGU@{topk}": eval_dic["DGU"],
            f"DivRatio@{topk}": eval_dic["DivRatio"],
            f"ORRatio@{topk}": eval_dic["ORRatio"],
            f"PredictNotInRatio@{topk}": eval_dic["Predict_NotIn_Ratio"],
            f"NDCG@{topk}": eval_dic["NDCG"],
            f"HR@{topk}": eval_dic["HR"],
            "Predict_NotIn_Ratio": eval_dic["Predict_NotIn_Ratio"],
        }
        update_csv(category, config.get("model_name", "Unknown"), config.get("sample_method", "Unknown"), metric_dic, config["exp_csv"])
    print("Evaluation complete.")

if __name__ == "__main__":
    print("Starting evaluation...")
    # This main() function is intended to be called from the runner script.
