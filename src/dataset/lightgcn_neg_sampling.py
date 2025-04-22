#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import re
import numpy as np
from tqdm import tqdm
from src.utils.io_utils import safe_load_json, safe_write_json, prepare_output_dir

def parse_titles(text: str):
    """Extract all substrings between double quotes."""
    return re.findall(r'"([^"]+)"', text)

def avg_emb(titles, book2idx, item_emb):
    """Compute the average embedding of a list of titles."""
    idxs = [book2idx.get(t, None) for t in titles]
    idxs = [i for i in idxs if i is not None]
    if not idxs:
        return None
    return item_emb[idxs].mean(axis=0)

def l2(a, b):
    """Compute Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)

def format_prompt(instruction, input_text):
    """
    Format a prompt string with an instruction and optional input.
    """
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text.strip():
        prompt += f"### Input:\n{input_text}\n"
    prompt += "### Response:"
    return prompt

def process_negative_sampling(config: dict, split: str, item_emb: np.ndarray, book2idx: dict):
    """
    Generate 12 CF‑based negative samples for each record,
    using precomputed item_emb and book2idx mapping.
    """
    # prepare input path
    in_base = prepare_output_dir(config["input_dir"], 
                                 f"{config['D']}_{config['num_return_sequences']}_{config['diversity_penalty']}",
                                 allow_existing=True)
    in_file = os.path.join(in_base, config[f"input_{split}_filename"])
    data = safe_load_json(in_file)

    # strategy names
    strat_names = [
        "past_farthest", "past_nearest",
        "chosen_farthest", "chosen_nearest",
        "in_chosen_farthest", "in_chosen_nearest",
        "out_chosen_farthest", "out_chosen_nearest",
        "in_past_farthest", "in_past_nearest",
        "out_past_farthest", "out_past_nearest",
    ]
    records = {name: [] for name in strat_names}

    # helper to pick max/min index
    def pick(dist_list, mode):
        valid = [(i, d) for i, d in enumerate(dist_list) if not np.isinf(d)]
        if not valid:
            return None
        return (max if mode=="max" else min)(valid, key=lambda x: x[1])[0]

    for rec in tqdm(data, desc=f"{split} sampling"):
        prompt = format_prompt(rec["instruction"], rec["input"])
        past_titles = parse_titles(rec["input"])
        chosen_title = parse_titles(rec["chosen"])[0]
        interest = set(rec.get("interest_clusters", []))
        details = rec.get("rejected_details", [])

        # compute centroids
        emb_past   = avg_emb(past_titles, book2idx, item_emb)
        emb_chosen = avg_emb([chosen_title],    book2idx, item_emb)

        # prepare candidate embeddings and cluster flags
        titles, c_emb, in_cl, out_cl = [], [], [], []
        for det in details:
            t = det["rejected"].strip().strip('"')
            titles.append(det["rejected"])
            idx = book2idx.get(t)
            if idx is None:
                c_emb.append(None); in_cl.append(False); out_cl.append(False)
            else:
                e = item_emb[idx]
                c_emb.append(e)
                genres = set(det.get("genre", []))
                in_flag = bool(genres & interest)
                in_cl.append(in_flag); out_cl.append(not in_flag)

        # compute distances
        dp = [l2(emb_past,   e) if emb_past is not None and e is not None else np.inf for e in c_emb]
        dc = [l2(emb_chosen, e) if e is not None else np.inf                     for e in c_emb]

        # select indices
        i1, i2 = pick(dp, "max"), pick(dp, "min")
        i3, i4 = pick(dc, "max"), pick(dc, "min")
        in_idx  = [i for i,f in enumerate(in_cl)  if f]
        out_idx = [i for i,f in enumerate(out_cl) if f]
        i5  = max(in_idx, key=lambda i: dc[i]) if in_idx else pick(dc, "max")
        i6  = min(in_idx, key=lambda i: dc[i]) if in_idx else pick(dc, "min")
        i7  = max(out_idx, key=lambda i: dc[i]) if out_idx else pick(dc, "max")
        i8  = min(out_idx, key=lambda i: dc[i]) if out_idx else pick(dc, "min")
        i9  = max(in_idx, key=lambda i: dp[i]) if in_idx else pick(dp, "max")
        i10 = min(in_idx, key=lambda i: dp[i]) if in_idx else pick(dp, "min")
        i11 = max(out_idx, key=lambda i: dp[i]) if out_idx else pick(dp, "max")
        i12 = min(out_idx, key=lambda i: dp[i]) if out_idx else pick(dp, "min")

        idx_map = dict(zip(strat_names, [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12]))
        base = {"prompt": prompt, "chosen": rec["chosen"]}
        for name, sel in idx_map.items():
            rej = titles[sel] if sel is not None else ""
            records[name].append({**base, "rejected": rej})

    return records

def lightgcn_neg_sampling(config: dict):
    """
    Run negative sampling for both 'train' and 'valid' splits,
    write out under {output_dir_base}/{D...}/{strategy}/{split}.json
    """
    # prepare output root
    root = os.path.join(config["output_dir_base"],
                        f"{config['D']}_{config['num_return_sequences']}_{config['diversity_penalty']}")
    os.makedirs(root, exist_ok=True)

    # load book2idx & item embeddings
    with open(config["book2idx"], "r") as f:
        book2idx = json.load(f)
    item_emb = np.load(config["item_emb"])  # shape (num_items, emb_dim)

    for split in ("train","valid"):
        results = process_negative_sampling(config, split, item_emb, book2idx)
        for strat, recs in results.items():
            out_dir = os.path.join(root, strat)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"{split}.json")
            safe_write_json(out_file, recs)
            print(f"Saved {split}→{strat}: {out_file}")


