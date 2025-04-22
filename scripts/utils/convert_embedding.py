import torch
import numpy as np
import os

pth = "/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/model/lightgcn/models/lightgcn_model_full_data.pth"
out = "/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/model/lightgcn/models/lightgcn_model_full_data.npy"

state = torch.load(pth, map_location="cpu")

print("STATE_DICT KEYS:")
for k in state.keys():
    print("  ", k)


item_key = "item_emb.weight"

if item_key not in state:
    raise KeyError(f"{item_key} not in state_dict!")
item_emb = state[item_key].cpu().numpy()
print(f"Loaded item embeddings with shape {item_emb.shape}")

os.makedirs(os.path.dirname(out), exist_ok=True)
np.save(out, item_emb)
print(f"Saved item embeddings to {out}")

num_items, emb_dim = item_emb.shape
print(f"Number of item embeddings: {num_items}")
print(f"Embedding dimension: {emb_dim}")