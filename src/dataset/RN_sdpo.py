import json
import random
import re
import os

# ==== 配置路徑 ====
name2id_path = "/scratch/user/chuanhsin0110/ClusterExposure-DPO/data/eval/Goodreads/name2id.json"
input_paths = {
    "train": "/scratch/user/chuanhsin0110/ClusterExposure-DPO/data/sampled/Goodreads/train.json",
    "valid": "/scratch/user/chuanhsin0110/ClusterExposure-DPO/data/sampled/Goodreads/valid.json",
}
output_dir = "/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/data/RN"
os.makedirs(output_dir, exist_ok=True)
output_paths = {
    "train": os.path.join(output_dir, "train.json"),
    "valid": os.path.join(output_dir, "valid.json"),
}

# ==== 讀取全部書名 ====
with open(name2id_path, "r") as f:
    name2id = json.load(f)
all_titles = set(name2id.keys())

# ==== 處理 train/valid 兩個檔案 ====
for split in ["train", "valid"]:
    # 1. 讀取原始樣本
    with open(input_paths[split], "r") as f:
        records = json.load(f)

    rn_records = []
    for rec in records:
        instr = rec["instruction"]
        inp = rec["input"]
        # 提取 input 中已經出現的書名（假設都是雙引號包住）
        seen = set(re.findall(r'"([^"]+)"', inp))
        # 從 all_titles 中排除這些 seen，再隨機抽 5 個
        candidates = list(all_titles - seen)
        negs = random.sample(candidates, 5)

        # 構造 RN 格式
        rn_records.append({
            "prompt": f"### Instruction:\n{instr}\n### Input:\n{inp}\n### Response:",
            "chosen": rec["output"].strip(),               # e.g. "\"City of Stairs...\""
            "rejected": [f"\"{t}\"" for t in negs]         # e.g. ["\"Harry Potter...\"", ...]
        })

    # 2. 寫出到硬編碼路徑
    with open(output_paths[split], "w") as f:
        json.dump(rn_records, f, ensure_ascii=False, indent=2)

print("Done: 已生成 RN 負樣本於", output_dir)
