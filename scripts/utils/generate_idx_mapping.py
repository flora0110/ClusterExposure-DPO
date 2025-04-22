import json
import re

in_files = [
#   "/scratch/user/chuanhsin0110/ClusterExposure-DPO/data/sampled/Goodreads/train.json",
#   "/scratch/user/chuanhsin0110/ClusterExposure-DPO/data/sampled/Goodreads/train.json",
    "/scratch/user/chuanhsin0110/ClusterExposure-DPO/data/raw/Goodreads/train.json",
    "/scratch/user/chuanhsin0110/ClusterExposure-DPO/data/raw/Goodreads/valid.json"
]
out = "/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/model/lightgcn/models/book2idx_full_data_with_output.json"

book_set = set()
for jf in in_files:
    data = json.load(open(jf, "r", encoding="utf8"))
    for entry in data:
        books = re.findall(r'"([^"]+)"', entry["input"])
        books += re.findall(r'"([^"]+)"', entry["output"])
        book_set.update(b.strip() for b in books)
book2idx = {b:i for i,b in enumerate(sorted(book_set))}
with open(out,"w", encoding="utf8") as f:
    json.dump(book2idx, f, ensure_ascii=False, indent=2)
print("Saved book2idx.json with", len(book2idx), "entries")
