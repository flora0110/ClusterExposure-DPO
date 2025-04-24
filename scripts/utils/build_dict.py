import re
import numpy as np
from tqdm import tqdm
import json
from collections import Counter


def parse_titles(text: str):
    """Extract all substrings between double quotes."""
    raw_titles = re.findall(r'"([^"]+)"', text)
    cleaned_titles = []
    for t in raw_titles:
        # 如果最前面有單引號，就去掉它
        t = re.sub(r'^[^0-9A-Za-z#(]+', '', t)
        # 如果最後面有單引號，就去掉它
        if t.endswith("'"):
            t = t[:-1]
        cleaned_titles.append(t)
    return cleaned_titles

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

def build_exposure_count(train_data: list) -> (Counter, dict):
    """
    Count the occurrence of each book name in the training data and create a ranking based on exposure.
    
    Args:
        train_data (list): List of data records, each should have an "input" field.
        
    Returns:
        tuple: (Counter, dict) where the Counter counts occurrences and the dict maps each book name to its rank (starting from 1)
               based on exposure (higher count gets a lower rank number).
    """
    counter = Counter()
    for d in train_data:
        book_names = re.findall(r'"(.*?)"', d["input"])
        counter.update(book_names)
        book_names = re.findall(r'"(.*?)"', d["output"])
        counter.update(book_names)
    sorted_books = counter.most_common()
    # exposure_rank = {book: rank + 1 for rank, (book, _) in enumerate(sorted_books)}
    return counter

if __name__ == "__main__":
    # 1. 读入 embeddings 和 book→idx 映射
    item_emb = np.load("/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/model/lightgcn/models/lightgcn_model_full_data_with_output.npy")
    with open("/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/model/lightgcn/models/book2idx_full_data_with_output.json", "r") as f:
        book2idx = json.load(f)  # { book_id (str): idx (int) }

    # 2. 反转出 idx→bookId，以便用 idx 索引 embedding
    idx2book = {idx: book for book, idx in book2idx.items()}

    # 3. 构造 book_id → embedding(np.ndarray) 的字典
    book2emb = {
        idx2book[i]: item_emb[i]
        for i in range(item_emb.shape[0])
        if i in idx2book
    }

    # 4. 为了 JSON 序列化，把 ndarray 转成 list
    book2emb_serializable = {book: emb.tolist() for book, emb in book2emb.items()}

    # 5. 写出到文件
    with open("/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/data/BnetDPO/book2emb.json", "w") as f:
        json.dump(book2emb_serializable, f, ensure_ascii=False, indent=2)

    # 6. 读入训练数据，统计曝光次数
    with open("/scratch/user/chuanhsin0110/ClusterExposure-DPO/data/raw/Goodreads/train.json", "r") as f:
        train_data = json.load(f)

    raw_counts = build_exposure_count(train_data)  # Counter({ book_id: count, ... })

    # 7. 归一化曝光：按最大值归一到 [0,1]
    max_cnt = max(raw_counts.values()) if raw_counts else 1
    book2exposure = {book: cnt / max_cnt for book, cnt in raw_counts.items()}

    # 8. 写出曝光字典
    with open("/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/data/BnetDPO/book2exposure.json", "w") as f:
        json.dump(book2exposure, f, ensure_ascii=False, indent=2)