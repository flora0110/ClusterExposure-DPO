import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import random
import math
from typing import Optional, Dict, Any
from collections import Counter

from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from datasets import load_dataset, DatasetDict
from trainer.softmax_dpo_trainer import DPOTrainer as SoftmaxDPOTrainer
from src.utils.io_utils import safe_write_json, safe_load_json, prepare_output_dir

# —— Callback for updating beta ——
class BetaUpdateCallback(TrainerCallback):
    """
    在 evaluate 完成后，读取 metrics 并更新 trainer.beta
    """
    def __init__(self, update_fn):
        super().__init__()
        self.update_fn = update_fn

    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float] = None,
        **kwargs
    ):
        if metrics is None:
            return control

        old_beta = getattr(state.trainer, "beta", None)
        if old_beta is not None:
            new_beta = self.update_fn(old_beta, metrics)
            state.trainer.beta = float(new_beta)
            print(f"[BetaCallback] β: {old_beta:.4f} → {new_beta:.4f}")
        return control

# —— Beta scheduler 示例函数 ——
def beta_scheduler(old_beta: float, metrics: Dict[str, float]) -> float:
    # 举例：beta 随 HR@5 走，HR 越高，beta 越小
    hr = metrics.get("eval_HR@5") or metrics.get("HR@5")
    if hr is None:
        return old_beta
    return old_beta * (1.0 - hr)


def train_bnetsdpo(config: dict):
    # —— 全局预加载资源 ——  
    eval_dir = config.get("eval_dir", "path/to/eval_dir")      # TODO: 改为你的评估目录
    category = config.get("category", "Goodreads")             # TODO: 改为你的类别名称
    id2name = safe_load_json(os.path.join(eval_dir, category, "id2name.json"))
    name2genre = safe_load_json(os.path.join(eval_dir, category, "name2genre.json"))
    genre_dict_template = safe_load_json(os.path.join(eval_dir, category, "genre_dict.json"))
    genres = list(genre_dict_template.keys())

    # compute_metrics 函数
    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        scores = torch.tensor(logits)
        gt = torch.tensor(labels)
        batch_size = scores.size(0)
        k = 5

        # Top-K 和命中率
        topk_scores, topk_idx = torch.topk(scores, k, dim=1)
        hits = topk_idx.eq(gt.unsqueeze(1))
        hr = hits.any(dim=1).float().mean().item()

        # NDCG@5
        ndcg_list = []
        for i in range(batch_size):
            hit_pos = torch.nonzero(hits[i], as_tuple=False)
            if hit_pos.numel() > 0:
                rank = hit_pos[0].item()
                ndcg_list.append(1.0 / math.log2(rank + 2))
            else:
                ndcg_list.append(0.0)
        ndcg = sum(ndcg_list) / batch_size

        # DivRatio@5
        all_recs = topk_idx.flatten().tolist()
        unique_items = set(all_recs)
        diversity = len(unique_items)
        div_ratio = diversity / (batch_size * k)

        # ORRatio@5
        freq = Counter(all_recs)
        top3 = sum(cnt for _, cnt in freq.most_common(3))
        or_ratio = top3 / (3 * batch_size)

        # DGU & MGU
        gp_counts = {g: 0.0 for g in genres}
        gh_counts = {g: 0.0 for g in genres}
        for i in range(batch_size):
            # predicted
            for idx in topk_idx[i].tolist():
                name = id2name[str(idx)]
                item_genres = name2genre.get(name, [])
                if item_genres:
                    for g in item_genres:
                        if g in gp_counts:
                            gp_counts[g] += 1.0 / len(item_genres)
            # ground truth
            gt_idx = gt[i].item()
            name = id2name[str(gt_idx)]
            item_genres = name2genre.get(name, [])
            if item_genres:
                for g in item_genres:
                    if g in gh_counts:
                        gh_counts[g] += 1.0 / len(item_genres)

        total_gp = sum(gp_counts.values()) or 1.0
        total_gh = sum(gh_counts.values()) or 1.0
        gp_norm = [gp_counts[g] / total_gp for g in genres]
        gh_norm = [gh_counts[g] / total_gh for g in genres]
        dis = [gp_norm[i] - gh_norm[i] for i in range(len(genres))]
        dgu = max(dis) - min(dis)
        mgu = sum(abs(x) for x in dis) / len(dis)

        return {
            "MGU@5": mgu,
            "DGU@5": dgu,
            "DivRatio@5": div_ratio,
            "ORRatio@5": or_ratio,
            "NDCG@5": ndcg,
            "HR@5": hr
        }

    """
    Train using S-DPO (Softmax Direct Preference Optimization) based on given config.
    """
    # Prepare output directory
    base_output = config["output_dir"]
    final_output_dir = prepare_output_dir(base_output, "final_model")

    # Data paths
    train_path = config["train_data_path"]
    valid_path = config["valid_data_path"]

    # Load and process dataset
    data_files = {"train": train_path, "validation": valid_path}
    ds = load_dataset("json", data_files=data_files)
    def process_data(examples):
        dic = {"prompt": examples["prompt"], "chosen": examples["chosen"]}
        max_neg = config.get("max_neg", 5)
        for i in range(1, max_neg + 1):
            dic[f"rejected{i}"] = [(
                ex[i - 1] if i - 1 < len(ex) else ""
            ) for ex in examples["rejected"]]
        return dic

    ds = ds.map(
        process_data,
        batched=True,
        remove_columns=ds["train"].column_names,
        num_proc=config.get("num_proc", 4)
    )
    if ds["validation"].num_rows > config.get("max_valid", 2000):
        ds["validation"] = ds["validation"].select(range(config.get("max_valid", 2000)))

    # Model setup
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_name = config["model_name"]
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base)
    base = PeftModel.from_pretrained(
        base,
        config["resume_from_checkpoint"],
        is_trainable=True
    )

    ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    ref = prepare_model_for_kbit_training(ref)
    ref = PeftModel.from_pretrained(ref, config["resume_from_checkpoint"])  

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    training_args = TrainingArguments(
        output_dir=base_output,
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        num_train_epochs=config.get("num_train_epochs", 1),
        learning_rate=float(config.get("learning_rate", 1e-5)),
        bf16=config.get("bf16", True),
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_steps=config.get("logging_steps", 1),
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    # Instantiate trainer with compute_metrics and BetaUpdateCallback
    callback = BetaUpdateCallback(beta_scheduler)
    trainer = SoftmaxDPOTrainer(
        model=base,
        ref_model=ref,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        beta=config.get("beta", 0.1),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[callback]
    )
    trainer.train()

    # Save final model
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Saved S-DPO final model to {final_output_dir}")
