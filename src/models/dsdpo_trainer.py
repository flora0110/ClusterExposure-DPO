import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import random
from typing import Optional
import re
import json
import numpy as np


from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from datasets import load_dataset, DatasetDict
from trainer.dynamic_dpo_trainer import DSDPOTrainer
from src.utils.io_utils import safe_write_json, safe_load_json, prepare_output_dir


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

def train_dsdpo(config: dict):
    """
    Train using S-DPO (Softmax Direct Preference Optimization) based on given config.
    Args:
        config (dict): Configuration dictionary.
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
        # print(f"prompt: {examples['prompt']}")
        # print(f"chosen: {examples['chosen']}")
        
        past_titles = [parse_titles(ex) for ex in examples["prompt"]]
        chosen_titles = [parse_titles(ex)[0] for ex in examples["chosen"]]

        # chosen_title = parse_titles(examples["chosen"])[0]
        # print(f"chosen_title: {chosen_title}")

        # compute centroids
        # emb_past   = [avg_emb(past_title, book2idx, item_emb) for past_title in past_titles]
        # emb_chosen = [avg_emb([chosen_title],    book2idx, item_emb) for chosen_title in chosen_titles]
        # # emb_chosen = avg_emb([chosen_title],    book2idx, item_emb)
        # emb_pc = [avg_emb(past_title + [chosen_title], book2idx, item_emb) for past_title, chosen_title in zip(past_titles, chosen_titles)]

        # Flatten rejected list into multiple columns
        dic = {"prompt": examples["prompt"], "chosen": examples["chosen"]}  
        max_neg = config.get("max_neg", 10)
        for i in range(1, max_neg + 1):
            dic[f"rejected{i}"] = [(
                ex[i - 1] if i - 1 < len(ex) else ""
            ) for ex in examples["rejected"]]
            # dic[f"emb_rejected{i}"] = [ item_emb[book2idx.get(n, 0)]  for n in dic[f"rejected{i}"] ]
            # dic[f"emb_rejected{i}"] = [ item_emb[book2idx.get(n, 0)]  for n in dic[f"rejected{i}"] ]

        # for k, v in dic.items():
        #     print(f"{k!r} -> {type(v).__name__}")
        #     if(type(v[0])==str):
        #         print(f"  {len(v)} {type(v[0]).__name__} {type(v[0])} {v[0]}")
        #     print("-" * 20)
        #     print("\n\n")
        return dic

    ds = ds.map(
        process_data,
        batched=True,
        remove_columns=ds["train"].column_names,
        num_proc=config.get("num_proc", 4)
    )

    # Optionally limit validation size
    if ds["validation"].num_rows > config.get("max_valid", 2000):
        ds["validation"] = ds["validation"].select(range(config.get("max_valid", 2000)))

    # Model setup
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_name = config["model_name"]
    # Policy model
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

    # Reference model
    ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    ref = prepare_model_for_kbit_training(ref)
    ref = PeftModel.from_pretrained(ref, config["resume_from_checkpoint"] )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Training arguments
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

    # Trainer
    with open(config["book2idx"], "r") as f:
        book2idx = json.load(f)
    item_emb = np.load(config["item_emb"])  # shape (num_items, emb_dim)
    trainer = DSDPOTrainer(
        model=base,
        ref_model=ref,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        beta=config.get("beta", 0.1),
        tokenizer=tokenizer,
        distance_type=config.get("distance_type", "dp"),
        max_neg=config.get("max_neg", 10),
        book2idx=book2idx,
        item_emb=item_emb,
        beta_range = (config.get("min_beta"), config.get("max_beta")),
    )
    trainer.train()

    # Save final
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Saved S-DPO final model to {final_output_dir}")


# if __name__ == "__main__":
#     import fire
#     fire.Fire(train_sdpo_from_config)