import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import random
from typing import Optional

from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from datasets import load_dataset, DatasetDict
from accelerator import Accelerator
from trainer.softmax_dpo_trainer import DPOTrainer as SoftmaxDPOTrainer


def train_sdpo_from_config(config: dict):
    """
    Train using S-DPO (Softmax Direct Preference Optimization) based on given config.
    Args:
        config (dict): Configuration dictionary.
    """
    # Prepare output directory
    base_output = config["output_dir"]
    final_output_dir = os.path.join(base_output, "final_model")
    if os.path.exists(final_output_dir):
        print(f"Warning: output dir '{final_output_dir}' exists, may overwrite.")
    else:
        os.makedirs(base_output, exist_ok=True)

    # Data paths
    train_path = config["train_data_path"]
    valid_path = config["valid_data_path"]

    # Load and process dataset
    data_files = {"train": train_path, "validation": valid_path}
    ds = load_dataset("json", data_files=data_files)
    
    def process_data(examples):
        # Flatten rejected list into multiple columns
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
        learning_rate=config.get("learning_rate", 1e-5),
        bf16=config.get("bf16", True),
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_steps=config.get("logging_steps", 1),
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = SoftmaxDPOTrainer(
        model=base,
        ref_model=ref,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        beta=config.get("beta", 0.1),
        tokenizer=tokenizer,
    )
    trainer.train()

    # Save final
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Saved S-DPO final model to {final_output_dir}")


# if __name__ == "__main__":
#     import fire
#     fire.Fire(train_sdpo_from_config)