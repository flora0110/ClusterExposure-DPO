import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from src.utils.io_utils import safe_write_json, safe_load_json, prepare_output_dir

def load_model(base_model_name):
    print("\nLoading model:", base_model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        # tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    collator = DataCollatorForCompletionOnlyLM(
        response_template="### Response:",
        tokenizer=tokenizer,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer, collator

def add_lora(model, lora_params):
    print("\nApplying LoRA")
    config = LoraConfig(
        r=lora_params.get("r", 16),
        lora_alpha=lora_params.get("lora_alpha", 32),
        target_modules=lora_params.get("target_modules", ['q_proj', 'v_proj']),
        lora_dropout=lora_params.get("lora_dropout", 0),
        bias=lora_params.get("bias", "none"),
        task_type=lora_params.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def formatting_prompts_func(example):
    instruction = example["instruction"]
    input_text = example["input"]
    response = example["output"].strip()
    
    formatted = f"### Instruction:\n{instruction}\n"
    if input_text.strip():
        formatted += f"### Input:\n{input_text}\n"
    formatted += f"### Response:\n{response}</s>"
    return formatted

def train_model(model, collator, train_data, val_data, training_params, output_dir):
    print("\nStarting training...")
    sft_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=training_params.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_params.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=training_params.get("gradient_accumulation_steps", 2),
        warmup_steps=training_params.get("gradient_accumulation_steps", 20), # new
        learning_rate=float(training_params.get("learning_rate", 2e-5)),
        num_train_epochs=training_params.get("num_train_epochs", 5), # 3 to 5
        logging_steps=training_params.get("logging_steps", 1),
        bf16=training_params.get("bf16", True),
        # new
        optim="adamw_torch",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=sft_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator
    )
    trainer.train()
    
    final_output_dir = os.path.join(output_dir, "final_model")
    trainer.model.save_pretrained(final_output_dir, safe_serialization=False)
    trainer.tokenizer.save_pretrained(final_output_dir)
    print("Training completed, model saved to:", final_output_dir)

def train_sft(config):
    seed = config.get("seed", 0)
    base_model_name = config["base_model_name"]
    output_path = config["output_path"]
    train_data_path = config["train_data_path"]
    valid_data_path = config["valid_data_path"]
    
    output_path = prepare_output_dir(output_path)

    print("Torch CUDA version: ", torch.version.cuda)
    print("Loading base model...")
    model, tokenizer, collator = load_model(base_model_name)
    
    print("Loading training dataset...")
    train_data = safe_load_json(train_data_path)
    train_data = Dataset.from_list(train_data)
    
    print("Loading validation dataset...")
    val_data = safe_load_json(valid_data_path)
    val_data = Dataset.from_list(val_data)
    
    print("Applying LoRA...")
    model = add_lora(model, config.get("lora", {}))
    
    print("Commencing training...")
    training_params = config.get("training", {})
    train_model(model, collator, train_data, val_data, training_params, output_path)
