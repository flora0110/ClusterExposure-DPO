import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, TrainerCallback
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from accelerate import Accelerator
import json
from src.utils.io_utils import safe_load_json, prepare_output_dir

def load_model(base_model_name, resume_from_checkpoint):
    """
    Load the policy and reference models, as well as the tokenizer.
    
    Args:
        base_model_name (str): The base model identifier.
        resume_from_checkpoint (str): Checkpoint path for the LoRA finetuned weights.
    
    Returns:
        tuple: (policy_model, reference_model, tokenizer)
    """
    accelerator = Accelerator()
    # Configure 4-bit quantization settings
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Load Policy Model
    policy_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=bnb_config
    )
    policy_model.config.use_cache = False  # Disable cache for training
    policy_model = prepare_model_for_kbit_training(policy_model)
    if resume_from_checkpoint!="base_model":
        policy_model = PeftModel.from_pretrained(
            policy_model, resume_from_checkpoint, is_trainable=True
        )
    else:
        peft_config = LoraConfig(
        inference_mode=False,
        r=16,
        lora_alpha=32,
        target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    policy_model.print_trainable_parameters()

    # Load Reference Model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=bnb_config
    )
    ref_model = prepare_model_for_kbit_training(ref_model)
    reference_model = PeftModel.from_pretrained(ref_model, resume_from_checkpoint)
    reference_model.print_trainable_parameters()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    return policy_model, reference_model, tokenizer

def train_dpo(config):
    """
    Train a model using Direct Preference Optimization (DPO).

    The function:
      - Loads the policy and reference models along with the tokenizer.
      - Loads the training and evaluation datasets from JSON lines files.
      - Prepares training arguments from the configuration.
      - Initializes and runs the DPOTrainer.
      - Saves the best model to the specified output directory.

    Args:
        config (dict): Configuration dictionary with parameters for model,
                       data paths, and DPO training settings.
    
    Returns:
        None
    """
    base_model_name = config["base_model"]
    resume_from_checkpoint = config["resume_from_checkpoint"]
    output_dir = config["output_dir"]

    output_dir = prepare_output_dir(output_dir)

    # Load models and tokenizer
    policy_model, reference_model, tokenizer = load_model(base_model_name, resume_from_checkpoint)

    # Load DPO training dataset (JSON lines format: one JSON per line)
    train_data = safe_load_json(config["train_data_path"])
    valid_data = safe_load_json(config["valid_data_path"])

    # Convert lists to HuggingFace Dataset objects
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(valid_data)

    # Prepare training arguments from the dpo section of config
    training_args = DPOConfig(
        beta = config["dpo"]["beta"],
        per_device_train_batch_size = config["dpo"]["per_device_train_batch_size"],
        per_device_eval_batch_size = config["dpo"]["per_device_eval_batch_size"],
        gradient_accumulation_steps = config["dpo"]["gradient_accumulation_steps"],
        warmup_steps = config["dpo"]["warmup_steps"],
        num_train_epochs = config["dpo"]["num_train_epochs"],
        learning_rate = float(config["dpo"]["learning_rate"]),
        bf16 = config["dpo"]["bf16"],
        logging_steps = config["dpo"]["logging_steps"],
        optim = config["dpo"]["optim"],
        evaluation_strategy = config["dpo"]["evaluation_strategy"],
        save_strategy = config["dpo"]["save_strategy"],
        output_dir = output_dir,
        save_total_limit = config["dpo"].get("save_total_limit", 1),
        load_best_model_at_end = config["dpo"].get("load_best_model_at_end", True),
        max_prompt_length = config["dpo"].get("max_prompt_length", 512),
        max_length = config["dpo"].get("max_length", 512),
    )

    # Initialize the DPOTrainer with callbacks for early stopping and loss threshold
    trainer = DPOTrainer(
        model = policy_model,
        ref_model = reference_model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        processing_class = tokenizer,
    )

    # Start the training process
    trainer.train()

    # Save the trained model and tokenizer in the output directory
    final_output_dir = os.path.join(output_dir, "final_model")
    trainer.model.save_pretrained(final_output_dir, safe_serialization=False)
    tokenizer.save_pretrained(final_output_dir)
    print("\nTraining completed. Best model saved to:", final_output_dir)
