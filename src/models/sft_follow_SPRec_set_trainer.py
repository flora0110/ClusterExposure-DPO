import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset, load_dataset
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
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # collator = DataCollatorForCompletionOnlyLM(
    #     response_template="### Response:",
    #     tokenizer=tokenizer,
    # )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def add_lora(model, lora_params, resume_from_checkpoint="base_model"):
    print("\nApplying LoRA")
    if resume_from_checkpoint!="base_model":
        model = PeftModel.from_pretrained(
            model, 
            resume_from_checkpoint, 
            is_trainable=True
        )
    else:
        config = LoraConfig(
            inference_mode=lora_params.get("inference_mode", False),
            r=lora_params.get("r", 16),
            lora_alpha=lora_params.get("lora_alpha", 32),
            target_modules=lora_params.get("target_modules", ['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],),
            lora_dropout=lora_params.get("lora_dropout", 0.05),
            bias=lora_params.get("bias", "none"),
            task_type=lora_params.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def formatting_prompts_func(examples):
    # --- 修正開始：判斷傳入的是單筆還是批次 ---
    if isinstance(examples["instruction"], list):
        # 如果是列表，代表是 Batch (訓練時)
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
    else:
        # 如果是字串，代表是單筆 (檢查時)
        instructions = [examples["instruction"]]
        inputs = [examples["input"]]
        outputs = [examples["output"]]
    # --- 修正結束 ---

    output_text = []
    # 這裡改成對整理好的 instructions 列表跑迴圈
    for i in range(len(instructions)):
        instruction = instructions[i]
        input_text = inputs[i]
        response = outputs[i]

        if len(input_text) >= 2:
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}

            ### Input:
            {input_text}

            ### Response:
            {response}
            '''
        else:
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            {instruction}

            ### Response:
            {response}
            '''
        output_text.append(text)

    return output_text


def train_model(model, tokenizer, train_data, val_data, training_params, output_dir):
    print("\nStarting training...")
    sft_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=training_params.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=training_params.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_params.get("gradient_accumulation_steps", 16),
        warmup_steps=training_params.get("warmup_steps", 20), # debug
        learning_rate=float(training_params.get("learning_rate", 2e-5)),
        num_train_epochs=training_params.get("num_train_epochs", 5), # 3 to 5
        logging_steps=training_params.get("logging_steps", 1),
        bf16=training_params.get("bf16", True),
        optim="adamw_torch",
        save_strategy = "steps",

        save_total_limit=training_params.get("save_total_limit", 1),
        report_to = None,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=sft_args,
        formatting_func=formatting_prompts_func,
        # data_collator=collator,
    )
    trainer.train()
    
    final_output_dir = os.path.join(output_dir, "final_model")
    trainer.model.save_pretrained(final_output_dir, safe_serialization=False)
    trainer.tokenizer.save_pretrained(final_output_dir)
    print("Training completed, model saved to:", final_output_dir)

def train_sft(config):
    seed = config.get("seed", 0)
    train_sample_size = config.get("train_sample_size", 1024)
    
    base_model_name = config["base_model_name"]
    output_path = config["output_path"]
    train_data_path = config["train_data_path"]
    valid_data_path = config["valid_data_path"]
    
    output_path = prepare_output_dir(output_path)

    print("Torch CUDA version: ", torch.version.cuda)
    print("Loading base model...")
    model, tokenizer = load_model(base_model_name)
    
    print("Loading training dataset...")
    raw_train_list = safe_load_json(train_data_path)
    train_dataset = Dataset.from_list(raw_train_list)

    real_train_size = len(train_dataset)
    select_train_size = min(train_sample_size, real_train_size)

    train_data = train_dataset.shuffle(seed=seed).select(range(select_train_size))
    print(f"Training data processed: {len(train_data)} samples")
    
    print("Loading validation dataset...")
    raw_val_list = safe_load_json(valid_data_path)
    val_dataset = Dataset.from_list(raw_val_list)

    val_target_size = config.get("valid_sample_size", int(train_sample_size / 8))
    real_val_size = len(val_dataset)
    select_val_size = min(val_target_size, real_val_size)

    val_data = val_dataset.shuffle(seed=seed).select(range(select_val_size))
    print(f"Validation data processed: {len(val_data)} samples")
    
    print("Applying LoRA...")
    resume_from_checkpoint = config.get("resume_from_checkpoint", "base_model")
    model = add_lora(model, config.get("lora", {}), resume_from_checkpoint=resume_from_checkpoint)
    
    print("Commencing training...")
    training_params = config.get("training", {})
    train_model(model, tokenizer, train_data, val_data, training_params, output_path)
