import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from src.utils.io_utils import safe_load_json, safe_write_json, prepare_output_dir
from peft import PeftModel
from datasets import Dataset


def format_prompt(instruction, input_text):
    """
    Format a prompt string given an instruction and optional input text.

    Args:
        instruction (str): The instruction text.
        input_text (str): The supplementary input text.

    Returns:
        str: A formatted prompt.
    """
    if input_text.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:"

def generate_predictions_batch(model, tokenizer, dataset, batch_size=8, max_new_tokens=50):
    """
    Generate predictions in batch for a given dataset using the provided model and tokenizer.

    Args:
        model: The language model used for generation.
        tokenizer: The corresponding tokenizer.
        dataset: A HuggingFace Dataset object containing samples with 'instruction', 'input', and 'output'.
        batch_size (int, optional): The batch size for generation. Default is 8.
        max_new_tokens (int, optional): The maximum new tokens generated per sample. Default is 50.

    Returns:
        list: A list of dictionaries, each containing:
              - prompt: The formatted prompt.
              - prediction: The generated prediction (model output after the prompt).
              - ground_truth: The expected output from the dataset.
    """
    results = []
    model.eval()
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating Predictions"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        prompts = [format_prompt(sample["instruction"], sample["input"]) for sample in batch]

        # Tokenize the batch of prompts
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        for j, output in enumerate(outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt portion to obtain the prediction
            response = decoded[len(prompts[j]):].strip()
            results.append({
                "prompt": prompts[j],
                "prediction": response,
                "ground_truth": batch[j]["output"].strip('" \n')
            })
    return results

def generate_predictions(config):
    base_model = config["base_model"]
    # use_lora = config.get("use_lora", True)
    finetuned_path = config.get("finetuned_path", "")
    test_data_path = config["test_data_path"]
    output_dir = config["output_dir"]
    batch_size = config.get("batch_size", 8)
    max_new_tokens = config.get("max_new_tokens", 50)
    test_sample_size = config.get("test_sample_size", 1000)
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", local_files_only=True)
    if finetuned_path:
        model = PeftModel.from_pretrained(model, finetuned_path)
    # If using multiple GPUs, wrap with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference...")
        model = torch.nn.DataParallel(model)
        inference_model = model.module  # use model.module for inference in DataParallel mode
    else:
        inference_model = model
    
    # Load test dataset and select the required number of samples
    dataset = safe_load_json(test_data_path)
    dataset = Dataset.from_list(dataset)
    
    # Generate predictions
    results = generate_predictions_batch(inference_model, tokenizer, dataset, batch_size=batch_size, max_new_tokens=max_new_tokens)
    
    # Save predictions
    predict_dir = prepare_output_dir(output_dir, "predictions")
    raw_results_filename = f"raw_results_{test_sample_size}.json"
    raw_results_path = os.path.join(predict_dir, raw_results_filename)
    
    safe_write_json(raw_results_path, results)
    
    print(f"\nInference completed. Results saved to: {raw_results_path}")