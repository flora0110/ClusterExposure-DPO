import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import re
import json
import random
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from peft import PeftModel
from src.utils.io_utils import safe_load_json, prepare_output_dir, safe_write_json

# Set thread environment for efficiency
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Set number of torch threads
torch.set_num_threads(1)

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_prompt(instruction, input_text=None):
    """
    Generate a prompt from the given instruction and optional input.
    
    Args:
        instruction (str): The instruction describing the task.
        input_text (str, optional): Additional context for the task.
        
    Returns:
        str: A formatted prompt string.
    """
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""

def evaluate(instructions, inputs, temperature=0, top_p=0.9, top_k=40, num_beams=1, max_new_tokens=128, **kwargs):
    """
    Evaluate the model with a batch of prompts and generate responses.
    
    Args:
        instructions (list of str): List of instructions.
        inputs (list of str): List of input texts corresponding to each instruction.
        temperature (float, optional): Generation temperature.
        top_p (float, optional): Nucleus sampling parameter.
        top_k (int, optional): Top-k sampling parameter.
        num_beams (int, optional): Number of beams for beam search.
        max_new_tokens (int, optional): Maximum tokens to generate.
        **kwargs: Additional generation parameters.
        
    Returns:
        list of list: A list (per input) of generated outputs, each as a list containing `num_beams` strings.
    """
    # Prepare prompts
    prompts = [generate_prompt(inst, inp) for inst, inp in zip(instructions, inputs)]
    # Tokenize inputs with padding
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    gen_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            **encoded,
            generation_config=gen_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    sequences = generation_output.sequences
    decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    # Extract the generated response following the "Response:" marker
    outputs = [text.split('Response:\n')[-1] for text in decoded]
    # Group outputs per prompt based on num_beams
    real_outputs = [outputs[i * num_beams: (i + 1) * num_beams] for i in range(len(outputs) // num_beams)]
    return real_outputs

def batch(lst, batch_size):
    """
    Split a list into batches of a specified size.
    
    Args:
        lst (list): The list to split.
        batch_size (int): The size of each batch.
        
    Yields:
        list: Batches of the input list.
    """
    chunk_size = (len(lst) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield lst[batch_size * i: batch_size * (i + 1)]

def generate_selfplay_negatives(config):
    """
    Main function to generate self-play negatives for DPO training.
    
    This function:
    1. Loads the base model and tokenizer.
    2. Optionally loads LoRA finetuned weights.
    3. Loads and samples training and validation data from JSON files.
    4. Writes the sampled SFT data to output files.
    5. Uses batch evaluation to generate model responses.
    6. Creates DPO formatted data with 'prompt', 'chosen', and 'rejected' fields.
    7. Splits and writes the DPO data into train and validation output files.
    
    Args:
        config (dict): Configuration dictionary containing:
            - train_json_file (str): Path to the input training JSON.
            - valid_json_file (str): Path to the input validation JSON.
            - result_json_sft_data_train (str): Output path for SFT train data.
            - result_json_sft_data_valid (str): Output path for SFT validation data.
            - result_json_dpo_data_train (str): Output path for DPO train data.
            - result_json_dpo_data_valid (str): Output path for DPO validation data.
            - base_model (str): Base model name or path.
            - lora_weights (str): Path to LoRA weights (if used).
            - batch_size (int): Batch size for evaluation.
            - train_sample_size (int): Number of samples to use from train JSON.
            - valid_sample_size (int): Number of samples to use from validation JSON.
            - load_8bit (bool): Whether to load model in 8-bit mode (not used in this version).
            - random_neg (bool): Whether to use random negative sampling (not used here).
    
    Returns:
        None. Outputs are saved to files as specified in the config.
    """
    # Set model and tokenizer
    base_model = config["base_model"]
    lora_weights = config.get("lora_weights", None)
    batch_size = config.get("batch_size", 4)
    train_sample_size = config.get("train_sample_size", 1024)
    
    output_dir = prepare_output_dir(config["output_path"], None)
    
    # Load the model and tokenizer
    print("Loading base model:", base_model)
    global tokenizer, model  # Make them global for use in evaluate()
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    if lora_weights:
        print("Loading LoRA weights from:", lora_weights)
        model = PeftModel.from_pretrained(model, lora_weights)
    model.eval()
    model.to(device)

    # Load and sample SFT data
    train_data = safe_load_json(config["train_data_path"])
    valid_data = safe_load_json(config["valid_data_path"])
    
    # Combine data for evaluation
    data = train_data + valid_data
    instructions = [d['instruction'] for d in data]
    inputs = [d['input'] for d in data]

    outputs = []
    # Process in batches
    for batch_instructions, batch_inputs in tqdm(zip(batch(instructions, batch_size), batch(inputs, batch_size)),
                                                   total=(len(instructions) - 1) // batch_size + 1,
                                                   desc="Evaluating batches"):
        batch_outputs = evaluate(batch_instructions, batch_inputs)
        outputs.extend(batch_outputs)
    
    # Attach the generated prediction to each data point
    for i, d in enumerate(data):
        d['predict'] = outputs[i][0] if outputs[i] else ""
    
    # Create DPO formatted data
    dpo_data = []
    for d in data:
        dpo_case = {}
        # Use concatenated instruction and input as prompt (alternatively, could call generate_prompt)
        dpo_case['prompt'] = d['instruction'] + (d['input'] if d['input'] else "")
        dpo_case['chosen'] = d['output'].strip()
        # Use regex to extract a quoted string from the prediction as the "rejected" output
        pattern = r'"(.*?)"'
        item_names = re.findall(pattern, d['predict'])
        if item_names:
            dpo_case['rejected'] = f"\"{item_names[0]}\"\n"
        else:
            dpo_case['rejected'] = d['predict'].split("\n")[0] + "\n"
        dpo_data.append(dpo_case)
    
    # Split DPO data into training and validation sets
    dpo_train_data = dpo_data[:train_sample_size]
    dpo_valid_data = dpo_data[train_sample_size:]
    
    
    # Save DPO data
    safe_write_json(f"{output_dir}/train.json", dpo_train_data)
    safe_write_json(f"{output_dir}/valid.json", dpo_valid_data)
    # with open(f"{output_dir}/train.json", 'w') as f:
    #     for item in dpo_train_data:
    #         json.dump(item, f)
    #         f.write('\n')
    # with open(f"{output_dir}/valid.json", 'w') as f:
    #     for item in dpo_valid_data:
    #         json.dump(item, f)
    #         f.write('\n')
    print("Self-play negatives generation complete.")

