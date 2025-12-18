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
torch.set_num_threads(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_prompt(instruction, input_text=None):
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
    prompts = [generate_prompt(inst, inp) for inst, inp in zip(instructions, inputs)]
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
    outputs = [text.split('Response:\n')[-1] for text in decoded]
    real_outputs = [outputs[i * num_beams: (i + 1) * num_beams] for i in range(len(outputs) // num_beams)]
    return real_outputs

def batch(lst, batch_size):
    chunk_size = (len(lst) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield lst[batch_size * i: batch_size * (i + 1)]

def generate_selfplay_negatives(config):
    base_model = config["base_model"]
    lora_weights = config.get("lora_weights", None)
    batch_size = config.get("batch_size", 4)
    train_sample_size = config.get("train_sample_size", 1024)
    valid_sample_size = config.get("valid_sample_size", 128)
    
    # 修改：準備檔案路徑變數，對齊論文的儲存邏輯
    output_dir = prepare_output_dir(config["output_path"], None)
    
    print("Loading base model:", base_model)
    global tokenizer, model 
    # 修改：加入 torch_dtype=torch.float16 對齊論文
    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        device_map={"": 0}, 
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    if lora_weights:
        print("Loading LoRA weights from:", lora_weights)
        # 修改：Peft model 也建議指定 dtype
        model = PeftModel.from_pretrained(
            model, 
            lora_weights,
            torch_dtype=torch.float16
        )
    model.eval()
    model.to(device)

    # Load data
    train_data = safe_load_json(config["train_data_path"])
    valid_data = safe_load_json(config["valid_data_path"])

    # 修改：在此處進行 Random Sampling，與論文一致
    # 論文邏輯：先 sample 再合併
    if len(train_data) > train_sample_size:
        train_data = random.sample(train_data, train_sample_size)
    
    if len(valid_data) > valid_sample_size:
        valid_data = random.sample(valid_data, valid_sample_size)

    # 論文這一步會把 sample 後的 SFT 資料存下來 (Optionally implemented)
    # 你可以選擇是否要存這份 sample 過的 raw data
    
    # 合併資料進行推論
    data = train_data + valid_data
    instructions = [d['instruction'] for d in data]
    inputs = [d['input'] for d in data]

    outputs = []
    for batch_instructions, batch_inputs in tqdm(zip(batch(instructions, batch_size), batch(inputs, batch_size)),
                                                   total=(len(instructions) - 1) // batch_size + 1,
                                                   desc="Evaluating batches"):
        batch_outputs = evaluate(batch_instructions, batch_inputs)
        outputs.extend(batch_outputs)
    
    for i, d in enumerate(data):
        d['predict'] = outputs[i][0] if outputs[i] else ""
    
    dpo_data = []
    for d in data:
        dpo_case = {}
        dpo_case['prompt'] = d['instruction'] + (d['input'] if d['input'] else "")
        dpo_case['chosen'] = d['output'].strip() # 保留 strip 是 ok 的
        
        # 修改：Regex 失敗時的處理邏輯，對齊論文 (改為 newline)
        pattern = r'"(.*?)"'
        item_names = re.findall(pattern, d['predict'])
        if item_names:
            dpo_case['rejected'] = f"\"{item_names[0]}\"\n"
        else:
            # 論文設定：如果沒抓到引號內容，設為換行符號
            dpo_case['rejected'] = "\n"
            
        dpo_data.append(dpo_case)
    
    # Split back to train/valid based on the sample sizes
    dpo_train_data = dpo_data[:len(train_data)]
    dpo_valid_data = dpo_data[len(train_data):]
    
    # 修改：使用 JSON Lines 格式寫入，對齊論文
    # (如果你的 safe_write_json 寫的是 standard list，這裡建議直接用 open 寫)
    
    def write_jsonl(data, filename):
        with open(filename, 'w') as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')

    write_jsonl(dpo_train_data, f"{output_dir}/train.json")
    write_jsonl(dpo_valid_data, f"{output_dir}/valid.json")
    
    print(f"Self-play negatives generation complete. Saved to {output_dir}")