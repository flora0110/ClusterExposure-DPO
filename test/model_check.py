# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# save_dir = "/scratch/user/chuanhsin0110/models/Llama-3.2-1B-Instruct"

# model = AutoModelForCausalLM.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)

# print(f"Model saved to: {save_dir}")


import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
