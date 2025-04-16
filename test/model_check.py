from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
save_dir = "/scratch/user/chuanhsin0110/models/SmolLM2-1.7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Model saved to: {save_dir}")
