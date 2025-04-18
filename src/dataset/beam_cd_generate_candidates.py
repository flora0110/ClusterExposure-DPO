import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import re
import json
import random
from tqdm import tqdm
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.utils.io_utils import safe_load_json, safe_write_json, prepare_output_dir

# Use the global device (could be "cuda" or "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

def format_prompt(instruction, input_text):
    """
    Format a prompt string with an instruction and optional input.

    Args:
        instruction (str): The instruction text.
        input_text (str): Supplementary input text.

    Returns:
        str: The formatted prompt.
    """
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text.strip():
        prompt += f"### Input:\n{input_text}\n"
    prompt += "### Response:"
    return prompt

def get_user_interest_cluster(input_text, name2genre, topk=3):
    """
    Extract the top-k genres from the input text by scanning for book names and
    their associated genres.

    Args:
        input_text (str): The text to be analyzed.
        name2genre (dict): Mapping from book name to its genres.
        topk (int, optional): Number of top genres to extract. Default is 3.

    Returns:
        set: A set of top genres.
    """
    genre_count = {}
    book_names = re.findall(r'"(.*?)"', input_text)
    for name in book_names:
        genres = name2genre.get(name, [])
        for g in genres:
            genre_count[g] = genre_count.get(g, 0) + 1
    top_genres = sorted(genre_count, key=genre_count.get, reverse=True)[:topk]
    return set(top_genres)

def filter_rejected_candidates(candidates, correct_answer, input_text):
    """
    Filter candidate responses by removing the ones equal to the correct answer
    or matching books already in the input history.

    Args:
        candidates (list of str): Candidate responses.
        correct_answer (str): The expected (correct) output.
        input_text (str): The original input text.

    Returns:
        list: Filtered candidate responses.
    """
    history_books = set(re.findall(r'"(.*?)"', input_text))
    filtered = set()
    for cand in candidates:
        cleaned = cand.strip()
        if cleaned and cleaned != correct_answer.strip() and cleaned != "\"\"":
            cand_name = cleaned.strip('"')
            if cand_name not in history_books:
                filtered.add(cleaned)
    return list(filtered)

def generate_candidates_beam_batch(model, tokenizer, prompts, config, trie=None):
    """
    Generate candidate responses using beam search (optionally with diverse beam search
    and constrained decoding using a trie).

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompts (list of str): List of prompts.
        config (dict): A configuration dictionary containing:
            - max_new_tokens: Maximum tokens to generate.
            - num_return_sequences: Number of sequences to return per prompt.
            - diversity_penalty: Diversity penalty (if using diverse beam search).
            - diverse_beam_search (bool): Whether to use diverse beam search.
        trie: (Optional) A Trie object for constrained decoding.

    Returns:
        list of list of str: For each prompt, a list of candidate responses.
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    batch_size = len(prompts)
    prompt_end_text = "### Response:"
    prompt_end_ids = tokenizer.encode(prompt_end_text, add_special_tokens=False)

    def find_response_start(input_ids, prompt_end_ids):
        for i in range(len(input_ids) - len(prompt_end_ids) + 1):
            if input_ids[i:i+len(prompt_end_ids)] == prompt_end_ids:
                return i + len(prompt_end_ids)
        return None

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        input_ids_list = input_ids.tolist()
        response_start = find_response_start(input_ids_list, prompt_end_ids)
        if response_start is None:
            response_start = (input_ids != tokenizer.pad_token_id).sum().item()
        TRIE_START_OFFSET = 2
        if input_ids.shape[-1] <= response_start + TRIE_START_OFFSET:
            return list(range(tokenizer.vocab_size))
        response_only_prefix = input_ids[response_start + TRIE_START_OFFSET:]
        allowed = trie.get_allowed_tokens(response_only_prefix.tolist()) if trie else None
        return allowed if allowed else [tokenizer.eos_token_id]

    with torch.no_grad():
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=config.get("max_new_tokens", 100),
            do_sample=False,
            num_return_sequences=config.get("num_return_sequences", 3),
            pad_token_id=tokenizer.eos_token_id
        )
        if config.get("diverse_beam_search", False):
            gen_kwargs.update({
                "num_beams": config.get("num_return_sequences", 3) * 2,
                "num_beam_groups": config.get("num_return_sequences", 3),
                "diversity_penalty": config.get("diversity_penalty", 1.0)
            })
        else:
            gen_kwargs.update({
                "num_beams": config.get("num_return_sequences", 3)
            })
        if trie:
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        outputs = model.generate(**gen_kwargs)

    all_candidates = []
    for i in range(batch_size):
        prompt = prompts[i]
        candidates = []
        for j in range(config.get("num_return_sequences", 3)):
            output = outputs[i * config.get("num_return_sequences", 3) + j]
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            response = decoded[len(prompt):].strip()
            match = re.search(r'"([^"\n]*)', response)
            if match:
                candidate = f'"{match.group(1)}"\n'
                candidates.append(candidate)
        all_candidates.append(candidates)
    return all_candidates

def build_exposure_count(train_data):
    """
    Count occurrences of book names in train data and return both
    the counter and an exposure rank dictionary.

    Args:
        train_data (list): A list of training data dictionaries (each with an "input" key).

    Returns:
        tuple: (Counter, dict) where the dict maps each book name to its rank.
    """
    counter = Counter()
    for d in train_data:
        book_names = re.findall(r'"(.*?)"', d["input"])
        counter.update(book_names)
    sorted_books = counter.most_common()
    exposure_rank = {book: rank+1 for rank, (book, _) in enumerate(sorted_books)}
    return counter, exposure_rank

class Trie:
    """
    A simple Trie (prefix tree) implementation for constrained decoding.
    """
    def __init__(self, eos_token_id):
        """
        Initialize the Trie.
        
        Args:
            eos_token_id (int): The EOS token ID to allow when a prefix completes a valid token sequence.
        """
        self.children = {}
        self.is_end = False
        self.eos_token_id = eos_token_id

    def insert(self, token_ids):
        """
        Insert a sequence of token IDs into the Trie.
        
        Args:
            token_ids (list): List of token IDs.
        """
        node = self
        for token in token_ids:
            if token not in node.children:
                node.children[token] = Trie(self.eos_token_id)
            node = node.children[token]
        node.is_end = True

    def get_allowed_tokens(self, prefix):
        """
        Get allowed tokens given a prefix token sequence.
        
        Args:
            prefix (list): The current prefix token IDs.
        
        Returns:
            list: List of allowed token IDs.
        """
        node = self
        for token in prefix:
            if token in node.children:
                node = node.children[token]
            else:
                return []
        return list(node.children.keys()) + ([self.eos_token_id] if node.is_end else [])

def process_data(train_path, valid_path, model, tokenizer, config):
    """
    Process training and validation data to generate beam search candidates.
    
    Steps:
      1. Load name-to-genre mapping.
      2. Load and sample train/valid data.
      3. Build an exposure counter.
      4. Build a Trie from valid book names.
      5. Generate candidates for each sample using beam search.
      6. Filter candidates and structure the output.
      7. Save results into JSON files.
    
    Args:
        train_path (str): Path to training sample JSON.
        valid_path (str): Path to validation sample JSON.
        model: The language model.
        tokenizer: The tokenizer.
        config (dict): Configuration dictionary with keys:
            - save_path: Where to save the generated data.
            - batch_size, train_size, valid_size, etc.
    
    Returns:
        None.
    """

    name2genre = safe_load_json(config["name2genre_path"])
    # with open(config["name2genre_path"], "r", encoding="utf-8") as f:
    #     name2genre = json.load(f)


    D = "Div" if config.get("diverse_beam_search", False) else ""
    save_dir = prepare_output_dir(config["save_dir"], f"{D}_{config.get('num_return_sequences', 10)}_{config.get('diversity_penalty', 1.0)}")
    # os.makedirs(config["save_path"], exist_ok=True)

    
    # def load_and_sample(path, size):
    #     with open(path, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #     return data[:size]

    train_data = safe_load_json(train_path)
    valid_data = safe_load_json(valid_path)
    # train_data = load_and_sample(train_path, config.get("train_size", 1024))
    # valid_data = load_and_sample(valid_path, config.get("valid_size", 128))
    exposure_count, exposure_rank = build_exposure_count(train_data)
    full_data = train_data + valid_data

    # Build Trie from valid book names (keys in name2genre)
    valid_books = list(name2genre.keys())
    trie = Trie(tokenizer.eos_token_id)
    for name in valid_books:
        token_ids = tokenizer.encode(name, add_special_tokens=False)
        trie.insert(token_ids)

    correct_in_beam_count = 0
    all_results = []

    for i in tqdm(range(0, len(full_data), config.get("batch_size", 8)), desc="Generating Negatives"):
        batch = full_data[i:i + config.get("batch_size", 8)]
        prompts = [format_prompt(d["instruction"], d["input"]) for d in batch]
        # You can also get user interest clusters if needed (currently not used further)
        beam_candidates = generate_candidates_beam_batch(model, tokenizer, prompts, config, trie=trie)

        for idx, (d, prompt) in enumerate(zip(batch, prompts)):
            if any(cand.strip() == d["output"].strip() for cand in beam_candidates[idx]):
                correct_in_beam_count += 1
            rejected_list = filter_rejected_candidates(beam_candidates[idx], d["output"], d["input"])
            result_dict = {
                "prompt": prompt,
                "chosen": d["output"].strip(),
                "rejected": rejected_list
            }
            all_results.append(result_dict)

    final_results = [all_results]

    train_results = final_results[0][:config.get("train_size", 1024)]
    valid_results = final_results[0][config.get("train_size", 1024):config.get("train_size", 1024)+config.get("valid_size", 128)]

    
    train_filename = f"train.json"
    valid_filename = f"valid.json"

    safe_write_json(os.path.join(save_dir, train_filename), train_results)
    safe_write_json(os.path.join(save_dir, valid_filename), valid_results)
    # with open(os.path.join(config["save_path"], train_filename), "w", encoding="utf-8") as f:
    #     json.dump(train_results, f, indent=2, ensure_ascii=False)
    # with open(os.path.join(config["save_path"], valid_filename), "w", encoding="utf-8") as f:
    #     json.dump(valid_results, f, indent=2, ensure_ascii=False)

    print(f"\nAll datasets saved to {save_dir}")
    print(f"Beam search generated Top-K candidates containing correct answer: {correct_in_beam_count} times")

def beam_cd_generate_candidtate(config):

    # Load tokenizer and model based on config
    base_model = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    if config.get("use_lora", True):
        tuned_path = config.get("finetuned_path", "")
        if tuned_path:
            model = PeftModel.from_pretrained(model, tuned_path)
    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs for inference...")
        model = torch.nn.DataParallel(model)
        inference_model = model.module
    else:
        inference_model = model

    # Call process_data to generate candidates and save result
    process_data(
        train_path=config["train_data_path"],
        valid_path=config["valid_data_path"],
        model=inference_model,
        tokenizer=tokenizer,
        config = config
    )