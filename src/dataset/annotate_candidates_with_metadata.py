import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import re
from collections import Counter
from src.utils.io_utils import safe_load_json, safe_write_json, prepare_output_dir

def get_user_interest_cluster(input_text: str, name2genre: dict, topk: int = 3) -> list:
    """
    Extract the top-K genres from the input text based on the frequency of the associated book names.

    Args:
        input_text (str): The text from which to extract book names.
        name2genre (dict): A mapping from book names to a list of genres.
        topk (int, optional): The number of top genres to return. Default is 3.

    Returns:
        list: A list of unique genres (order is not preserved).
    """
    genre_count = {}
    # Extract book names using a regex (names enclosed in double quotes)
    book_names = re.findall(r'"(.*?)"', input_text)
    for name in book_names:
        genres = name2genre.get(name, [])
        for g in genres:
            genre_count[g] = genre_count.get(g, 0) + 1
    top_genres = sorted(genre_count, key=genre_count.get, reverse=True)[:topk]
    return list(set(top_genres))

def build_exposure_count(train_data: list) -> (Counter, dict):
    """
    Count the occurrence of each book name in the training data and create a ranking based on exposure.
    
    Args:
        train_data (list): List of data records, each should have an "input" field.
        
    Returns:
        tuple: (Counter, dict) where the Counter counts occurrences and the dict maps each book name to its rank (starting from 1)
               based on exposure (higher count gets a lower rank number).
    """
    counter = Counter()
    for d in train_data:
        book_names = re.findall(r'"(.*?)"', d["input"])
        counter.update(book_names)
    sorted_books = counter.most_common()
    exposure_rank = {book: rank + 1 for rank, (book, _) in enumerate(sorted_books)}
    return counter, exposure_rank

def process_candidates(input_file: str, name2genre_file: str) -> list:
    """
    Process raw candidate data: parse prompts, extract instruction and input, compute user interest clusters,
    and add rejected candidate metadata including genre information, exposure count, and rank.
    
    Steps:
      1. Load raw candidate JSON and name2genre mapping.
      2. For each record, extract instruction and input from the prompt.
      3. Compute user interest clusters from the input.
      4. Build processed records with fields: instruction, input, chosen, rejected, and interest_clusters.
      5. Compute exposure count and rank based on processed data.
      6. For each record, append "rejected_details" for each candidate in "rejected".
      
    Args:
        input_file (str): Path to the raw candidate JSON file.
        name2genre_file (str): Path to the name2genre mapping JSON file.
        
    Returns:
        list: A list of processed records with additional metadata.
    """
    raw_data = safe_load_json(input_file)
    name2genre = safe_load_json(name2genre_file)
    
    processed_data = []
    for d in raw_data:
        prompt_text = d.get("prompt", "")
        instruction_match = re.search(r"### Instruction:\s*(.*?)\s*### Input:", prompt_text, re.DOTALL)
        input_match = re.search(r"### Input:\s*(.*?)\s*### Response:", prompt_text, re.DOTALL)
        instruction = instruction_match.group(1).strip() if instruction_match else ""
        input_text = input_match.group(1).strip() if input_match else ""
        
        interest_clusters = get_user_interest_cluster(input_text, name2genre, topk=3)
        
        record = {
            "instruction": instruction,
            "input": input_text,
            "chosen": d.get("chosen", ""),
            "rejected": d.get("rejected", []),
            "interest_clusters": interest_clusters
        }
        processed_data.append(record)
    
    exposure_count, exposure_rank = build_exposure_count(processed_data)
    
    for record in processed_data:
        rejected_details = []
        for candidate in record.get("rejected", []):
            name_match = re.search(r'"(.*?)"', candidate)
            candidate_name = name_match.group(1) if name_match else candidate
            # Use the already loaded name2genre mapping instead of reloading it
            genres = name2genre.get(candidate_name, [])
            exposure = exposure_count.get(candidate_name, 0)
            rank = exposure_rank.get(candidate_name, None)
            rejected_details.append({
                "rejected": candidate,
                "genre": genres,
                "exposure_count": exposure,
                "exposure_rank": rank
            })
        record["rejected_details"] = rejected_details
    return processed_data

def annotate_candidates_with_metadata(config: dict) -> None:
    """
    Annotate candidate responses with additional metadata (e.g., interest clusters, exposure details) and save the results.
    
    Args:
        config (dict): Configuration with the following keys:
            - input_train_file: Path to raw candidate JSON for training.
            - input_valid_file: Path to raw candidate JSON for validation.
            - name2genre_file: Path to the name2genre mapping JSON file.
            - save_dir: Base directory to save the annotated data.
            - save_train_filename: Filename for annotated training data.
            - save_valid_filename: Filename for annotated validation data.
    
    Returns:
        None.
    """

    D = "Div" if config.get("diverse_beam_search", False) else ""
    input_dir = prepare_output_dir(config["input_dir"], f"{D}_{config.get('num_return_sequences', 10)}_{config.get('diversity_penalty', 1.0)}", True)
    input_train_file = os.path.join(input_dir, "train.json")
    input_valid_file = os.path.join(input_dir, "valid.json")
    name2genre_file = config["name2genre_file"]

    
    save_dir = prepare_output_dir(config["save_dir"], f"{D}_{config.get('num_return_sequences', 10)}_{config.get('diversity_penalty', 1.0)}", True)
    train_save_path = os.path.join(save_dir, config["save_train_filename"])
    valid_save_path = os.path.join(save_dir, config["save_valid_filename"])
    
    processed_train_data = process_candidates(input_train_file, name2genre_file)
    processed_valid_data = process_candidates(input_valid_file, name2genre_file)
    
    safe_write_json(train_save_path, processed_train_data)
    safe_write_json(valid_save_path, processed_valid_data)
    print("Annotated candidate data saved.")

if __name__ == "__main__":
    # For standalone testing
    input_file = "./eval/Goodreads/sample_candidates.json"
    name2genre_file = "./eval/Goodreads/name2genre.json"
    save_path = "./analysis/annotated_candidates.json"
    
    processed_data = process_candidates(input_file, name2genre_file)
    safe_write_json(save_path, processed_data)
