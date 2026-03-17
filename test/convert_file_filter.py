import json
import argparse


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Warning] Skip invalid JSONL line {line_num}: {e}")
    return data


def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_title(text):
    """
    Normalize chosen/rejected fields:
    - remove leading/trailing whitespace/newlines
    - ensure the result is wrapped in double quotes
    """
    if text is None:
        return ""

    text = str(text).strip()

    # remove outer quotes first if they are mismatched / repeated handling is needed
    text = text.strip().strip('"').strip("'").strip()

    return f"\"{text}\"" if text else ""


def convert_prompt_b_to_a(raw_prompt):
    """
    Convert B-format prompt into A-format prompt.

    Expected target format:
    ### Instruction:
    ...
    ### Input:
    ...
    ### Response:

    Handles the duplicated pattern seen in B.
    """
    if raw_prompt is None:
        return ""

    raw_prompt = str(raw_prompt).strip()

    input_marker = "The user has played the following books before:"
    end_marker = ", please write a new books that the user may bought"

    start_idx = raw_prompt.find(input_marker)
    if start_idx == -1:
        # fallback: if parsing fails, still put everything into Instruction
        return (
            "### Instruction:\n"
            f"{raw_prompt}\n"
            "### Response:"
        )

    instruction = raw_prompt[:start_idx].strip()

    # Find the first complete input segment:
    end_idx = raw_prompt.find(end_marker, start_idx)
    if end_idx == -1:
        input_text = raw_prompt[start_idx:].strip()
    else:
        end_idx += len(end_marker)
        input_text = raw_prompt[start_idx:end_idx].strip()

    return (
        "### Instruction:\n"
        f"{instruction}\n"
        "### Input:\n"
        f"{input_text}\n"
        "### Response:"
    )


def convert_b_to_c(b_data):
    """
    Convert list of B-format records into A-format records.
    """
    converted = []
    for item in b_data:
        new_item = {
            "prompt": convert_prompt_b_to_a(item.get("prompt", "")),
            "chosen": normalize_title(item.get("chosen", "")),
            "rejected": normalize_title(item.get("rejected", ""))
        }
        converted.append(new_item)
    return converted


def build_match_key(item):
    """
    Match by exact prompt + exact normalized chosen.
    """
    prompt = item.get("prompt", "")
    chosen = normalize_title(item.get("chosen", ""))
    return (prompt, chosen)


def filter_c_by_a(a_data, c_data):
    """
    Keep only records in C whose (prompt, chosen) also appear in A.
    """
    a_keys = {build_match_key(item) for item in a_data}
    filtered = [item for item in c_data if build_match_key(item) in a_keys]
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL file B to A-style JSON file C, then filter matching prompt+chosen records into JSON file D."
    )
    parser.add_argument("--json_a_path", type=str, required=True, help="Path to json file A")
    parser.add_argument("--jsonl_b_path", type=str, required=True, help="Path to jsonl file B")
    parser.add_argument("--json_c_path", type=str, required=True, help="Output path for converted json file C")
    parser.add_argument("--json_d_path", type=str, required=True, help="Output path for matched json file D")
    parser.add_argument("--json_e_path", type=str, required=True, help="Output path for filtered json file E")

    args = parser.parse_args()

    # Load data
    a_data = load_json(args.json_a_path)
    b_data = load_jsonl(args.jsonl_b_path)

    print(f"Loaded A: {len(a_data)} records")
    print(f"Loaded B: {len(b_data)} records")

    # Convert B -> C
    c_data = convert_b_to_c(b_data)
    # save_json(c_data, args.json_c_path)
    save_json(c_data[:1024], args.json_e_path)
    print(f"Saved C: {len(c_data)} records -> {args.json_c_path}")

    # # Filter C by matching prompt + chosen with A -> D
    # d_data = filter_c_by_a(a_data, c_data)
    # save_json(d_data, args.json_d_path)
    # print(f"Saved D: {len(d_data)} records -> {args.json_d_path}")




if __name__ == "__main__":
    main()