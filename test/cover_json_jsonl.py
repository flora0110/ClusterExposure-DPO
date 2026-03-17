import json
import argparse


def convert_jsonl_to_json(input_path, output_path):
    output_data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            prompt = item["prompt"]
            chosen = item["chosen"].strip()
            rejected = item["rejected"].strip()

            # 嘗試把 instruction 和 input 分開
            split_text = "The user has played the following books before:"
            if split_text in prompt:
                instruction, rest = prompt.split(split_text, 1)

                instruction = instruction.strip()
                input_text = split_text + rest
            else:
                instruction = prompt.strip()
                input_text = ""

            formatted_prompt = (
                "### Instruction:\n"
                f"{instruction}\n"
                "### Input:\n"
                f"{input_text}\n"
                "### Response:"
            )

            output_data.append({
                "prompt": formatted_prompt,
                "chosen": chosen,
                "rejected": rejected
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    convert_jsonl_to_json(args.input, args.output)