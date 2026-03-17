import os
import json

import sys
import shutil
from datetime import datetime
from contextlib import contextmanager


# ------------------------------------------------
# Config backup
# ------------------------------------------------
def backup_config_file(config_path, output_dir):
    """
    Copy a yaml config file into the experiment directory.
    """
    if config_path is None:
        print("[Warning] config_path not provided. Skip config backup.")
        return

    if not os.path.isfile(config_path):
        print(f"[Warning] Config file not found: {config_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    dst = os.path.join(output_dir, os.path.basename(config_path))
    shutil.copy2(config_path, dst)

    print(f"[Info] Config file backed up to: {dst}")


# ------------------------------------------------
# Save runtime config (json)
# ------------------------------------------------
def save_runtime_config(config, output_dir, filename="runtime_config.json"):
    """
    Save the runtime config dictionary to JSON for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[Info] Runtime config saved to: {path}")


# ------------------------------------------------
# Tee logger
# ------------------------------------------------
class Tee:
    """
    Duplicate stdout/stderr to terminal and file.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ------------------------------------------------
# Terminal output capture
# ------------------------------------------------
@contextmanager
def capture_terminal_output(output_dir, filename="training_terminal.log"):
    """
    Capture terminal output into a log file while keeping console printing.
    """

    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, filename)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(log_path, "a", encoding="utf-8") as log_file:

        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        print("\n" + "=" * 80)
        print("[Experiment Logging]")
        print(f"log file: {log_path}")
        print(f"start time: {datetime.now()}")
        print("=" * 80 + "\n")

        try:
            yield log_path
        finally:

            print("\n" + "=" * 80)
            print(f"end time: {datetime.now()}")
            print("=" * 80 + "\n")

            sys.stdout = original_stdout
            sys.stderr = original_stderr


def safe_write_json(file_path, data, indent=2):
    """
    Safely write data to a JSON file. If the directory does not exist,
    it will be created automatically. If the file already exists,
    a warning will be shown and a numeric suffix will be appended to avoid overwriting.

    Args:
        file_path (str): Target file path for saving the JSON file.
        data: The data to be serialized using json.dump.
        indent (int, optional): Indentation level for JSON formatting. Defaults to 2.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    base_path, ext = os.path.splitext(file_path)
    final_path = file_path
    counter = 1

    while os.path.exists(final_path):
        print(f"Warning: {final_path} already exists. Generating new file name...")
        final_path = f"{base_path}_{counter}{ext}"
        counter += 1

    with open(final_path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    print(f"Saved data to {final_path}")



# def safe_load_json(file_path):
#     """
#     Safely load a JSON file. If the file does not exist or is not valid JSON, raise an error.
#     """
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File not found: {file_path}")
    
#     with open(file_path, "r") as f:
#         try:
#             data = json.load(f)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"Failed to decode JSON from {file_path}: {e}")
    
#     return data


def safe_load_json(file_path):
    """
    Safely load data from either:
    1. standard JSON file
    2. JSONL file (one JSON object per line)

    Returns:
        dict or list
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            # First try normal JSON
            return json.load(f)
        except json.JSONDecodeError:
            # If normal JSON fails, try JSONL
            f.seek(0)
            data = []
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to decode JSONL from {file_path} at line {line_num}: {e}"
                    )
            if len(data) == 0:
                raise ValueError(f"No valid JSON/JSONL content found in {file_path}")
            return data

def prepare_output_dir(output_path: str, check_subdir: str = "final_model", allow_existing: bool = False) -> str:
    """
    Check if a specified subdirectory (default "final_model") exists within the output path.
    If it exists, append a numeric suffix to the output directory to avoid overwriting.
    
    If check_subdir is None and the output_path already exists, then:
      - If the output_path is empty, print a warning and return it as is.
      - Otherwise, generate a new directory name with a numeric suffix.
    
    Additionally, if allow_existing is True:
      - If check_subdir is None and output_path exists, directly return output_path regardless of contents.
      - If check_subdir is provided and it exists under output_path, return output_path.
      - Otherwise, create the missing subdirectory and return output_path.
    
    Args:
        output_path (str): The desired directory path where output will be saved.
        check_subdir (str or None): The subdirectory name to check. If None, the output_path itself is checked.
        allow_existing (bool): If True, use the existing directory (or create missing subdirectory) without appending a suffix.
        
    Returns:
        str: The final output directory path that can be used safely.
    """
    # Create the base output directory if it does not exist.
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"Created base output dir: {output_path}")

    # If allow_existing is True, then use the existing directory
    if allow_existing:
        if check_subdir is None:
            print(f"Using existing output directory: {output_path}")
            return output_path
        else:
            subdir = os.path.join(output_path, check_subdir)
            if not os.path.exists(subdir):
                os.makedirs(subdir, exist_ok=True)
                print(f"Created subdirectory: {subdir}")
            else:
                print(f"Using existing subdirectory: {subdir}")
            return subdir

    # When allow_existing is False, follow the original logic.
    if check_subdir is None:
        if len(os.listdir(output_path)) == 0:
            print(f"Warning: Output directory '{output_path}' exists and is empty. No need to create a new directory.")
            return output_path
        else:
            base_path = output_path
            counter = 1
            final_output_path = base_path
            while os.path.exists(final_output_path) and os.listdir(final_output_path):
                print(f"Warning: '{final_output_path}' already exists and is not empty. Generating a new output directory name...")
                final_output_path = f"{base_path}_{counter}"
                counter += 1
            os.makedirs(final_output_path, exist_ok=True)
            print(f"Using output directory: {final_output_path}")
            return final_output_path
    else:
        if check_subdir:
            final_model_dir = os.path.join(output_path, check_subdir)
        else:
            final_model_dir = output_path

        counter = 1
        final_output_path = output_path
        while os.path.exists(final_model_dir):
            print(f"Warning: '{final_model_dir}' already exists. Generating a new output directory name...")
            final_output_path = f"{output_path}_{counter}"
            if check_subdir:
                final_model_dir = os.path.join(final_output_path, check_subdir)
            else:
                final_model_dir = final_output_path
            counter += 1

        os.makedirs(final_output_path, exist_ok=True)
        print(f"Using output directory: {final_output_path}")
        return final_output_path
