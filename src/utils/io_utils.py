import os
import json

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



def safe_load_json(file_path):
    """
    Safely load a JSON file. If the file does not exist or is not valid JSON, raise an error.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from {file_path}: {e}")
    
    return data

def prepare_output_dir(output_path: str, check_subdir: str = "final_model") -> str:
    """
    Check if a specified subdirectory (default "final_model") exists within the output path.
    If it exists, append a numeric suffix to the output directory to avoid overwriting.
    Otherwise, create the output directory.
    
    Args:
        output_path (str): The desired directory path where output will be saved.
        check_subdir (str): The subdirectory name to check for (default is "final_model").
        
    Returns:
        str: The final output directory path that can be used safely.
    """
    # Create the base output directory if it does not exist.
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"Created base output dir: {output_path}")
    
    # Check if the specified subdirectory (e.g., "final_model") exists within the output_path.
    if(check_subdir):
        final_model_dir = os.path.join(output_path, check_subdir)
    else:
        final_model_dir = output_path
    counter = 1
    final_output_path = output_path

    while os.path.exists(final_model_dir):
        print(f"Warning: {final_model_dir} already exists. Generating a new output directory name...")
        # Append a numeric suffix to the base output path.
        final_output_path = f"{output_path}_{counter}"
        # final_model_dir = os.path.join(final_output_path, check_subdir)
        if(check_subdir):
            final_model_dir = os.path.join(final_output_path, check_subdir)
        else:
            final_model_dir = final_output_path
        counter += 1

    # Create the new output directory (with suffix if necessary)
    os.makedirs(final_output_path, exist_ok=True)
    print(f"Using output directory: {final_output_path}")
    return final_output_path