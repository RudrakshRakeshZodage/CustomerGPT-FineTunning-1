import os
import json
import logging
import pandas as pd

def setup_directories(base_dir="d:/Rudraksh/College/app/CustomerGPT-FineTunning"):
    """
    Sets up all required directories for the data engineering pipeline.
    """
    directories = [
        os.path.join(base_dir, "data", "raw"),
        os.path.join(base_dir, "data", "cleaned"),
        os.path.join(base_dir, "data", "augmented"),
        os.path.join(base_dir, "data", "validated"),
        os.path.join(base_dir, "data", "tokenized"),
        os.path.join(base_dir, "data", "final"),
        os.path.join(base_dir, "reports"),
        os.path.join(base_dir, "reports", "eda"),
        os.path.join(base_dir, "reports", "cleaning"),
        os.path.join(base_dir, "reports", "quality"),
        os.path.join(base_dir, "reports", "validation"),
        os.path.join(base_dir, "reports", "tokenization"),
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return directories

def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a logger that logs to both file and console.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger is re-initialized
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

def load_csv(file_path):
    """
    Safely load a CSV file.
    """
    return pd.read_csv(file_path)

def save_csv(df, file_path):
    """
    Safely save a DataFrame to CSV.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)

def load_json(file_path):
    """
    Load a JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """
    Save data to a JSON file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_jsonl(data_list, file_path):
    """
    Save a list of dicts to a JSONL file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
