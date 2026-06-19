import os
import json
import pandas as pd
from utils.helpers import setup_logger, load_csv, save_json, save_jsonl

def to_alpaca(df):
    """
    Converts DataFrame to Alpaca format list of dicts.
    """
    alpaca_data = []
    for _, row in df.iterrows():
        alpaca_data.append({
            "instruction": row['instruction'],
            "input": "",
            "output": row['response']
        })
    return alpaca_data

def to_chatml(df):
    """
    Converts DataFrame to ChatML format list of lists of messages.
    """
    chatml_data = []
    for _, row in df.iterrows():
        chatml_data.append([
            {
                "role": "user",
                "content": row['instruction']
            },
            {
                "role": "assistant",
                "content": row['response']
            }
        ])
    return chatml_data

def to_qwen_format(df):
    """
    Converts DataFrame to Qwen2.5 compatible training format.
    Standard training format contains system prompt and a 'messages' list.
    """
    qwen_data = []
    for _, row in df.iterrows():
        # Include metadata (intent, sentiment, urgency) in system prompt or system context
        # to teach the model to leverage this metadata if desired
        system_content = "You are CustomerGPT, a helpful and professional customer support assistant."
        
        qwen_data.append({
            "messages": [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": row['instruction']
                },
                {
                    "role": "assistant",
                    "content": row['response']
                }
            ],
            "metadata": {
                "intent": row.get('intent', ''),
                "category": row.get('category', ''),
                "sentiment": row.get('sentiment', ''),
                "urgency": row.get('urgency', ''),
                "escalation": bool(row.get('escalation', False)),
                "complexity": row.get('complexity', '')
            }
        })
    return qwen_data

def run_formatting(input_path, output_dir, reports_dir):
    """
    Runs chat formatting and saves in respective files.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    logger = setup_logger("chat_formatting", os.path.join(reports_dir, "chat_formatting.log"))
    logger.info("Starting Chat Formatting pipeline...")
    
    df = load_csv(input_path)
    logger.info(f"Loaded dataset shape: {df.shape}")
    
    # 1. Convert to Alpaca
    logger.info("Converting to Alpaca format...")
    alpaca_data = to_alpaca(df)
    save_json(alpaca_data, os.path.join(output_dir, "dataset_alpaca.json"))
    
    # 2. Convert to ChatML
    logger.info("Converting to ChatML format...")
    chatml_data = to_chatml(df)
    save_json(chatml_data, os.path.join(output_dir, "dataset_chatml.json"))
    
    # 3. Convert to Qwen Format
    logger.info("Converting to Qwen Chat Template format...")
    qwen_data = to_qwen_format(df)
    save_json(qwen_data, os.path.join(output_dir, "dataset_qwen.json"))
    save_jsonl(qwen_data, os.path.join(output_dir, "dataset_qwen.jsonl"))
    
    # Generate statistics
    stats = {
        "total_formatted_records": len(df),
        "alpaca_file": "dataset_alpaca.json",
        "chatml_file": "dataset_chatml.json",
        "qwen_file": "dataset_qwen.json",
        "qwen_jsonl_file": "dataset_qwen.jsonl"
    }
    save_json(stats, os.path.join(reports_dir, "chat_formatting_report.json"))
    
    logger.info("Chat formatting completed successfully.")
    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Chat Formatting")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input augmented CSV dataset")
    parser.add_argument("--output_dir", type=str, default="data/final", help="Output directory")
    parser.add_argument("--reports_dir", type=str, default="reports/quality", help="Reports directory")
    args = parser.parse_args()
    
    run_formatting(args.input_path, args.output_dir, args.reports_dir)
