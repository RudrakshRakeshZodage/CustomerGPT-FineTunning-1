import os
import json
import random
import pandas as pd
from utils.helpers import setup_logger, load_json, save_jsonl, save_json

# Seed for reproducibility
random.seed(42)

def run_split(final_dir, reports_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Shuffles the dataset and splits it into train, validation, and test splits.
    Saves them in JSONL format for training.
    """
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    logger = setup_logger("dataset_split", os.path.join(reports_dir, "dataset_split.log"))
    logger.info("Starting Dataset Split pipeline...")
    
    qwen_dataset_path = os.path.join(final_dir, "dataset_qwen.json")
    if not os.path.exists(qwen_dataset_path):
        logger.error(f"Qwen dataset not found at {qwen_dataset_path}. Please format first.")
        return
        
    data = load_json(qwen_dataset_path)
    total_samples = len(data)
    logger.info(f"Loaded {total_samples} records for splitting.")
    
    # 1. Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 2. Compute split indices
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    train_split = shuffled_data[:train_end]
    val_split = shuffled_data[train_end:val_end]
    test_split = shuffled_data[val_end:]
    
    logger.info(f"Split sizes - Train: {len(train_split)}, Validation: {len(val_split)}, Test: {len(test_split)}")
    
    # 3. Save as JSONL files
    save_jsonl(train_split, os.path.join(final_dir, "train.jsonl"))
    save_jsonl(val_split, os.path.join(final_dir, "validation.jsonl"))
    save_jsonl(test_split, os.path.join(final_dir, "test.jsonl"))
    
    # Also save as standard JSON lists for viewing/debugging
    with open(os.path.join(final_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_split, f, indent=2, ensure_ascii=False)
    with open(os.path.join(final_dir, "validation.json"), "w", encoding="utf-8") as f:
        json.dump(val_split, f, indent=2, ensure_ascii=False)
    with open(os.path.join(final_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_split, f, indent=2, ensure_ascii=False)
        
    # 4. Generate Split Statistics
    def get_intent_distribution(split_data):
        intents = {}
        for item in split_data:
            intent = item.get('metadata', {}).get('intent', 'unknown')
            intents[intent] = intents.get(intent, 0) + 1
        return intents
        
    stats = {
        "total_records": total_samples,
        "train_count": len(train_split),
        "validation_count": len(val_split),
        "test_count": len(test_split),
        "ratios": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": test_ratio
        },
        "train_intents": get_intent_distribution(train_split),
        "validation_intents": get_intent_distribution(val_split),
        "test_intents": get_intent_distribution(test_split)
    }
    
    save_json(stats, os.path.join(reports_dir, "split_statistics.json"))
    
    # Markdown report
    split_report_md = f"""# Dataset Split Report

## Splitting Configuration
- **Shuffling Seed**: `42`
- **Target Ratios**: Train = {train_ratio*100:.0f}%, Validation = {val_ratio*100:.0f}%, Test = {test_ratio*100:.0f}%
- **Total Records Split**: {total_samples}

## Generated Split Sizes
- **train.jsonl**: {stats["train_count"]} samples ({(stats["train_count"]/total_samples)*100:.2f}%)
- **validation.jsonl**: {stats["validation_count"]} samples ({(stats["validation_count"]/total_samples)*100:.2f}%)
- **test.jsonl**: {stats["test_count"]} samples ({(stats["test_count"]/total_samples)*100:.2f}%)

## Dataset Split Location
All split files are saved in the `data/final/` directory:
- `train.jsonl` / `train.json`
- `validation.jsonl` / `validation.json`
- `test.jsonl` / `test.json`
"""
    
    with open(os.path.join(reports_dir, "split_report.md"), "w", encoding="utf-8") as f:
        f.write(split_report_md)
        
    logger.info("Dataset split reports generated.")
    return stats

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Split Dataset into Train/Val/Test")
    parser.add_argument("--final_dir", type=str, default="data/final", help="Final directory")
    parser.add_argument("--reports_dir", type=str, default="reports/validation", help="Reports directory")
    args = parser.parse_args()
    
    run_split(args.final_dir, args.reports_dir)
