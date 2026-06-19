import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.helpers import setup_logger, load_csv, save_json

def run_eda(raw_data_path, output_dir, reports_dir):
    """
    Performs Exploratory Data Analysis and generates reports and charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    logger = setup_logger("eda_analysis", os.path.join(reports_dir, "eda_analysis.log"))
    logger.info("Starting Exploratory Data Analysis on raw dataset...")
    
    # 1. Load dataset
    df = load_csv(raw_data_path)
    logger.info(f"Dataset loaded. Shape: {df.shape}")
    
    stats = {}
    stats["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    stats["columns"] = list(df.columns)
    
    # 2. Check null values
    null_counts = df.isnull().sum().to_dict()
    stats["null_counts"] = {k: int(v) for k, v in null_counts.items()}
    logger.info(f"Null values check: {null_counts}")
    
    # 3. Check duplicate rows
    duplicate_count = int(df.duplicated().sum())
    stats["duplicate_rows"] = duplicate_count
    logger.info(f"Duplicate rows: {duplicate_count}")
    
    # 4. Analyze class distribution
    # Category
    category_counts = df['category'].value_counts().to_dict()
    stats["category_distribution"] = {str(k): int(v) for k, v in category_counts.items()}
    
    # Intent
    intent_counts = df['intent'].value_counts().to_dict()
    stats["intent_distribution"] = {str(k): int(v) for k, v in intent_counts.items()}
    
    # 5. Analyze text lengths
    # Fill NA temporarily to compute lengths
    instructions = df['instruction'].fillna("").astype(str)
    responses = df['response'].fillna("").astype(str)
    
    instruction_char_lens = instructions.apply(len)
    instruction_word_lens = instructions.apply(lambda x: len(x.split()))
    response_char_lens = responses.apply(len)
    response_word_lens = responses.apply(lambda x: len(x.split()))
    
    stats["instruction_lengths"] = {
        "char_len_avg": float(instruction_char_lens.mean()),
        "char_len_max": int(instruction_char_lens.max()),
        "char_len_min": int(instruction_char_lens.min()),
        "char_len_median": float(instruction_char_lens.median()),
        "word_len_avg": float(instruction_word_lens.mean()),
        "word_len_max": int(instruction_word_lens.max()),
        "word_len_min": int(instruction_word_lens.min()),
        "word_len_median": float(instruction_word_lens.median())
    }
    
    stats["response_lengths"] = {
        "char_len_avg": float(response_char_lens.mean()),
        "char_len_max": int(response_char_lens.max()),
        "char_len_min": int(response_char_lens.min()),
        "char_len_median": float(response_char_lens.median()),
        "word_len_avg": float(response_word_lens.mean()),
        "word_len_max": int(response_word_lens.max()),
        "word_len_min": int(response_word_lens.min()),
        "word_len_median": float(response_word_lens.median())
    }
    
    # 6. Detect noisy samples
    # Noise indicators:
    # - Instruction or response is null/empty
    # - Instruction length too short (< 10 chars)
    # - Response length too short (< 20 chars)
    # - Outlier lengths (instruction > 500 chars or response > 2500 chars)
    # - Contains placeholder templates like {{...}} that represent mock data, though these might be typical in support template datasets. Let's record them.
    
    null_records = df[df.isnull().any(axis=1)].index.tolist()
    short_instruction = df[instructions.apply(len) < 10].index.tolist()
    short_response = df[responses.apply(len) < 20].index.tolist()
    
    # Check for placeholder counts
    placeholder_instruction = df[instructions.str.contains(r'\{\{.*?\}\}', regex=True, na=False)].index.tolist()
    placeholder_response = df[responses.str.contains(r'\{\{.*?\}\}', regex=True, na=False)].index.tolist()
    
    stats["noise_detection"] = {
        "null_records_count": len(null_records),
        "short_instructions_count": len(short_instruction),
        "short_responses_count": len(short_response),
        "instructions_with_placeholders_count": len(placeholder_instruction),
        "responses_with_placeholders_count": len(placeholder_response),
    }
    
    # 7. Generate EDA Reports
    save_json(stats, os.path.join(reports_dir, "eda_statistics.json"))
    
    # Create markdown quality report
    report_md = f"""# Data Quality & EDA Report

## Dataset Statistics
- **Total Records**: {stats["shape"]["rows"]}
- **Columns**: {", ".join(stats["columns"])}
- **Duplicate Rows**: {stats["duplicate_rows"]}

## Missing Values
{chr(10).join([f"- **{col}**: {count} missing values" for col, count in stats["null_counts"].items()])}

## Length Analysis
### Instructions (Customer Queries)
- **Average Character Length**: {stats["instruction_lengths"]["char_len_avg"]:.2f} (Median: {stats["instruction_lengths"]["char_len_median"]})
- **Max Character Length**: {stats["instruction_lengths"]["char_len_max"]}
- **Min Character Length**: {stats["instruction_lengths"]["char_len_min"]}
- **Average Word Length**: {stats["instruction_lengths"]["word_len_avg"]:.2f} (Median: {stats["instruction_lengths"]["word_len_median"]})

### Responses (Agent Replies)
- **Average Character Length**: {stats["response_lengths"]["char_len_avg"]:.2f} (Median: {stats["response_lengths"]["char_len_median"]})
- **Max Character Length**: {stats["response_lengths"]["char_len_max"]}
- **Min Character Length**: {stats["response_lengths"]["char_len_min"]}
- **Average Word Length**: {stats["response_lengths"]["word_len_avg"]:.2f} (Median: {stats["response_lengths"]["word_len_median"]})

## Class Distributions
- **Unique Categories**: {len(category_counts)}
- **Unique Intents**: {len(intent_counts)}

## Noise Detection
- **Null Records**: {len(null_records)}
- **Short Instructions (< 10 chars)**: {len(short_instruction)}
- **Short Responses (< 20 chars)**: {len(short_response)}
- **Instructions with placeholders `{{...}}`**: {len(placeholder_instruction)}
- **Responses with placeholders `{{...}}`**: {len(placeholder_response)}
"""
    
    with open(os.path.join(reports_dir, "eda_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
        
    logger.info("EDA report generated.")
    
    # 8. Create Visualizations
    logger.info("Generating EDA visualization charts...")
    sns.set_theme(style="whitegrid")
    
    # Set premium colors
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    # Plot 1: Query & Response Length Distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(instruction_word_lens, bins=30, ax=axes[0], color="#3498db", kde=True)
    axes[0].set_title("Customer Query Length (Words)", fontsize=14, fontweight="bold", pad=15)
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Count")
    
    sns.histplot(response_word_lens, bins=30, ax=axes[1], color="#2ecc71", kde=True)
    axes[1].set_title("Agent Response Length (Words)", fontsize=14, fontweight="bold", pad=15)
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "length_distributions.png"), dpi=300)
    plt.close()
    
    # Plot 2: Category Distribution
    plt.figure(figsize=(12, 6))
    cat_df = pd.Series(category_counts).reset_index()
    cat_df.columns = ['Category', 'Count']
    cat_df = cat_df.sort_values(by='Count', ascending=False)
    
    sns.barplot(x='Count', y='Category', data=cat_df, hue='Category', palette="viridis", legend=False)
    plt.title("Distribution of Conversations by Category", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Conversations")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "category_distribution.png"), dpi=300)
    plt.close()
    
    # Plot 3: Intent Distribution (Top 20 if too many)
    plt.figure(figsize=(14, 8))
    intent_df = pd.Series(intent_counts).reset_index()
    intent_df.columns = ['Intent', 'Count']
    intent_df = intent_df.sort_values(by='Count', ascending=False).head(20)
    
    sns.barplot(x='Count', y='Intent', data=intent_df, hue='Intent', palette="rocket", legend=False)
    plt.title("Top 20 Intent Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Conversations")
    plt.ylabel("Intent")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "intent_distribution_top20.png"), dpi=300)
    plt.close()
    
    logger.info("EDA visualizations saved successfully.")
    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run EDA on raw dataset")
    parser.add_argument("--raw_path", type=str, required=True, help="Path to raw CSV dataset")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory for raw data duplicate")
    parser.add_argument("--reports_dir", type=str, default="reports/eda", help="Reports directory")
    args = parser.parse_args()
    
    run_eda(args.raw_path, args.output_dir, args.reports_dir)
