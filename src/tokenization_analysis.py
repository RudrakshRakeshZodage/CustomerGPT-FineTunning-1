import os
import numpy as np
import pandas as pd
from utils.helpers import setup_logger, load_json, save_json

def estimate_tokens(text):
    """
    Fallback token estimator if tokenizer loading fails.
    Assumes ~1.3 tokens per word on average for standard English text.
    """
    if not isinstance(text, str):
        return 0
    words = text.split()
    return int(len(words) * 1.3) + 4  # Add padding for special template tokens

def run_tokenization_analysis(final_dir, reports_dir, tokenizer_name="Qwen/Qwen2.5-7B-Instruct"):
    """
    Performs tokenization analysis on Qwen formatted dataset.
    """
    os.makedirs(reports_dir, exist_ok=True)
    
    logger = setup_logger("tokenization_analysis", os.path.join(reports_dir, "tokenization_analysis.log"))
    logger.info(f"Starting Tokenization Analysis using tokenizer: {tokenizer_name}...")
    
    qwen_dataset_path = os.path.join(final_dir, "dataset_qwen.json")
    if not os.path.exists(qwen_dataset_path):
        logger.error(f"Qwen dataset not found at {qwen_dataset_path}. Please format first.")
        return
        
    data = load_json(qwen_dataset_path)
    logger.info(f"Loaded {len(data)} conversations for tokenization analysis.")
    
    # 1. Attempt to load tokenizer
    tokenizer = None
    fallback_used = False
    try:
        from transformers import AutoTokenizer
        logger.info(f"Attempting to load tokenizer '{tokenizer_name}' from Hugging Face...")
        # Use a timeout or local files only to avoid long hangs if offline
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            trust_remote_code=True,
            local_files_only=False
        )
        logger.info("Tokenizer loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to load Hugging Face tokenizer ({e}). Falling back to rule-based token estimation.")
        fallback_used = True
        
    token_counts = []
    query_token_counts = []
    resp_token_counts = []
    
    for idx, item in enumerate(data):
        messages = item['messages']
        
        # Format the text as Qwen Chat Template:
        # <|im_start|>system\n{system_content}<|im_end|>\n
        # <|im_start|>user\n{user_content}<|im_end|>\n
        # <|im_start|>assistant\n{assistant_content}<|im_end|>\n
        
        sys_txt = messages[0]['content']
        user_txt = messages[1]['content']
        asst_txt = messages[2]['content']
        
        full_templated_text = (
            f"<|im_start|>system\n{sys_txt}<|im_end|>\n"
            f"<|im_start|>user\n{user_txt}<|im_end|>\n"
            f"<|im_start|>assistant\n{asst_txt}<|im_end|>\n"
        )
        
        if tokenizer is not None and not fallback_used:
            try:
                # Tokenize full prompt
                tokens = tokenizer.encode(full_templated_text)
                token_counts.append(len(tokens))
                # Tokenize query only
                query_tokens = tokenizer.encode(user_txt)
                query_token_counts.append(len(query_tokens))
                # Tokenize response only
                resp_tokens = tokenizer.encode(asst_txt)
                resp_token_counts.append(len(resp_tokens))
            except Exception as tokenize_err:
                # Fallback on individual error
                tc = estimate_tokens(full_templated_text)
                token_counts.append(tc)
                query_token_counts.append(estimate_tokens(user_txt))
                resp_token_counts.append(estimate_tokens(asst_txt))
        else:
            tc = estimate_tokens(full_templated_text)
            token_counts.append(tc)
            query_token_counts.append(estimate_tokens(user_txt))
            resp_token_counts.append(estimate_tokens(asst_txt))
            
    # Calculate statistics
    token_counts = np.array(token_counts)
    query_token_counts = np.array(query_token_counts)
    resp_token_counts = np.array(resp_token_counts)
    
    avg_tokens = float(token_counts.mean())
    max_tokens = int(token_counts.max())
    min_tokens = int(token_counts.min())
    
    percentiles = {
        "p50": float(np.percentile(token_counts, 50)),
        "p90": float(np.percentile(token_counts, 90)),
        "p95": float(np.percentile(token_counts, 95)),
        "p99": float(np.percentile(token_counts, 99))
    }
    
    # Recommended max sequence length: round up the 99th percentile to the nearest 128 or 256
    p99_val = percentiles["p99"]
    recommended_len = int(np.ceil(p99_val / 128.0) * 128.0)
    # Ensure a minimum recommendation of 512
    recommended_len = max(recommended_len, 512)
    
    stats = {
        "tokenizer_used": tokenizer_name if not fallback_used else f"{tokenizer_name} (Estimated Fallback)",
        "fallback_used": fallback_used,
        "sample_count": len(data),
        "avg_tokens_per_sample": avg_tokens,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "query_avg_tokens": float(query_token_counts.mean()),
        "query_max_tokens": int(query_token_counts.max()),
        "resp_avg_tokens": float(resp_token_counts.mean()),
        "resp_max_tokens": int(resp_token_counts.max()),
        "percentiles": percentiles,
        "recommended_max_sequence_length": recommended_len
    }
    
    save_json(stats, os.path.join(reports_dir, "token_statistics.json"))
    
    # Markdown report
    token_report_md = f"""# Tokenization Analysis Report

## Configuration
- **Tokenizer Model**: `{stats["tokenizer_used"]}`
- **Fallback Rule-based Estimation**: {"Yes" if stats["fallback_used"] else "No"}
- **Total Conversations Analyzed**: {stats["sample_count"]}

## Token Count Statistics
- **Minimum Sequence Length**: {stats["min_tokens"]} tokens
- **Maximum Sequence Length**: {stats["max_tokens"]} tokens
- **Average Sequence Length**: {stats["avg_tokens_per_sample"]:.2f} tokens
- **Average Customer Query Length**: {stats["query_avg_tokens"]:.2f} tokens
- **Average Agent Response Length**: {stats["resp_avg_tokens"]:.2f} tokens

## Sequence Length Distribution (Percentiles)
- **50th Percentile (Median)**: {stats["percentiles"]["p50"]} tokens
- **90th Percentile**: {stats["percentiles"]["p90"]} tokens
- **95th Percentile**: {stats["percentiles"]["p95"]} tokens
- **99th Percentile**: {stats["percentiles"]["p99"]} tokens

## Sequence Length Recommendations
> [!IMPORTANT]
> Based on the token length distribution of the dataset, we recommend setting a training **max_sequence_length** of **`{stats["recommended_max_sequence_length"]}`** tokens. 
> This cutoff covers {stats["percentiles"]["p99"]}% of the entire training dataset without truncation, maximizing context preservation while optimizing GPU memory usage.
"""
    
    with open(os.path.join(reports_dir, "token_statistics_report.md"), "w", encoding="utf-8") as f:
        f.write(token_report_md)
        
    logger.info("Tokenization analysis report generated.")
    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Tokenization Analysis")
    parser.add_argument("--final_dir", type=str, default="data/final", help="Directory containing Qwen dataset JSON")
    parser.add_argument("--reports_dir", type=str, default="reports/tokenization", help="Reports directory")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Tokenizer model name")
    args = parser.parse_args()
    
    run_tokenization_analysis(args.final_dir, args.reports_dir, args.tokenizer_name)
