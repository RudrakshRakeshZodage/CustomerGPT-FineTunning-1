import os
import re
import pandas as pd
import numpy as np
from utils.helpers import setup_logger, load_csv, save_csv, save_json

def compute_grammar_score(row):
    """
    Grammar quality score (1-10).
    Penalizes typos, double spaces, and consecutive duplicate punctuation.
    """
    score = 10.0
    instruction = row['instruction']
    response = row['response']
    flags = str(row.get('flags', ''))
    
    # 1. Typo checks via dataset flag 'Z' (stands for typo in Bitext flags)
    if 'Z' in flags:
        score -= 2.0
        
    # 2. Check for double spaces
    if '  ' in instruction or '  ' in response:
        score -= 1.0
        
    # 3. Check for multiple consecutive punctuation (e.g. ,, or !!)
    if re.search(r',,', instruction) or re.search(r',,', response):
        score -= 1.0
    if re.search(r'\?\?', instruction) or re.search(r'\?\?', response):
        score -= 0.5
    if re.search(r'!!', instruction) or re.search(r'!!', response):
        score -= 0.5
        
    return max(1.0, score)

def compute_completeness_score(row):
    """
    Completeness score (1-10).
    Checks if response has standard termination, has key support indicators,
    and is of sufficient length to solve a query.
    """
    score = 10.0
    response = row['response']
    
    # 1. Does it end with sentence termination?
    if not response.endswith(('.', '!', '?', '"', "'")):
        score -= 3.0
        
    # 2. Is response extremely short (indicating incomplete answers)?
    word_count = len(response.split())
    if word_count < 10:
        score -= 4.0
    elif word_count < 20:
        score -= 2.0
        
    # 3. Does it contain structured elements or helpful phrases?
    # (polite wrappers or step-by-step markers represent high-completeness)
    completeness_markers = ["step", "1.", "2.", "please", "assist", "contact", "support", "help", "thank"]
    marker_count = sum(1 for marker in completeness_markers if marker in response.lower())
    if marker_count == 0:
        score -= 1.0
        
    return max(1.0, score)

def compute_length_score(row):
    """
    Length quality score (1-10).
    Measures if instruction and response length fall in the ideal range.
    """
    score = 10.0
    inst_len = len(row['instruction'])
    resp_len = len(row['response'])
    
    # Instruction ideal: 15 to 200 chars
    if inst_len < 10:
        score -= 4.0
    elif inst_len < 20:
        score -= 2.0
    elif inst_len > 300:
        score -= 1.0
        
    # Response ideal: 100 to 1500 chars
    if resp_len < 50:
        score -= 4.0
    elif resp_len < 100:
        score -= 2.0
    elif resp_len > 2000:
        score -= 2.0
    elif resp_len > 1500:
        score -= 1.0
        
    return max(1.0, score)

def compute_relevance_score(row):
    """
    Relevance score (1-10).
    Checks overlap of intent words/category words with response contents,
    and query word overlap with response content.
    """
    score = 10.0
    inst_words = set(re.findall(r'\w+', row['instruction'].lower()))
    resp_words = set(re.findall(r'\w+', row['response'].lower()))
    
    # 1. Query vs Response keyword overlap (excluding short stop words)
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'he', 'him', 'she', 'her', 'it', 'its', 'they', 'them', 'what', 'which', 'who', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', 'now'}
    
    meaningful_inst = inst_words - stop_words
    if len(meaningful_inst) > 0:
        overlap = len(meaningful_inst.intersection(resp_words)) / len(meaningful_inst)
        if overlap < 0.1:
            score -= 4.0
        elif overlap < 0.25:
            score -= 2.0
        elif overlap < 0.5:
            score -= 0.5
            
    # 2. Check if the response matches key terms from the intent label (e.g. cancel_order -> cancel/order)
    intent = str(row['intent']).replace('_', ' ').lower()
    intent_words = set(intent.split())
    if len(intent_words) > 0:
        intent_overlap = any(iw in resp_words for iw in intent_words)
        if not intent_overlap:
            score -= 2.0
            
    return max(1.0, score)

def compute_readability_score(row):
    """
    Readability score (1-10).
    A heuristic proxy for readability based on average sentence length
    and average word length.
    """
    score = 10.0
    response = row['response']
    
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0:
        return 1.0
        
    words = response.split()
    if len(words) == 0:
        return 1.0
        
    avg_sentence_len = len(words) / len(sentences)
    avg_word_len = sum(len(w) for w in words) / len(words)
    
    # 1. Penalize overly long sentences (hard to read)
    if avg_sentence_len > 30:
        score -= 2.0
    elif avg_sentence_len > 20:
        score -= 1.0
        
    # 2. Penalize overly long words (jargon or text issues)
    if avg_word_len > 7:
        score -= 2.0
    elif avg_word_len > 6:
        score -= 1.0
        
    return max(1.0, score)

def score_and_filter(input_path, output_path, reports_dir, threshold=5.0):
    """
    Applies quality scoring to each row and filters out low quality samples.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    logger = setup_logger("quality_scoring", os.path.join(reports_dir, "quality_scoring.log"))
    logger.info(f"Starting quality scoring with threshold = {threshold}...")
    
    df = load_csv(input_path)
    logger.info(f"Loaded dataset of shape: {df.shape}")
    
    # Apply scoring functions
    logger.info("Computing metrics for all records...")
    df['grammar_score'] = df.apply(compute_grammar_score, axis=1)
    df['completeness_score'] = df.apply(compute_completeness_score, axis=1)
    df['length_score'] = df.apply(compute_length_score, axis=1)
    df['relevance_score'] = df.apply(compute_relevance_score, axis=1)
    df['readability_score'] = df.apply(compute_readability_score, axis=1)
    
    # Weighted average score
    df['quality_score'] = (
        0.20 * df['grammar_score'] +
        0.25 * df['completeness_score'] +
        0.15 * df['length_score'] +
        0.20 * df['relevance_score'] +
        0.20 * df['readability_score']
    ).round(2)
    
    # Sort and analyze
    avg_score = df['quality_score'].mean()
    min_score = df['quality_score'].min()
    max_score = df['quality_score'].max()
    logger.info(f"Quality score stats - Mean: {avg_score:.2f}, Min: {min_score}, Max: {max_score}")
    
    # Filter by threshold
    df_filtered = df[df['quality_score'] >= threshold].copy()
    num_removed = len(df) - len(df_filtered)
    logger.info(f"Filtered out {num_removed} records below threshold {threshold}. Remaining: {len(df_filtered)}")
    
    # Save the scored and filtered dataset
    save_csv(df_filtered, output_path)
    
    # Generate statistics
    stats = {
        "total_records_evaluated": int(df.shape[0]),
        "total_records_passed": int(df_filtered.shape[0]),
        "total_records_filtered": int(num_removed),
        "threshold": float(threshold),
        "mean_quality_score": float(avg_score),
        "min_quality_score": float(min_score),
        "max_quality_score": float(max_score),
        "component_averages": {
            "grammar": float(df['grammar_score'].mean()),
            "completeness": float(df['completeness_score'].mean()),
            "length": float(df['length_score'].mean()),
            "relevance": float(df['relevance_score'].mean()),
            "readability": float(df['readability_score'].mean())
        }
    }
    
    save_json(stats, os.path.join(reports_dir, "quality_scoring_report.json"))
    
    # Write quality report markdown
    quality_report_md = f"""# Data Quality Scoring Report

## Configuration
- **Quality Score Threshold**: {threshold} / 10.0
- **Total Records Evaluated**: {stats["total_records_evaluated"]}
- **Records Passed**: {stats["total_records_passed"]}
- **Records Filtered Out**: {stats["total_records_filtered"]} ({(stats["total_records_filtered"] / stats["total_records_evaluated"])*100:.2f}%)

## Overall Score Distribution
- **Average Quality Score**: {stats["mean_quality_score"]:.2f}
- **Minimum Score**: {stats["min_quality_score"]}
- **Maximum Score**: {stats["max_quality_score"]}

## Component Quality Metrics (Averages)
- **Grammar Quality Score**: {stats["component_averages"]["grammar"]:.2f}
- **Response Completeness Score**: {stats["component_averages"]["completeness"]:.2f}
- **Length Quality Score**: {stats["component_averages"]["length"]:.2f}
- **Relevance Score**: {stats["component_averages"]["relevance"]:.2f}
- **Readability Score**: {stats["component_averages"]["readability"]:.2f}

## Quality Metrics Methodology
- **Grammar (20%)**: Evaluates spelling flags, double spacing, and punctuation issues.
- **Completeness (25%)**: Checks sentence termination, response length limits, and presence of standard support keywords.
- **Length (15%)**: Penalizes queries or responses that are outlier short or outlier long.
- **Relevance (20%)**: Computes word-overlap between queries and responses, checking intent word alignment.
- **Readability (20%)**: Approximates text readability using word-length and sentence-length heuristics.
"""
    
    with open(os.path.join(reports_dir, "quality_report.md"), "w", encoding="utf-8") as f:
        f.write(quality_report_md)
        
    logger.info("Quality scoring report saved.")
    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Quality Scoring on dataset")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input cleaned CSV dataset")
    parser.add_argument("--output_path", type=str, default="data/cleaned/cleaned_filtered_dataset.csv", help="Path to output CSV")
    parser.add_argument("--reports_dir", type=str, default="reports/quality", help="Reports directory")
    parser.add_argument("--threshold", type=float, default=5.0, help="Minimum quality score to keep")
    args = parser.parse_args()
    
    score_and_filter(args.input_path, args.output_path, args.reports_dir, args.threshold)
