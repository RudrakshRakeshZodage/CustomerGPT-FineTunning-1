import os
import re
import pandas as pd
from utils.helpers import setup_logger, load_csv, save_csv, save_json

def determine_sentiment(query):
    """
    Heuristically determines the sentiment of a customer query.
    """
    q_lower = query.lower()
    
    # Negative lexicons
    neg_words = [
        "cannot", "can't", "unable to", "don't want", "problem", "issue", "error", 
        "failed", "broken", "frustrated", "angry", "terrible", "bad", "wrong", 
        "mistake", "unauthorized", "stolen", "lost", "complaint", "disappointed",
        "delay", "slow", "chargeback", "double charge", "fraud", "scam"
    ]
    
    # Positive lexicons
    pos_words = [
        "thank", "thanks", "great", "excellent", "happy", "good", "love", "wonderful",
        "appreciate", "helpful", "perfect", "glad", "solved", "fixed"
    ]
    
    neg_score = sum(1 for word in neg_words if word in q_lower)
    pos_score = sum(1 for word in pos_words if word in q_lower)
    
    # Emojis or punctuation indicators
    if "?" in q_lower:  # Questions are typically neutral/seeking help
        neg_score += 0.2
        
    if "!" in q_lower and neg_score > 0:  # Exclamation with negative terms increases negativity
        neg_score += 1.0
        
    if neg_score > pos_score:
        return "negative"
    elif pos_score > neg_score:
        return "positive"
    else:
        return "neutral"

def determine_urgency(query, intent, category):
    """
    Heuristically determines the urgency level (high, medium, low).
    """
    q_lower = query.lower()
    intent_lower = str(intent).lower()
    cat_lower = str(category).lower()
    
    # High urgency terms
    high_urgency_words = [
        "asap", "urgent", "immediately", "right away", "compromised", "hacked", 
        "stolen", "unauthorized", "fraud", "double charge", "cancel", "refund",
        "error", "lost my", "blocked", "suspended", "locked out"
    ]
    
    # High urgency intents/categories
    high_intents = ["cancel_order", "refund", "unauthorized_charge", "reset_password", "account_blocked", "security_issue"]
    
    # Calculate
    is_high_intent = any(hi in intent_lower for hi in high_intents) or "security" in cat_lower
    has_high_words = any(word in q_lower for word in high_urgency_words)
    
    if is_high_intent or (has_high_words and "help" in q_lower):
        return "high"
    
    # Medium urgency: account changes, billing questions, tracking
    medium_intents = ["change_address", "invoice", "track_order", "shipping", "payment_method"]
    is_medium_intent = any(mi in intent_lower for mi in medium_intents) or "billing" in cat_lower or "account" in cat_lower
    
    if is_medium_intent or "problem" in q_lower or "issue" in q_lower:
        return "medium"
        
    return "low"

def determine_escalation(row, urgency, sentiment):
    """
    Heuristically determines whether the conversation should be escalated to a human agent.
    """
    query = row['instruction'].lower()
    
    # Escalation keywords
    escalation_words = [
        "speak to a manager", "human", "representative", "person", "supervisor", 
        "agent", "call me", "phone number", "escalate", "ombudsman", "complain", 
        "threaten", "sue", "legal"
    ]
    
    has_escalation_word = any(word in query for word in escalation_words)
    
    # Escalation condition: 
    # 1. Contains escalation words
    # 2. High urgency AND negative sentiment
    # 3. Mention of fraud or illegal activity
    if has_escalation_word:
        return True
    if urgency == "high" and sentiment == "negative":
        return True
    if "fraud" in query or "scam" in query or "unauthorized" in query:
        return True
        
    return False

def determine_complexity(query, flags):
    """
    Heuristically determines query complexity (high, medium, low).
    """
    word_count = len(query.split())
    flags_str = str(flags)
    
    # High complexity criteria:
    # - Long query (> 20 words)
    # - Multiple flags indicating various constraints (e.g. BCELN)
    # - Contains conditional logic (if, when, unless, but, although)
    has_conditionals = any(cond in query.lower() for cond in ["if", "when", "unless", "but", "although", "except"])
    
    if (word_count > 20 and len(flags_str) >= 4) or (word_count > 15 and has_conditionals):
        return "high"
    elif word_count < 8 and len(flags_str) <= 2:
        return "low"
    else:
        return "medium"

def run_enrichment(input_path, output_path, reports_dir):
    """
    Enriches the dataset with metadata columns.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    logger = setup_logger("data_enrichment", os.path.join(reports_dir, "data_enrichment.log"))
    logger.info("Starting Data Enrichment...")
    
    df = load_csv(input_path)
    logger.info(f"Loaded dataset of shape: {df.shape}")
    
    logger.info("Enriching sentiment, urgency, escalation, and complexity metadata...")
    df['sentiment'] = df['instruction'].apply(determine_sentiment)
    df['urgency'] = df.apply(lambda r: determine_urgency(r['instruction'], r['intent'], r['category']), axis=1)
    df['escalation'] = df.apply(lambda r: determine_escalation(r, r['urgency'], r['sentiment']), axis=1)
    df['complexity'] = df.apply(lambda r: determine_complexity(r['instruction'], r.get('flags', '')), axis=1)
    
    # Save enriched dataset
    save_csv(df, output_path)
    
    # Generate statistics
    sentiment_dist = df['sentiment'].value_counts().to_dict()
    urgency_dist = df['urgency'].value_counts().to_dict()
    escalation_dist = df['escalation'].value_counts().to_dict()
    complexity_dist = df['complexity'].value_counts().to_dict()
    
    stats = {
        "total_enriched_records": int(df.shape[0]),
        "sentiment_distribution": {str(k): int(v) for k, v in sentiment_dist.items()},
        "urgency_distribution": {str(k): int(v) for k, v in urgency_dist.items()},
        "escalation_distribution": {str(k): int(v) for k, v in escalation_dist.items()},
        "complexity_distribution": {str(k): int(v) for k, v in complexity_dist.items()}
    }
    
    save_json(stats, os.path.join(reports_dir, "enrichment_report.json"))
    
    # Write quality report markdown
    enrichment_report_md = f"""# Data Enrichment Summary Report

## Overview
- **Total Records Enriched**: {stats["total_enriched_records"]}

## Metadata Distributions

### Sentiment Analysis
- **Negative**: {sentiment_dist.get('negative', 0)} ({sentiment_dist.get('negative', 0)/stats['total_enriched_records']*100:.2f}%)
- **Neutral**: {sentiment_dist.get('neutral', 0)} ({sentiment_dist.get('neutral', 0)/stats['total_enriched_records']*100:.2f}%)
- **Positive**: {sentiment_dist.get('positive', 0)} ({sentiment_dist.get('positive', 0)/stats['total_enriched_records']*100:.2f}%)

### Urgency Level
- **Low**: {urgency_dist.get('low', 0)} ({urgency_dist.get('low', 0)/stats['total_enriched_records']*100:.2f}%)
- **Medium**: {urgency_dist.get('medium', 0)} ({urgency_dist.get('medium', 0)/stats['total_enriched_records']*100:.2f}%)
- **High**: {urgency_dist.get('high', 0)} ({urgency_dist.get('high', 0)/stats['total_enriched_records']*100:.2f}%)

### Escalation Requirement
- **True (Needs human)**: {escalation_dist.get(True, 0)} ({escalation_dist.get(True, 0)/stats['total_enriched_records']*100:.2f}%)
- **False (Autoresolve)**: {escalation_dist.get(False, 0)} ({escalation_dist.get(False, 0)/stats['total_enriched_records']*100:.2f}%)

### Query Complexity
- **Low**: {complexity_dist.get('low', 0)} ({complexity_dist.get('low', 0)/stats['total_enriched_records']*100:.2f}%)
- **Medium**: {complexity_dist.get('medium', 0)} ({complexity_dist.get('medium', 0)/stats['total_enriched_records']*100:.2f}%)
- **High**: {complexity_dist.get('high', 0)} ({complexity_dist.get('high', 0)/stats['total_enriched_records']*100:.2f}%)
"""
    
    with open(os.path.join(reports_dir, "enrichment_report.md"), "w", encoding="utf-8") as f:
        f.write(enrichment_report_md)
        
    logger.info("Enrichment report saved.")
    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Data Enrichment on dataset")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input cleaned CSV dataset")
    parser.add_argument("--output_path", type=str, default="data/cleaned/cleaned_enriched_dataset.csv", help="Path to output CSV")
    parser.add_argument("--reports_dir", type=str, default="reports/quality", help="Reports directory")
    args = parser.parse_args()
    
    run_enrichment(args.input_path, args.output_path, args.reports_dir)
