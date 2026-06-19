import os
import random
import re
import pandas as pd
from utils.helpers import setup_logger, load_csv, save_csv, save_json

# Seed for reproducibility
random.seed(42)

# Common keyboard adjacent keys for typo generation
ADJACENT_KEYS = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfxc', 'e': 'wsdr', 'f': 'rtgvcd',
    'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'ujko', 'j': 'uikmnh', 'k': 'ijlm', 'l': 'okp',
    'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wedxza',
    't': 'rfgy', 'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
}

def generate_typos(text, typo_rate=0.08):
    """
    Introduces keyboard typos (character swap, replacement, double key, omission).
    """
    if len(text) < 5:
        return text
        
    words = text.split()
    augmented_words = []
    
    for word in words:
        # Don't inject typos into placeholders like {{Order Number}}
        if word.startswith('{{') or word.endswith('}}') or '{' in word or '}' in word:
            augmented_words.append(word)
            continue
            
        word_chars = list(word)
        # Introduce typos in ~8% of characters in normal words
        for i in range(len(word_chars)):
            if random.random() < typo_rate:
                typo_type = random.choice(['replace', 'double', 'swap', 'omit'])
                char = word_chars[i].lower()
                
                if typo_type == 'replace' and char in ADJACENT_KEYS:
                    replacement = random.choice(ADJACENT_KEYS[char])
                    word_chars[i] = replacement if word_chars[i].islower() else replacement.upper()
                elif typo_type == 'double':
                    word_chars.insert(i, word_chars[i])
                elif typo_type == 'swap' and i < len(word_chars) - 1:
                    word_chars[i], word_chars[i+1] = word_chars[i+1], word_chars[i]
                elif typo_type == 'omit':
                    word_chars[i] = ''
        
        # Reconstruct word
        augmented_words.append("".join(word_chars))
        
    return " ".join([w for w in augmented_words if w])

def generate_short_form(query, intent):
    """
    Generates a concise, short-form version of the customer query.
    """
    # Try to extract placeholders
    placeholders = re.findall(r'\{\{.*?\}\}', query)
    
    # Intent-based short-forms
    intent_clean = str(intent).replace('_', ' ')
    
    templates = [
        f"how to {intent_clean}",
        f"{intent_clean} help",
        f"need help with {intent_clean}",
        f"can you do {intent_clean}"
    ]
    
    # If placeholders are present, append them
    if placeholders:
        ph_str = " " + " ".join(placeholders)
        templates = [t + ph_str for t in templates]
        
    short = random.choice(templates)
    
    # Capitalize first letter
    if short:
        short = short[0].upper() + short[1:]
    return short

def generate_long_form(query):
    """
    Wraps the query in polite, indirect, or wordy customer language.
    """
    prefixes = [
        "Hi there, sorry to bother you, but I was wondering if you could help me. ",
        "Hello customer support team, I hope you are having a nice day. I have an issue. ",
        "Excuse me, I'm writing this message because I need some assistance with a problem. ",
        "Can you please help me? I have been trying to figure this out on my own but I am stuck. "
    ]
    
    suffixes = [
        " Thank you very much in advance for your assistance!",
        " Please reply as soon as you can. I appreciate your support.",
        " I look forward to hearing from you. Thanks!",
        " It would be great if you could resolve this for me."
    ]
    
    # Remove capital letter of query if prepending
    query_body = query[0].lower() + query[1:] if query else ""
    
    return random.choice(prefixes) + query_body + random.choice(suffixes)

def paraphrase_query_and_response(query, response, intent):
    """
    Paraphrases query and response using high-quality rule-based transformations.
    """
    # 1. Paraphrase query
    q_para = query
    # Replace synonyms
    synonyms = {
        "cancel": "stop",
        "purchase": "order",
        "puchase": "order", # handling typos in raw
        "assistance": "help",
        "assist": "help",
        "problem": "issue",
        "difficulties": "problems",
        "affordable": "cheap",
        "afford": "pay for"
    }
    
    for word, syn in synonyms.items():
        q_para = re.sub(rf'\b{word}\b', syn, q_para, flags=re.IGNORECASE)
        
    if q_para == query:  # If no synonym replaced, add polite words
        q_para = "Please, " + query[0].lower() + query[1:]
        
    # 2. Rewrite response (swap polite starts and endings)
    r_rewritten = response
    
    starts = [
        "I've understood you have a question",
        "I've been informed that you have a question",
        "I can sense that you're seeking assistance",
        "I understood that you need assistance",
        "I'm sensitive to the fact that you're facing",
        "Of course, I'm here to assist you",
        "I pick up what you're putting down",
        "I comprehend the urgency",
        "I catch on to the fact that",
        "I'm on the same wavelength"
    ]
    
    new_starts = [
        "I would be glad to help you",
        "I am ready to assist you",
        "I understand your request completely and will assist you",
        "Let's get this sorted out for you right away",
        "I appreciate you bringing this to our attention. Let me help you"
    ]
    
    # Replace the verbose greeting structures in the Bitext dataset
    for start in starts:
        if r_rewritten.startswith(start):
            r_rewritten = r_rewritten.replace(start, random.choice(new_starts), 1)
            break
            
    return q_para, r_rewritten

def run_augmentation(input_path, output_path, reports_dir, sample_pct=0.15):
    """
    Runs augmentation, generating 2-3 variations for a subset of the dataset
    to keep dataset sizes balanced and clean.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    logger = setup_logger("data_augmentation", os.path.join(reports_dir, "data_augmentation.log"))
    logger.info("Starting Data Augmentation pipeline...")
    
    df = load_csv(input_path)
    logger.info(f"Loaded dataset shape: {df.shape}")
    
    # Sample a portion of the dataset to augment to prevent massive bloating,
    # or augment the entire dataset. Let's make it configurable. 
    # To satisfy "2-3 variations per sample" on the dataset, we can augment a 100% of the dataset
    # but to prevent memory issues let's process it row by row. 
    augmented_rows = []
    
    # Save original rows
    for _, row in df.iterrows():
        augmented_rows.append(row.to_dict())
        
    logger.info(f"Generating 2-3 augmented variations per conversation...")
    
    # We will generate 3 variations for every conversation:
    # Var 1: Typo customer query (realistic typo scenario)
    # Var 2: Short-form / Long-form customer query (realistic format variations)
    # Var 3: Paraphrased customer query + Rewritten response
    
    for idx, row in df.iterrows():
        # Keep track of category, intent, flags, sentiment, urgency, escalation, complexity
        base_dict = row.to_dict()
        
        # Variation 1: Typo query
        var1 = base_dict.copy()
        var1['instruction'] = generate_typos(base_dict['instruction'])
        var1['is_augmented'] = True
        var1['augmentation_type'] = "typos"
        augmented_rows.append(var1)
        
        # Variation 2: Alternating Short and Long form queries
        var2 = base_dict.copy()
        if idx % 2 == 0:
            var2['instruction'] = generate_short_form(base_dict['instruction'], base_dict['intent'])
            var2['augmentation_type'] = "short_form"
        else:
            var2['instruction'] = generate_long_form(base_dict['instruction'])
            var2['augmentation_type'] = "long_form"
        var2['is_augmented'] = True
        augmented_rows.append(var2)
        
        # Variation 3: Paraphrased Query + Rewritten Response
        var3 = base_dict.copy()
        q_p, r_r = paraphrase_query_and_response(base_dict['instruction'], base_dict['response'], base_dict['intent'])
        var3['instruction'] = q_p
        var3['response'] = r_r
        var3['is_augmented'] = True
        var3['augmentation_type'] = "paraphrase_rewrite"
        augmented_rows.append(var3)
        
    df_aug = pd.DataFrame(augmented_rows)
    # Fill in is_augmented for original samples
    df_aug['is_augmented'] = df_aug['is_augmented'].fillna(False)
    df_aug['augmentation_type'] = df_aug['augmentation_type'].fillna("original")
    
    logger.info(f"Augmented dataset shape: {df_aug.shape}")
    save_csv(df_aug, output_path)
    
    # Generate statistics
    aug_counts = df_aug['augmentation_type'].value_counts().to_dict()
    
    stats = {
        "original_records": int(df.shape[0]),
        "total_records_after_augmentation": int(df_aug.shape[0]),
        "added_records": int(df_aug.shape[0] - df.shape[0]),
        "augmentation_distribution": {str(k): int(v) for k, v in aug_counts.items()}
    }
    
    save_json(stats, os.path.join(reports_dir, "augmentation_report.json"))
    
    # Write quality report markdown
    augmentation_report_md = f"""# Data Augmentation Summary Report

## Overview
- **Original Dataset Size**: {stats["original_records"]}
- **Total Dataset Size after Augmentation**: {stats["total_records_after_augmentation"]}
- **Augmented Samples Added**: {stats["added_records"]}

## Augmentation Distribution
- **Original Records**: {aug_counts.get('original', 0)}
- **Typo Variations**: {aug_counts.get('typos', 0)}
- **Short-form Variations**: {aug_counts.get('short_form', 0)}
- **Long-form Variations**: {aug_counts.get('long_form', 0)}
- **Paraphrase/Rewrite Variations**: {aug_counts.get('paraphrase_rewrite', 0)}

## Methodology
1. **Typo Variations**: Simulates real customer keyboard input errors by swapping adjacent keys, omitting keys, or doubling keys (excluding placeholders).
2. **Short-form Variations**: Uses intent keyword mapping to extract a condensed, query-only representation of the prompt.
3. **Long-form Variations**: Prepends polite greetings and appends formal closing comments to simulate wordy/indirect queries.
4. **Paraphrase & Rewrite**: Synonyms are replaced in the query, and verbose polite greetings in the response are rewritten with alternate support scripts to provide sentence variation.
"""
    
    with open(os.path.join(reports_dir, "augmentation_report.md"), "w", encoding="utf-8") as f:
        f.write(augmentation_report_md)
        
    logger.info("Augmentation report saved.")
    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Data Augmentation")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input enriched CSV dataset")
    parser.add_argument("--output_path", type=str, default="data/augmented/augmented_dataset.csv", help="Path to output CSV")
    parser.add_argument("--reports_dir", type=str, default="reports/quality", help="Reports directory")
    args = parser.parse_args()
    
    run_augmentation(args.input_path, args.output_path, args.reports_dir)
