import os
import re
import html
import unicodedata
import pandas as pd
from utils.helpers import setup_logger, load_csv, save_csv, save_json

def clean_text(text):
    """
    Applies standard cleaning, encoding, whitespace, and punctuation fixes to text.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # 1. HTML unescape (e.g. &amp; to &, &lt; to <, etc.)
    text = html.unescape(text)
    
    # 2. Normalize unicode (compatibility form KC)
    text = unicodedata.normalize('NFKC', text)
    
    # 3. Fix smart quotes and apostrophes to standard ascii quotes
    smart_quotes = {
        '‘': "'", '’': "'", '‚': "'", '‛': "'",
        '“': '"', '”': '"', '„': '"', '‟': '"',
        '‹': '<', '›': '>', '–': '-', '—': '-'
    }
    for sq, ascii_q in smart_quotes.items():
        text = text.replace(sq, ascii_q)
        
    # 4. Standardize punctuation spacing and weird symbols
    # Standardize multiple question marks or exclamation marks (e.g., ??? -> ?)
    text = re.sub(r'\?+', '?', text)
    text = re.sub(r'!+', '!', text)
    # Remove excessive commas or dots (keep ellipses)
    text = re.sub(r',+', ',', text)
    text = re.sub(r'\.{4,}', '...', text)
    
    # 5. Normalize whitespace
    # For multiline responses, strip each line and keep newlines
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Replace multiple spaces/tabs with a single space
        line = re.sub(r'[ \t]+', ' ', line)
        line = line.strip()
        cleaned_lines.append(line)
    
    # Filter empty lines, but preserve double-newline spacing between non-empty paragraphs
    cleaned_text = '\n'.join([line for line in cleaned_lines])
    # Replace 3 or more consecutive newlines with exactly 2 newlines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def clean_instruction(text):
    """
    Instruction specific cleaning: normalize capitalization (capitalize first letter, keep the rest).
    """
    cleaned = clean_text(text)
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned

def clean_response(text):
    """
    Response specific cleaning: check capitalization of sentences.
    """
    cleaned = clean_text(text)
    # Standardize sentence casing: Capitalize the first letter of each sentence
    # This regex matches sentence endings followed by spaces and a lowercase letter
    def capitalize_match(match):
        return match.group(1) + match.group(2).upper()
        
    # Capitalize start of response
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
        
    # Capitalize after sentence boundary (.!? followed by whitespace)
    cleaned = re.sub(r'([.!?]\s+)([a-z])', capitalize_match, cleaned)
    return cleaned

def run_cleaning(input_path, output_path, reports_dir):
    """
    Performs data cleaning pipeline and generates reports.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    logger = setup_logger("data_cleaning", os.path.join(reports_dir, "data_cleaning.log"))
    logger.info("Starting Data Cleaning pipeline...")
    
    # Load raw dataset
    df = load_csv(input_path)
    initial_shape = df.shape
    logger.info(f"Loaded dataset of shape: {initial_shape}")
    
    report = {
        "initial_records": int(initial_shape[0]),
        "removed_records": {}
    }
    
    # 1. Remove rows with null instruction or response
    df_clean = df.dropna(subset=['instruction', 'response'])
    null_removed = initial_shape[0] - df_clean.shape[0]
    report["removed_records"]["null_values"] = int(null_removed)
    logger.info(f"Removed {null_removed} records due to null values.")
    
    # 2. Clean instruction and response text
    df_clean = df_clean.copy()
    df_clean['instruction'] = df_clean['instruction'].apply(clean_instruction)
    df_clean['response'] = df_clean['response'].apply(clean_response)
    
    # 3. Remove rows with empty instruction or response after cleaning
    before_empty = df_clean.shape[0]
    df_clean = df_clean[df_clean['instruction'].str.len() > 0]
    df_clean = df_clean[df_clean['response'].str.len() > 0]
    empty_removed = before_empty - df_clean.shape[0]
    report["removed_records"]["empty_strings"] = int(empty_removed)
    logger.info(f"Removed {empty_removed} records with empty instructions or responses.")
    
    # 4. Remove duplicate conversations (based on instruction + response pair)
    before_dup = df_clean.shape[0]
    # We drop duplicates on instruction and response combined
    df_clean = df_clean.drop_duplicates(subset=['instruction', 'response'])
    dup_removed = before_dup - df_clean.shape[0]
    report["removed_records"]["duplicate_records"] = int(dup_removed)
    logger.info(f"Removed {dup_removed} duplicate records.")
    
    # 5. Remove malformed records
    # For example, records where response equals instruction, or contains only noise symbols
    before_malformed = df_clean.shape[0]
    df_clean = df_clean[df_clean['instruction'] != df_clean['response']]
    
    # Filter out records where instruction has no alphabetic chars
    df_clean = df_clean[df_clean['instruction'].apply(lambda x: any(c.isalpha() for c in x))]
    df_clean = df_clean[df_clean['response'].apply(lambda x: any(c.isalpha() for c in x))]
    
    malformed_removed = before_malformed - df_clean.shape[0]
    report["removed_records"]["malformed_records"] = int(malformed_removed)
    logger.info(f"Removed {malformed_removed} malformed records.")
    
    # Final state
    final_shape = df_clean.shape
    report["final_records"] = int(final_shape[0])
    report["total_removed"] = int(initial_shape[0] - final_shape[0])
    
    logger.info(f"Cleaning finished. Final dataset shape: {final_shape}")
    logger.info(f"Total records removed: {report['total_removed']}")
    
    # Save cleaned dataset
    save_csv(df_clean, output_path)
    
    # Save JSON cleaning report
    save_json(report, os.path.join(reports_dir, "cleaning_report.json"))
    
    # Generate before-vs-after text comparison report
    cleaning_summary_md = f"""# Data Cleaning Summary Report

## Overview
- **Initial Dataset Shape**: {initial_shape}
- **Cleaned Dataset Shape**: {final_shape}
- **Total Records Removed**: {report["total_removed"]} ({(report["total_removed"]/initial_shape[0])*100:.2f}%)

## Details of Removed Records
- **Null Values**: {report["removed_records"]["null_values"]}
- **Empty Responses/Instructions**: {report["removed_records"]["empty_strings"]}
- **Duplicate Conversations**: {report["removed_records"]["duplicate_records"]}
- **Malformed / Corrupted Records**: {report["removed_records"]["malformed_records"]}

## Cleaning Heuristics Applied
1. **Unicode Normalization**: Canonical compatibility form (NFKC) was applied.
2. **Whitespace Normalization**: Removed leading/trailing whitespace, multiple spaces normalized to single space, double-newlines preserved for paragraphs.
3. **Punctuation Standardization**: Converted curly quotes, smart apostrophes, and unusual dashes to ASCII equivalents; removed excessive consecutive symbols.
4. **Capitalization Standardization**: Capitalized starting characters of sentences in instructions and responses.
5. **Malformed Filtration**: Filtered out empty responses, matching instruction-response pairs, and text devoid of alphabetic characters.
"""
    
    with open(os.path.join(reports_dir, "cleaning_report.md"), "w", encoding="utf-8") as f:
        f.write(cleaning_summary_md)
        
    logger.info("Data cleaning reports saved.")
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Data Cleaning on raw dataset")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input CSV dataset")
    parser.add_argument("--output_path", type=str, default="data/cleaned/cleaned_dataset.csv", help="Path to output CSV")
    parser.add_argument("--reports_dir", type=str, default="reports/cleaning", help="Reports directory")
    args = parser.parse_args()
    
    run_cleaning(args.input_path, args.output_path, args.reports_dir)
