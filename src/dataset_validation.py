import os
import json
import pandas as pd
from utils.helpers import setup_logger, load_csv, save_json, load_json

def validate_utf8(text):
    """
    Checks if a string can be encoded to UTF-8 without error.
    """
    if not isinstance(text, str):
        return False
    try:
        text.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False

def validate_csv_dataset(df_path, max_query_chars=1000, max_resp_chars=3000):
    """
    Validates intermediate CSV file.
    """
    df = load_csv(df_path)
    total_records = len(df)
    
    missing_fields = 0
    invalid_utf8 = 0
    empty_records = 0
    excessive_query_len = 0
    excessive_resp_len = 0
    
    invalid_rows = []
    
    for idx, row in df.iterrows():
        instruction = row.get('instruction', '')
        response = row.get('response', '')
        
        # 1. Null / Missing field check
        if pd.isna(instruction) or pd.isna(response):
            missing_fields += 1
            invalid_rows.append(idx)
            continue
            
        inst_str = str(instruction).strip()
        resp_str = str(response).strip()
        
        # 2. Empty string check
        if len(inst_str) == 0 or len(resp_str) == 0:
            empty_records += 1
            invalid_rows.append(idx)
            continue
            
        # 3. UTF-8 check
        if not validate_utf8(inst_str) or not validate_utf8(resp_str):
            invalid_utf8 += 1
            invalid_rows.append(idx)
            continue
            
        # 4. Length checks (character boundaries)
        if len(inst_str) > max_query_chars:
            excessive_query_len += 1
        if len(resp_str) > max_resp_chars:
            excessive_resp_len += 1
            
    return {
        "total_records": total_records,
        "missing_fields_count": missing_fields,
        "empty_records_count": empty_records,
        "invalid_utf8_count": invalid_utf8,
        "excessive_query_char_length_count": excessive_query_len,
        "excessive_resp_char_length_count": excessive_resp_len,
        "is_valid": (missing_fields == 0 and empty_records == 0 and invalid_utf8 == 0),
        "invalid_row_indices": invalid_rows
    }

def validate_alpaca_format(json_path):
    """
    Validates Alpaca format output file structure.
    """
    data = load_json(json_path)
    schema_errors = 0
    
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            schema_errors += 1
            continue
        # Check required keys
        required_keys = {"instruction", "input", "output"}
        if not required_keys.issubset(item.keys()):
            schema_errors += 1
            
    return {
        "total_records": len(data),
        "schema_errors": schema_errors,
        "is_valid": (schema_errors == 0)
    }

def validate_chatml_format(json_path):
    """
    Validates ChatML format output file structure.
    """
    data = load_json(json_path)
    schema_errors = 0
    
    for idx, item in enumerate(data):
        if not isinstance(item, list) or len(item) != 2:
            schema_errors += 1
            continue
        user_msg = item[0]
        asst_msg = item[1]
        
        if (not isinstance(user_msg, dict) or user_msg.get('role') != 'user' or 'content' not in user_msg or
            not isinstance(asst_msg, dict) or asst_msg.get('role') != 'assistant' or 'content' not in asst_msg):
            schema_errors += 1
            
    return {
        "total_records": len(data),
        "schema_errors": schema_errors,
        "is_valid": (schema_errors == 0)
    }

def validate_qwen_format(json_path):
    """
    Validates Qwen format output file structure.
    """
    data = load_json(json_path)
    schema_errors = 0
    
    for idx, item in enumerate(data):
        if not isinstance(item, dict) or 'messages' not in item:
            schema_errors += 1
            continue
        messages = item['messages']
        if not isinstance(messages, list) or len(messages) < 2:
            schema_errors += 1
            continue
        # Verify structure
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                schema_errors += 1
                break
                
    return {
        "total_records": len(data),
        "schema_errors": schema_errors,
        "is_valid": (schema_errors == 0)
    }

def run_validation(csv_path, final_dir, reports_dir):
    """
    Runs full validation suite on intermediate and final datasets.
    """
    os.makedirs(reports_dir, exist_ok=True)
    
    logger = setup_logger("dataset_validation", os.path.join(reports_dir, "dataset_validation.log"))
    logger.info("Starting Dataset Validation suite...")
    
    # 1. Validate CSV
    logger.info(f"Validating intermediate CSV: {csv_path}")
    csv_report = validate_csv_dataset(csv_path)
    
    # 2. Validate Formatted Datasets
    alpaca_path = os.path.join(final_dir, "dataset_alpaca.json")
    chatml_path = os.path.join(final_dir, "dataset_chatml.json")
    qwen_path = os.path.join(final_dir, "dataset_qwen.json")
    
    logger.info("Validating Alpaca format JSON...")
    alpaca_report = validate_alpaca_format(alpaca_path)
    
    logger.info("Validating ChatML format JSON...")
    chatml_report = validate_chatml_format(chatml_path)
    
    logger.info("Validating Qwen format JSON...")
    qwen_report = validate_qwen_format(qwen_path)
    
    # Aggregate Report
    report = {
        "csv_validation": csv_report,
        "alpaca_validation": alpaca_report,
        "chatml_validation": chatml_report,
        "qwen_validation": qwen_report,
        "overall_valid": csv_report["is_valid"] and alpaca_report["is_valid"] and chatml_report["is_valid"] and qwen_report["is_valid"]
    }
    
    save_json(report, os.path.join(reports_dir, "validation_report.json"))
    
    # Markdown Report
    validation_report_md = f"""# Dataset Validation Report

## Overall Status
- **Pipeline Validation Passed**: {"✅ PASSED" if report["overall_valid"] else "❌ FAILED"}

## Intermediate CSV Dataset Validation
- **Path**: `{csv_path}`
- **Total Records Checked**: {csv_report["total_records"]}
- **Missing Fields**: {csv_report["missing_fields_count"]}
- **Empty Records**: {csv_report["empty_records_count"]}
- **Invalid UTF-8 Encodings**: {csv_report["invalid_utf8_count"]}
- **Query Char Length Outliers (> 1000 chars)**: {csv_report["excessive_query_char_length_count"]}
- **Response Char Length Outliers (> 3000 chars)**: {csv_report["excessive_resp_char_length_count"]}
- **Status**: {"✅ Valid" if csv_report["is_valid"] else "❌ Invalid"}

## Final Export Format Validations

### Alpaca JSON
- **Path**: `dataset_alpaca.json`
- **Total Records Checked**: {alpaca_report["total_records"]}
- **Schema Key Errors**: {alpaca_report["schema_errors"]}
- **Status**: {"✅ Valid" if alpaca_report["is_valid"] else "❌ Invalid"}

### ChatML JSON
- **Path**: `dataset_chatml.json`
- **Total Records Checked**: {chatml_report["total_records"]}
- **Schema Structure Errors**: {chatml_report["schema_errors"]}
- **Status**: {"✅ Valid" if chatml_report["is_valid"] else "❌ Invalid"}

### Qwen JSON
- **Path**: `dataset_qwen.json`
- **Total Records Checked**: {qwen_report["total_records"]}
- **Schema Template Errors**: {qwen_report["schema_errors"]}
- **Status**: {"✅ Valid" if qwen_report["is_valid"] else "❌ Invalid"}
"""
    
    with open(os.path.join(reports_dir, "validation_report.md"), "w", encoding="utf-8") as f:
        f.write(validation_report_md)
        
    logger.info("Validation reports generated.")
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Dataset Validation")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input augmented CSV dataset")
    parser.add_argument("--final_dir", type=str, default="data/final", help="Final directory containing json datasets")
    parser.add_argument("--reports_dir", type=str, default="reports/validation", help="Reports directory")
    args = parser.parse_args()
    
    run_validation(args.csv_path, args.final_dir, args.reports_dir)
