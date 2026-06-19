import os
import shutil
import argparse
import sys
from src.utils.helpers import setup_directories, setup_logger

def run_pipeline(raw_dataset_path, quality_threshold, base_dir):
    """
    Main orchestrator for Phase 4A Data Engineering Pipeline.
    """
    # 1. Setup folders
    setup_directories(base_dir)
    reports_dir = os.path.join(base_dir, "reports")
    logger = setup_logger("pipeline_orchestrator", os.path.join(reports_dir, "pipeline.log"))
    
    logger.info("=================================================================")
    logger.info("STARTING PHASE 4A — DATA PREPARATION & QUALITY PIPELINE")
    logger.info("=================================================================")
    logger.info(f"Raw Dataset Path: {raw_dataset_path}")
    logger.info(f"Quality Score Threshold: {quality_threshold}")
    logger.info(f"Workspace Directory: {base_dir}")
    
    # 2. Version Raw File
    raw_dest = os.path.join(base_dir, "data", "raw", os.path.basename(raw_dataset_path))
    if not os.path.exists(raw_dest):
        logger.info(f"Copying raw dataset to versioned raw directory: {raw_dest}")
        shutil.copy(raw_dataset_path, raw_dest)
    else:
        logger.info("Raw dataset already versioned in data/raw/.")
        
    # Import pipeline scripts
    sys.path.append(os.path.join(base_dir, "src"))
    from eda_analysis import run_eda
    from data_cleaning import run_cleaning
    from quality_scoring import score_and_filter
    from data_enrichment import run_enrichment
    from data_augmentation import run_augmentation
    from chat_formatting import run_formatting
    from dataset_validation import run_validation
    from tokenization_analysis import run_tokenization_analysis
    from dataset_split import run_split
    from generate_reports import compile_consolidated_report

    # Path setups
    cleaned_csv = os.path.join(base_dir, "data", "cleaned", "cleaned_dataset.csv")
    filtered_csv = os.path.join(base_dir, "data", "cleaned", "cleaned_filtered_dataset.csv")
    enriched_csv = os.path.join(base_dir, "data", "cleaned", "cleaned_enriched_dataset.csv")
    augmented_csv = os.path.join(base_dir, "data", "augmented", "augmented_dataset.csv")
    final_dir = os.path.join(base_dir, "data", "final")
    
    # Run Step 1: EDA Analysis
    logger.info("\n--- STEP 1: Running EDA & Dataset Analysis ---")
    run_eda(raw_dest, os.path.join(base_dir, "data", "raw"), os.path.join(reports_dir, "eda"))
    
    # Run Step 2: Data Cleaning
    logger.info("\n--- STEP 2: Running Data Cleaning ---")
    run_cleaning(raw_dest, cleaned_csv, os.path.join(reports_dir, "cleaning"))
    
    # Run Step 3: Data Quality Scoring
    logger.info("\n--- STEP 3: Running Quality Scoring & Filtering ---")
    score_and_filter(cleaned_csv, filtered_csv, os.path.join(reports_dir, "quality"), quality_threshold)
    
    # Run Step 4: Data Enrichment
    logger.info("\n--- STEP 4: Running Data Enrichment (Sentiment, Urgency, Complexity, Escalation) ---")
    run_enrichment(filtered_csv, enriched_csv, os.path.join(reports_dir, "quality"))
    
    # Run Step 5: Data Augmentation
    logger.info("\n--- STEP 5: Running Synthetic Data Augmentation (Typos, Short/Long form, Paraphrases) ---")
    run_augmentation(enriched_csv, augmented_csv, os.path.join(reports_dir, "quality"))
    
    # Run Step 6: Chat Formatting
    logger.info("\n--- STEP 6: Formatting Conversations (Alpaca, ChatML, Qwen) ---")
    run_formatting(augmented_csv, final_dir, os.path.join(reports_dir, "quality"))
    
    # Run Step 7: Dataset Validation
    logger.info("\n--- STEP 7: Validating Dataset Integrity & Schema Compliance ---")
    run_validation(augmented_csv, final_dir, os.path.join(reports_dir, "validation"))
    
    # Run Step 8: Tokenization Analysis
    logger.info("\n--- STEP 8: Analyzing Token Distribution & Recommendations ---")
    run_tokenization_analysis(final_dir, os.path.join(reports_dir, "tokenization"))
    
    # Run Step 9: Final Dataset Partition Split
    logger.info("\n--- STEP 9: Splitting Final Dataset (Train 80% / Val 10% / Test 10%) ---")
    run_split(final_dir, os.path.join(reports_dir, "validation"))
    
    # Run Step 10: Consolidated Reporting (PDF, JSON, CSV)
    logger.info("\n--- STEP 10: Generating Consolidated Data Engineering Reports ---")
    compile_consolidated_report(reports_dir)
    
    logger.info("=================================================================")
    logger.info("PHASE 4A PIPELINE EXECUTED SUCCESSFULLY!")
    logger.info("Generated: JSON, CSV, and PDF reports in reports/ folder.")
    logger.info("Generated: train.jsonl, validation.jsonl, test.jsonl in data/final/")
    logger.info("=================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete Data Engineering Pipeline for Phase 4A")
    parser.add_argument("--raw_path", type=str, default="Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv", help="Path to raw CSV dataset")
    parser.add_argument("--quality_threshold", type=float, default=5.0, help="Threshold for quality scoring (1-10)")
    parser.add_argument("--base_dir", type=str, default="d:/Rudraksh/College/app/CustomerGPT-FineTunning", help="Base workspace directory")
    args = parser.parse_args()
    
    run_pipeline(args.raw_path, args.quality_threshold, args.base_dir)
