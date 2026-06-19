import os
import json
import csv
import pandas as pd
from fpdf import FPDF
from utils.helpers import setup_logger, load_json, save_json

class PremiumPDF(FPDF):
    def header(self):
        # Draw header banner
        self.set_fill_color(31, 58, 86)  # Deep Navy
        self.rect(0, 0, 210, 25, "F")
        
        self.set_font("helvetica", "B", 14)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "CUSTOMERGPT DATA ENGINEERING PIPELINE REPORT", align="C", ln=True)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} | CustomerGPT Fine-Tuning Prep Pipeline", align="R")

def generate_pdf_report(consolidated, charts_dir, output_path):
    """
    Compiles the final PDF report with clean structure, premium formatting, and embedded charts.
    """
    pdf = PremiumPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    # 1. Executive Summary
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(31, 58, 86) # Deep Navy
    pdf.cell(0, 8, "1. Executive Summary", ln=True)
    pdf.set_draw_color(31, 58, 86)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    summary_text = (
        "This report summarizes the data engineering and quality assurance pipeline executed for CustomerGPT fine-tuning. "
        "The raw customer support dataset was analyzed, cleaned, filtered for quality, enriched with custom metadata fields, "
        "synthetically augmented for conversational diversity, validated, tokenized, and split into train, validation, and test partitions. "
        "No model training is initiated until all metrics are verified in this report."
    )
    pdf.multi_cell(0, 5, summary_text)
    pdf.ln(5)
    
    # 2. Dataset Progression Table
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(31, 58, 86)
    pdf.cell(0, 8, "Dataset Progression Summary", ln=True)
    pdf.ln(1)
    
    # Draw Table
    pdf.set_font("helvetica", "B", 10)
    pdf.set_fill_color(240, 244, 248) # Ice Blue
    pdf.set_text_color(31, 58, 86)
    pdf.cell(60, 8, " Pipeline Stage", border=1, fill=True)
    pdf.cell(45, 8, " Record Count", border=1, fill=True)
    pdf.cell(45, 8, " Delta (Removed/Added)", border=1, fill=True)
    pdf.cell(40, 8, " Status", border=1, fill=True, ln=True)
    
    stages = [
        ("Raw Input", consolidated.get("raw_records"), "-", "Loaded"),
        ("Cleaned", consolidated.get("cleaned_records"), f"-{consolidated.get('removed_cleaning_count')}", "Normalized"),
        ("Quality Filtered", consolidated.get("quality_passed_records"), f"-{consolidated.get('removed_quality_count')}", "High Quality"),
        ("Augmented (Final)", consolidated.get("augmented_records"), f"+{consolidated.get('augmented_added_count')}", "Enriched & Expanded")
    ]
    
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    for stage, count, delta, status in stages:
        pdf.cell(60, 8, f" {stage}", border=1)
        pdf.cell(45, 8, f" {count:,}", border=1)
        pdf.cell(45, 8, f" {delta}", border=1)
        pdf.cell(40, 8, f" {status}", border=1, ln=True)
        
    pdf.ln(6)
    
    # 3. Data Cleaning Details
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(31, 58, 86)
    pdf.cell(0, 8, "2. Data Cleaning Summary", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font("helvetica", "", 10)
    cleaning_text = (
        f"- Null Values Removed: {consolidated.get('cleaning', {}).get('removed_records', {}).get('null_values', 0):,}\n"
        f"- Empty Responses Removed: {consolidated.get('cleaning', {}).get('removed_records', {}).get('empty_strings', 0):,}\n"
        f"- Duplicate Conversations Removed: {consolidated.get('cleaning', {}).get('removed_records', {}).get('duplicate_records', 0):,}\n"
        f"- Malformed Records Filtered: {consolidated.get('cleaning', {}).get('removed_records', {}).get('malformed_records', 0):,}\n"
    )
    pdf.multi_cell(0, 5, cleaning_text)
    pdf.ln(3)
    
    # 4. Quality Scoring & Filtration
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(31, 58, 86)
    pdf.cell(0, 8, "3. Quality Scoring & Filtration", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font("helvetica", "", 10)
    quality_text = (
        f"- Minimum Quality Score Threshold: {consolidated.get('quality', {}).get('threshold', 5.0)} / 10.0\n"
        f"- Mean Quality Score: {consolidated.get('quality', {}).get('mean_quality_score', 0):.2f}\n"
        f"- Records Passed Threshold: {consolidated.get('quality', {}).get('total_records_passed', 0):,}\n"
        f"- Records Dropped (Score < threshold): {consolidated.get('quality', {}).get('total_records_filtered', 0):,}\n\n"
        f"Component Averages (1-10 Scale):\n"
        f"  * Grammar: {consolidated.get('quality', {}).get('component_averages', {}).get('grammar', 0):.2f} | "
        f"Completeness: {consolidated.get('quality', {}).get('component_averages', {}).get('completeness', 0):.2f} | "
        f"Length Quality: {consolidated.get('quality', {}).get('component_averages', {}).get('length', 0):.2f}\n"
        f"  * Relevance: {consolidated.get('quality', {}).get('component_averages', {}).get('relevance', 0):.2f} | "
        f"Readability: {consolidated.get('quality', {}).get('component_averages', {}).get('readability', 0):.2f}"
    )
    pdf.multi_cell(0, 5, quality_text)
    
    pdf.add_page()
    
    # 5. Enrichment & Augmentation
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(31, 58, 86)
    pdf.cell(0, 8, "4. Metadata Enrichment & Synthetic Augmentation", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font("helvetica", "", 10)
    aug_stats = consolidated.get('augmentation', {})
    enrich_stats = consolidated.get('enrichment', {})
    
    enrich_text = (
        f"Metadata Enrichment Distributions:\n"
        f"- Sentiment: Positive={enrich_stats.get('sentiment_distribution', {}).get('positive', 0):,}, "
        f"Neutral={enrich_stats.get('sentiment_distribution', {}).get('neutral', 0):,}, "
        f"Negative={enrich_stats.get('sentiment_distribution', {}).get('negative', 0):,}\n"
        f"- Urgency: Low={enrich_stats.get('urgency_distribution', {}).get('low', 0):,}, "
        f"Medium={enrich_stats.get('urgency_distribution', {}).get('medium', 0):,}, "
        f"High={enrich_stats.get('urgency_distribution', {}).get('high', 0):,}\n"
        f"- Escalations Tagged: {enrich_stats.get('escalation_distribution', {}).get('True', 0):,} conversations require human support.\n\n"
        f"Synthetic Augmentation Output:\n"
        f"- Total Augmented Samples Added: {aug_stats.get('added_records', 0):,} (2-3 variations generated per core conversation)\n"
        f"- Typo Variations: {aug_stats.get('augmentation_distribution', {}).get('typos', 0):,}\n"
        f"- Short-form/Long-form Variations: {aug_stats.get('augmentation_distribution', {}).get('short_form', 0) + aug_stats.get('augmentation_distribution', {}).get('long_form', 0):,}\n"
        f"- Paraphrased Customer Queries + Rewritten Agent Responses: {aug_stats.get('augmentation_distribution', {}).get('paraphrase_rewrite', 0):,}"
    )
    pdf.multi_cell(0, 5, enrich_text)
    pdf.ln(3)
    
    # 6. Tokenization Analysis & Final Splits
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(31, 58, 86)
    pdf.cell(0, 8, "5. Tokenization & Final Dataset Split", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    token_stats = consolidated.get('tokenization', {})
    split_stats = consolidated.get('split', {})
    
    token_text = (
        f"Sequence Length Analysis (Tokenizer: {token_stats.get('tokenizer_used')}):\n"
        f"- Minimum Length: {token_stats.get('min_tokens', 0)} tokens\n"
        f"- Maximum Length: {token_stats.get('max_tokens', 0)} tokens\n"
        f"- Average Sequence Length: {token_stats.get('avg_tokens_per_sample', 0):.2f} tokens\n"
        f"- 95th Percentile: {token_stats.get('percentiles', {}).get('p95', 0)} tokens\n"
        f"- 99th Percentile: {token_stats.get('percentiles', {}).get('p99', 0)} tokens\n"
        f"- RECOMMENDED MAX SEQUENCE LENGTH: {token_stats.get('recommended_max_sequence_length', 512)} tokens\n\n"
        f"Final Stratified Partition Splits:\n"
        f"- Train Split (80%): {split_stats.get('train_count', 0):,} samples -> saved to data/final/train.jsonl\n"
        f"- Validation Split (10%): {split_stats.get('validation_count', 0):,} samples -> saved to data/final/validation.jsonl\n"
        f"- Test Split (10%): {split_stats.get('test_count', 0):,} samples -> saved to data/final/test.jsonl"
    )
    pdf.multi_cell(0, 5, token_text)
    pdf.ln(5)
    
    # 7. Embed Visualizations
    pdf.add_page()
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(31, 58, 86)
    pdf.cell(0, 8, "6. Exploratory Data Analysis Charts", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Add length distribution chart
    length_chart_path = os.path.join(charts_dir, "length_distributions.png")
    if os.path.exists(length_chart_path):
        pdf.image(length_chart_path, x=10, y=pdf.get_y(), w=190, h=70)
        pdf.set_y(pdf.get_y() + 75)
        
    # Add category distribution chart
    cat_chart_path = os.path.join(charts_dir, "category_distribution.png")
    if os.path.exists(cat_chart_path):
        pdf.image(cat_chart_path, x=10, y=pdf.get_y(), w=190, h=70)
        
    pdf.output(output_path)

def compile_consolidated_report(reports_dir):
    """
    Reads JSON outputs from all steps and merges them.
    Writes merged JSON, flattened CSV, and premium PDF.
    """
    logger = setup_logger("generate_reports", os.path.join(reports_dir, "reports_compilation.log"))
    logger.info("Compiling consolidated Data Engineering report...")
    
    # Load JSON files
    eda_stats = load_json(os.path.join(reports_dir, "eda", "eda_statistics.json"))
    cleaning_stats = load_json(os.path.join(reports_dir, "cleaning", "cleaning_report.json"))
    quality_stats = load_json(os.path.join(reports_dir, "quality", "quality_scoring_report.json"))
    enrichment_stats = load_json(os.path.join(reports_dir, "quality", "enrichment_report.json"))
    aug_stats = load_json(os.path.join(reports_dir, "quality", "augmentation_report.json"))
    validation_stats = load_json(os.path.join(reports_dir, "validation", "validation_report.json"))
    token_stats = load_json(os.path.join(reports_dir, "tokenization", "token_statistics.json"))
    split_stats = load_json(os.path.join(reports_dir, "validation", "split_statistics.json"))
    
    consolidated = {
        "raw_records": eda_stats.get("shape", {}).get("rows", 0),
        "cleaned_records": cleaning_stats.get("final_records", 0),
        "quality_passed_records": quality_stats.get("total_records_passed", 0),
        "augmented_records": aug_stats.get("total_records_after_augmentation", 0),
        "removed_cleaning_count": cleaning_stats.get("total_removed", 0),
        "removed_quality_count": quality_stats.get("total_records_filtered", 0),
        "augmented_added_count": aug_stats.get("added_records", 0),
        "train_records": split_stats.get("train_count", 0),
        "val_records": split_stats.get("validation_count", 0),
        "test_records": split_stats.get("test_count", 0),
        
        "eda": eda_stats,
        "cleaning": cleaning_stats,
        "quality": quality_stats,
        "enrichment": enrichment_stats,
        "augmentation": aug_stats,
        "validation": validation_stats,
        "tokenization": token_stats,
        "split": split_stats
    }
    
    # Save Consolidated JSON
    json_out = os.path.join(reports_dir, "data_engineering_report.json")
    save_json(consolidated, json_out)
    logger.info(f"Consolidated JSON report written to: {json_out}")
    
    # Save Flattened CSV
    csv_out = os.path.join(reports_dir, "data_engineering_report.csv")
    flattened_metrics = [
        ("raw_records", consolidated["raw_records"]),
        ("cleaned_records", consolidated["cleaned_records"]),
        ("quality_passed_records", consolidated["quality_passed_records"]),
        ("augmented_records", consolidated["augmented_records"]),
        ("train_records", consolidated["train_records"]),
        ("val_records", consolidated["val_records"]),
        ("test_records", consolidated["test_records"]),
        ("removed_in_cleaning", consolidated["removed_cleaning_count"]),
        ("removed_in_quality_filtering", consolidated["removed_quality_count"]),
        ("augmented_variations_added", consolidated["augmented_added_count"]),
        ("average_quality_score", consolidated["quality"].get("mean_quality_score")),
        ("average_tokens_per_sample", consolidated["tokenization"].get("avg_tokens_per_sample")),
        ("max_tokens_sample", consolidated["tokenization"].get("max_tokens")),
        ("recommended_max_sequence_length", consolidated["tokenization"].get("recommended_max_sequence_length")),
        ("pipeline_validation_passed", consolidated["validation"].get("overall_valid"))
    ]
    
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric Key", "Metric Value"])
        for key, val in flattened_metrics:
            writer.writerow([key, val])
    logger.info(f"Flattened CSV report written to: {csv_out}")
    
    # Save Premium PDF
    pdf_out = os.path.join(reports_dir, "data_engineering_report.pdf")
    generate_pdf_report(consolidated, os.path.join(reports_dir, "eda"), pdf_out)
    logger.info(f"Premium PDF report written to: {pdf_out}")
    
    logger.info("All reports compiled successfully!")
    return consolidated

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compile Consolidated Pipeline Reports")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Base reports directory")
    args = parser.parse_args()
    
    compile_consolidated_report(args.reports_dir)
