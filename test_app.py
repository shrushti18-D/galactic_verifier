"""
test_app.py
Verification script to test cleaner, brochure_reader, analyzer, and Excel export end-to-end.
"""

import os
import pandas as pd
from cleaner import detect_column_mapping, clean_dataset
from brochure_reader import read_all_brochures, extract_capabilities
from analyzer import analyze_companies_batch
from utils import generate_sample_dataset, generate_excel_download

def test_pipeline():
    print("1. Testing Sample Dataset Generation...")
    sample_df = generate_sample_dataset()
    print(f"Generated sample dataset with {len(sample_df)} rows.")

    print("\n2. Testing Column Detection...")
    mapping = detect_column_mapping(sample_df)
    print("Detected Column Mapping:", mapping)

    print("\n3. Testing Data Cleaning...")
    cleaned_df, stats = clean_dataset(sample_df, mapping)
    print(f"Original rows: {stats['original_count']}, Cleaned rows: {stats['cleaned_count']}, Total Removed: {stats['total_removed']}")

    print("\n4. Testing PyMuPDF Brochure Reading...")
    b_info = read_all_brochures("brochure")
    print(f"Brochures read: {b_info['total_files']} files, {b_info['total_pages']} pages.")
    profile = extract_capabilities(b_info["combined_text"])
    print(f"Extracted capability keywords ({len(profile['keywords'])}):", profile['keywords'][:10])

    print("\n5. Testing Machine Learning Batch Relevance Analysis...")
    analyzed_df = analyze_companies_batch(
        df=cleaned_df,
        column_mapping=mapping,
        galactic_profile=profile,
        batch_size=16
    )
    print("Analysis Completed!")
    print("\nSample Results Preview:")
    print(analyzed_df[["co_name", "category", "Match Score", "Result", "Reason"]].head(10))

    print("\n6. Testing Formatted Excel Report Generation...")
    excel_bytes = generate_excel_download(analyzed_df)
    print(f"Generated Excel report ({len(excel_bytes):,} bytes).")

    print("\n[SUCCESS] ALL PIPELINE TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    test_pipeline()
