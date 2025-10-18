#!/usr/bin/env python3
"""
Script to add item_id column to CSV file for direct FAISS index creation.
This creates a copy of the original CSV with sequential item_ids.
"""
import pandas as pd
import sys

def add_item_id_to_csv(input_csv, output_csv=None):
    """
    Add sequential item_id column to CSV file.

    Args:
        input_csv (str): Path to input CSV file
        output_csv (str, optional): Path to output CSV file. If None, adds '_with_ids' suffix.
    """
    if output_csv is None:
        # Insert '_with_ids' before the file extension
        parts = input_csv.rsplit('.', 1)
        if len(parts) == 2:
            output_csv = f"{parts[0]}_with_ids.{parts[1]}"
        else:
            output_csv = f"{input_csv}_with_ids"

    print(f"Reading CSV from: {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"Original CSV shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    # Add sequential item_id starting from 0
    df.insert(0, 'item_id', range(len(df)))

    print(f"Added item_id column. New shape: {df.shape}")
    print(f"New columns: {list(df.columns)}")

    print(f"Saving CSV with item_ids to: {output_csv}")
    df.to_csv(output_csv, index=False)

    print(f"✅ Successfully created CSV with {len(df)} rows and item_ids")
    return output_csv

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_item_id_to_csv.py <input_csv> [output_csv]")
        print("Example: python add_item_id_to_csv.py wikipedia_sample_150k.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result_csv = add_item_id_to_csv(input_csv, output_csv)
        print(f"\n🎉 CSV preparation complete! Output file: {result_csv}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
