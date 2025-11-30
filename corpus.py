# prepare_corpus.py
import os
import sys # Import sys to exit gracefully
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
# ⬇ ⬇ ⬇ EDIT THIS LINE WITH THE CORRECT FULL PATH ⬇ ⬇ ⬇
BASE_DIR = "C:/Users/Yeshwanth Reddy A R/Downloads/pibb/final_data" # <-- EXAMPLE PATH, REPLACE IT

# Where we will save the new CSV files
OUTPUT_DIR = "processed_corpus"

# --- New: Check if the base directory actually exists ---
if not os.path.isdir(BASE_DIR):
    print(f"❌ ERROR: The specified directory was not found: '{BASE_DIR}'")
    print("Please check the path in the script and make sure it's correct.")
    sys.exit() # Exit the script if the path is wrong

os.makedirs(OUTPUT_DIR, exist_ok=True)

lang_pair_dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

# --- New: Check if any language folders were found ---
if not lang_pair_dirs:
    print(f"⚠ Warning: Found the directory '{BASE_DIR}', but it contains no language-pair subfolders (e.g., 'en-hi').")
    print("Please ensure your data is organized in subfolders. Exiting.")
    sys.exit()

print(f"Found {len(lang_pair_dirs)} language pairs in '{BASE_DIR}'. Starting processing...")

# (The rest of the script is the same as before)
for pair_dir in tqdm(lang_pair_dirs, desc="Processing languages"):
    source_lang, target_lang = pair_dir.split('-')
    source_file_path = os.path.join(BASE_DIR, pair_dir, f"train.{source_lang}")
    target_file_path = os.path.join(BASE_DIR, pair_dir, f"train.{target_lang}")
    try:
        with open(source_file_path, 'r', encoding='utf-8') as f:
            source_sentences = [line.strip() for line in f.readlines()]
        with open(target_file_path, 'r', encoding='utf-8') as f:
            target_sentences = [line.strip() for line in f.readlines()]
        if len(source_sentences) != len(target_sentences):
            print(f"\n⚠  Warning: Mismatch in line count for {pair_dir}. Skipping.")
            continue
        df = pd.DataFrame({source_lang: source_sentences, target_lang: target_sentences})
        output_csv_path = os.path.join(OUTPUT_DIR, f"{pair_dir}.csv")
        df.to_csv(output_csv_path, index=False)
    except FileNotFoundError:
        print(f"\n⚠  Warning: Could not find expected files in {pair_dir}. Skipping.")
    except Exception as e:
        print(f"\nAn error occurred while processing {pair_dir}: {e}")

print(f"\n✅ Processing complete! All CSV files are saved in the '{OUTPUT_DIR}' folder.")