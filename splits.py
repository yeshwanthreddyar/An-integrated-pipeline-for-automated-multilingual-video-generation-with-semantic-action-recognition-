# create_all_splits.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# --- Configuration ---
CORPUS_DIR = "processed_corpus"
OUTPUT_DIR = "data_splits"

# Get a list of all CSV files in the corpus directory
try:
    csv_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith('.csv')]
    if not csv_files:
        print(f"❌ ERROR: No CSV files found in '{CORPUS_DIR}'.")
        exit()
except FileNotFoundError:
    print(f"❌ ERROR: The directory '{CORPUS_DIR}' was not found.")
    exit()

print(f"Found {len(csv_files)} language pairs to split. Starting process...")

# Loop through each CSV file
for filename in tqdm(csv_files, desc="Splitting datasets"):
    lang_pair = filename.replace('.csv', '') # e.g., 'en-hi'
    input_csv_path = os.path.join(CORPUS_DIR, filename)
    
    # Create a dedicated output folder for this language pair
    lang_output_dir = os.path.join(OUTPUT_DIR, lang_pair)
    os.makedirs(lang_output_dir, exist_ok=True)

    # Load the data
    df = pd.read_csv(input_csv_path)

    # Split the data
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Save the splits into the dedicated subfolder
    train_df.to_csv(os.path.join(lang_output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(lang_output_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(lang_output_dir, "test.csv"), index=False)

print(f"\n✅ All datasets have been split and saved in the '{OUTPUT_DIR}' folder.")