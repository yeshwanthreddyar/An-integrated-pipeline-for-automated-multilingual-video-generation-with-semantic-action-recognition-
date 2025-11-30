# preprocess.py
import pandas as pd
import re
from pathlib import Path
from rapidfuzz import fuzz

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(t):
    if not isinstance(t, str): return ""
    t = re.sub(r'\xa0',' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def split_into_sentences(text):
    if not text: return []
    sents = re.split(r'(?<=[।\.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]

def preprocess_file(csv_path):
    try:
        # Try to read the file
        df = pd.read_csv(csv_path, encoding="utf-8")
        # Also check if the dataframe is empty (e.g., only headers)
        if df.empty:
            print(f"⚠  Warning: {csv_path.name} is empty or has only headers. Skipping.")
            return

    except pd.errors.EmptyDataError:
        # If the file is completely empty, catch the error
        print(f"⚠  Warning: {csv_path.name} is an empty file. Skipping.")
        return

    df['text_clean'] = df['text'].fillna('').apply(clean_text)
    df['sentences'] = df['text_clean'].apply(split_into_sentences)

    out_path = PROC_DIR / Path(csv_path).name
    df.to_parquet(out_path.with_suffix('.parquet'), index=False)
    print(f"✅ Saved {out_path.with_suffix('.parquet')}")
    return out_path.with_suffix('.parquet')

if __name__ == "__main__":
    print("Starting preprocessing...")
    processed_count = 0
    for p in RAW_DIR.glob("*.csv"):
        if preprocess_file(p):
            processed_count += 1
    print(f"\nPreprocessing complete. Processed {processed_count} file(s).")