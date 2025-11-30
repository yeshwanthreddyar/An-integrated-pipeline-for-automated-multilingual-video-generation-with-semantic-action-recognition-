import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import evaluate
import torch
from tqdm import tqdm
import os
import time

# --- Configuration ---
SPLITS_DIR = "data_splits"
SAMPLE_SIZE = 1000

MODEL_MAP = {
    "as": "Helsinki-NLP/opus-mt-en-mul",
    "ta": "suriya7/English-to-Tamil",
    "te": "aryaumesh/english-to-telugu",
    "bn": "Helsinki-NLP/opus-mt-en-mul",
    "gu": "Helsinki-NLP/opus-mt-en-mul",
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "kn": "Helsinki-NLP/opus-mt-en-mul",
    "ml": "Helsinki-NLP/opus-mt-en-mul",
    "mr": "Helsinki-NLP/opus-mt-en-mul",
    "or": "Helsinki-NLP/opus-mt-en-mul",
    "pa": "Helsinki-NLP/opus-mt-en-mul",
}

FALLBACK_MODEL = "Helsinki-NLP/opus-mt-en-mul"


# --- Safe BLEU Loader ---
def load_bleu_metric():
    """Load sacrebleu metric with auto-repair if cache is corrupted"""
    try:
        bleu = evaluate.load("sacrebleu")
        print("✅ BLEU metric loaded successfully (local cache).")
        return bleu
    except Exception as e1:
        print(f"⚠️  Local sacrebleu load failed: {e1}")
        print("🔁 Retrying with forced remote download...")
        try:
            bleu = evaluate.load("sacrebleu", download_mode="force_redownload")
            print("✅ BLEU metric downloaded fresh from Hugging Face.")
            return bleu
        except Exception as e2:
            print(f"❌ Failed to load sacrebleu after retry: {e2}")
            raise


def evaluate_language_pair(lang_pair, sample_size=1000):
    """Evaluate translation model for one language pair"""
    print(f"\n--- Evaluating: {lang_pair} ---")

    source_lang, target_lang_short = lang_pair.split('-')
    test_file_path = os.path.join(SPLITS_DIR, lang_pair, "test.csv")

    if not os.path.exists(test_file_path):
        print(f"❌ Test file not found: {test_file_path}")
        return None

    try:
        df = pd.read_csv(
            test_file_path,
            engine="python",
            on_bad_lines="skip",
            encoding_errors="ignore"
        )

        if df.empty:
            print(f"⚠️  {lang_pair}: Empty CSV, skipping.")
            return None

        actual_sample_size = min(sample_size, len(df))
        test_df = df.sample(n=actual_sample_size, random_state=42)

        source_sentences = test_df[source_lang].astype(str).tolist()
        reference_translations = test_df[target_lang_short].astype(str).tolist()

        print(f"📊 Testing with {len(source_sentences)} sentences")

    except Exception as e:
        print(f"❌ Error loading data for {lang_pair}: {e}")
        return None

    model_name = MODEL_MAP.get(target_lang_short, FALLBACK_MODEL)
    print(f"🤖 Using model: {model_name}")

    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"🖥️ Using device: {device}")

        model_translations = []
        batch_size = 16 if device == "cuda" else 8

        for i in tqdm(range(0, len(source_sentences), batch_size), desc=f"Translating {lang_pair}"):
            batch = source_sentences[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=512)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            model_translations.extend(decoded)

        # --- Safe BLEU metric ---
        bleu = load_bleu_metric()
        score = bleu.compute(predictions=model_translations, references=[[r] for r in reference_translations])

        print(f"✅ {lang_pair}: BLEU = {score['score']:.2f}")

        print("\n📝 Sample Translation:")
        print(f"EN: {source_sentences[0][:100]}...")
        print(f"Model: {model_translations[0][:100]}...")
        print(f"Reference: {reference_translations[0][:100]}...")

        return score['score']

    except Exception as e:
        print(f"❌ Error with {lang_pair}: {e}")
        return 0.0


if __name__ == "__main__":
    start_time = time.time()
    lang_pair_dirs = [
        d for d in os.listdir(SPLITS_DIR)
        if os.path.isdir(os.path.join(SPLITS_DIR, d)) and d.startswith('en-')
    ]

    print("🌍 Starting evaluation for all languages...")
    print(f"📁 Found {len(lang_pair_dirs)} language pairs: {lang_pair_dirs}")

    results_summary = {}

    for lang_pair in sorted(lang_pair_dirs):
        score = evaluate_language_pair(lang_pair, SAMPLE_SIZE)
        if score is not None:
            results_summary[lang_pair] = score

    results_df = pd.DataFrame.from_dict(results_summary, orient='index', columns=['BLEU_Score'])
    results_df.to_csv('baseline_results.csv')

    print("\n" + "="*60)
    print("📊 COMPREHENSIVE BASELINE EVALUATION RESULTS")
    print("="*60)
    for lang_pair, score in sorted(results_summary.items()):
        print(f"{lang_pair.upper():<12} | BLEU Score: {score:>6.2f}")
    print("="*60)

    if results_summary:
        avg = sum(results_summary.values()) / len(results_summary)
        print(f"Average BLEU Score: {avg:.2f}")

    print(f"⏰ Total runtime: {(time.time() - start_time)/60:.2f} minutes")
