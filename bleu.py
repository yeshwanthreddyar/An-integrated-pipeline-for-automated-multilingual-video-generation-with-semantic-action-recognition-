import os
import re
import unicodedata
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    M2M100Tokenizer, M2M100ForConditionalGeneration
)
import evaluate
import matplotlib.pyplot as plt

# ============================================================
# ⚙️ CONFIGURATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "./final_data"
MODEL_DIR = "./mass_fine_tuned_models"
OUTPUT_DIR = "./bleu_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 4
MAX_NEW_TOKENS = 128

LANG_PAIRS = [
    "en-as", "en-bn", "en-gu", "en-hi", "en-kn",
    "en-ml", "en-mr", "en-or", "en-pa", "en-ta", "en-te"
]

# ============================================================
# 🧹 TEXT CLEANING
# ============================================================
def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace("\u200c", "").replace("\u200b", "")
    return text.strip()

def clean_file(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        lines = [clean_text(l) for l in f if l.strip()]
    lines = list(dict.fromkeys(lines))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"🧼 Cleaned {path} ({len(lines)} lines)")

def auto_clean_dataset(lang_pair):
    src_lang, tgt_lang = lang_pair.split("-")
    folder = os.path.join(DATA_DIR, lang_pair)
    src_path = os.path.join(folder, f"train.{src_lang}")
    tgt_path = os.path.join(folder, f"train.{tgt_lang}")

    if os.path.exists(src_path) and os.path.exists(tgt_path):
        clean_file(src_path)
        clean_file(tgt_path)
        print(f"✅ Auto-cleaned dataset for {lang_pair}")
    else:
        print(f"⚠️ Missing train files for {lang_pair}, skipping clean-up.")

# ============================================================
# 📁 LOAD DATA
# ============================================================
def get_data_files(lang_pair):
    src_lang, tgt_lang = lang_pair.split("-")
    folder = os.path.join(DATA_DIR, lang_pair)
    src_path = os.path.join(folder, f"train.{src_lang}")
    tgt_path = os.path.join(folder, f"train.{tgt_lang}")

    if not (os.path.exists(src_path) and os.path.exists(tgt_path)):
        print(f"⚠️ Missing files for {lang_pair}. Skipping.")
        return None, None

    with open(src_path, "r", encoding="utf-8") as f:
        src_texts = [clean_text(line) for line in f if line.strip()]
    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_texts = [clean_text(line) for line in f if line.strip()]

    n = min(500, len(src_texts), len(tgt_texts))
    print(f"📚 Loaded {n} samples for {lang_pair}")
    return src_texts[:n], tgt_texts[:n]

# ============================================================
# 🧠 TRANSLATION (patched)
# ============================================================
def translate_batch(model, tokenizer, sentences, tgt_lang):
    preds = []
    forced_id = None

    if hasattr(tokenizer, "get_lang_id"):
        try:
            forced_id = tokenizer.get_lang_id(tgt_lang)
        except Exception:
            forced_id = None

    # Ensure correct language context
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = "en"
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = tgt_lang

    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

        with torch.no_grad():
            if forced_id is not None:
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_id,
                    max_new_tokens=MAX_NEW_TOKENS
                )
            else:
                outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds.extend(decoded)
    return preds

# ============================================================
# 🧮 EVALUATION (with diagnostics)
# ============================================================
def evaluate_model(lang_pair):
    src_texts, tgt_texts = get_data_files(lang_pair)
    if not src_texts or not tgt_texts:
        return None

    model_path = os.path.join(MODEL_DIR, lang_pair)
    model, tokenizer = None, None

    if os.path.isdir(model_path):
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"🔗 Using local fine-tuned model: {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load local model for {lang_pair}: {e}")

    if model is None or tokenizer is None:
        print("ℹ️ Falling back to facebook/m2m100_418M")
        model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(DEVICE)
        tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    tgt_lang = lang_pair.split("-")[1]
    preds = translate_batch(model, tokenizer, src_texts, tgt_lang)

    # 🔍 Diagnostic: show sample predictions
    print(f"\n🔍 Sample predictions for {lang_pair}:")
    for i in range(min(5, len(preds))):
        print(f"SRC : {src_texts[i]}")
        print(f"PRED: {preds[i]}")
        print(f"REF : {tgt_texts[i]}")
        print("-" * 50)

    bleu = evaluate.load("sacrebleu" * 50)
    chrf = evaluate.load("chrf")
    ter = evaluate.load("ter")

    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in tgt_texts])
    chrf_res = chrf.compute(predictions=preds, references=[[r] for r in tgt_texts])
    ter_res = ter.compute(predictions=preds, references=[[r] for r in tgt_texts])

    return {
        "lang_pair": lang_pair,
        "BLEU": round(bleu_res["score"], 6),
        "CHRF": round(chrf_res["score"], 6),
        "TER": round(ter_res["score"], 6),
        "Samples": len(src_texts)
    }

# ============================================================
# 📊 PLOTTING
# ============================================================
def plot_metrics(df):
    plt.figure(figsize=(12, 6))
    x = range(len(df))
    plt.bar(x, df["BLEU"], width=0.25, label="BLEU", align="center")
    plt.bar([i + 0.25 for i in x], df["CHRF"], width=0.25, label="CHRF", align="center")
    plt.bar([i + 0.5 for i in x], df["TER"], width=0.25, label="TER", align="center")
    plt.xticks([i + 0.25 for i in x], df["lang_pair"], rotation=45)
    plt.ylabel("Score")
    plt.title("Translation Evaluation Metrics")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "bleu_chart.png")
    plt.savefig(out_path, dpi=300)
    print(f"📊 Chart saved to: {out_path}")

# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    summary = []
    for lang_pair in LANG_PAIRS:
        print(f"\n🌍 Evaluating {lang_pair} model...")
        auto_clean_dataset(lang_pair)
        result = evaluate_model(lang_pair)
        if result:
            summary.append(result)
            print(f"✅ {lang_pair}: BLEU={result['BLEU']}, CHRF={result['CHRF']}, TER={result['TER']} | n={result['Samples']}")
        else:
            print(f"⚠️ Skipped {lang_pair}")

    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(os.path.join(OUTPUT_DIR, "bleu_summary.csv"), index=False, encoding="utf-8")
        print("\n📊 Evaluation Summary:")
        print(df)
        plot_metrics(df)
        print(f"✅ Results saved to {OUTPUT_DIR}")
    else:
        print("❌ No evaluations completed.")

# ============================================================
# 🏁 ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
