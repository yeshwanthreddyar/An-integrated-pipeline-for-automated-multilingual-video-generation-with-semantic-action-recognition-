# mt_eval_pipeline.py
"""
End-to-end MT evaluation pipeline for Samanantar fine-tuned models.

Assumes folder structure:
    ./final_data/
        en-hi/
            train.en
            train.hi
        en-ta/
            train.en
            train.ta
        ...

    ./mass_fine_tuned_models/
        en-hi/
            config.json, tokenizer.json, pytorch_model.bin, ...
        en-ta/
            ...

Creates for each lang pair:
    ./eval_data/en-hi/test.src   (English)
    ./eval_data/en-hi/test.ref   (reference target)
    ./eval_data/en-hi/test.hyp   (model predictions)

Then computes:
    - BLEU, CHRF, TER  (sacrebleu)
    - METEOR           (evaluate)
    - BERTScore        (evaluate)
    - COMET            (evaluate)

And writes results to:
    text_metrics_summary.csv
"""

import os
from typing import List, Tuple, Dict

import sacrebleu
import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import csv

# ---------------------------
# 🔧 CONFIGURATION
# ---------------------------
FINAL_DATA_DIR = "./final_data"
MODEL_ROOT_DIR = "./mass_fine_tuned_models"
EVAL_ROOT_DIR = "./eval_data"
SUMMARY_CSV = "text_metrics_summary.csv"

# number of samples per language to evaluate
N_SAMPLES = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# 📂 UTILS
# ---------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def write_lines(path: str, lines: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")


def discover_lang_pairs(final_data_dir: str) -> List[str]:
    """
    Discover language pairs from final_data folder:
    expects subfolders name like 'en-hi', 'en-ta', etc.
    with files train.en and train.<tgt>
    """
    pairs = []
    for name in os.listdir(final_data_dir):
        folder = os.path.join(final_data_dir, name)
        if not os.path.isdir(folder):
            continue
        if "-" not in name:
            continue
        src, tgt = name.split("-", 1)
        if src != "en":
            # in your project everything is en-XX
            continue
        en_path = os.path.join(folder, "train.en")
        tgt_path = os.path.join(folder, f"train.{tgt}")
        if os.path.exists(en_path) and os.path.exists(tgt_path):
            pairs.append(name)
    return sorted(pairs)


# ---------------------------
# 🧪 CREATE TEST SPLIT
# ---------------------------
def create_test_files_for_pair(lang_pair: str) -> Tuple[str, str]:
    """
    For a given lang pair like 'en-hi', create:
        ./eval_data/en-hi/test.src
        ./eval_data/en-hi/test.ref
    sampled from train.en + train.<tgt>
    """
    src, tgt = lang_pair.split("-", 1)
    data_dir = os.path.join(FINAL_DATA_DIR, lang_pair)
    eval_dir = os.path.join(EVAL_ROOT_DIR, lang_pair)
    ensure_dir(eval_dir)

    train_en = os.path.join(data_dir, "train.en")
    train_tgt = os.path.join(data_dir, f"train.{tgt}")

    src_lines = safe_read_lines(train_en)
    tgt_lines = safe_read_lines(train_tgt)

    # align lengths
    n = min(len(src_lines), len(tgt_lines))
    src_lines = src_lines[:n]
    tgt_lines = tgt_lines[:n]

    # take first N_SAMPLES lines (or all if fewer)
    n_test = min(N_SAMPLES, n)
    src_test = src_lines[:n_test]
    tgt_test = tgt_lines[:n_test]

    test_src_path = os.path.join(eval_dir, "test.src")
    test_ref_path = os.path.join(eval_dir, "test.ref")

    # only create if not exists (so you can keep them fixed)
    if not os.path.exists(test_src_path) or not os.path.exists(test_ref_path):
        write_lines(test_src_path, src_test)
        write_lines(test_ref_path, tgt_test)
        print(f"🧪 Created test set for {lang_pair}: {n_test} samples")
    else:
        print(f"🧪 Reusing existing test files for {lang_pair}")

    return test_src_path, test_ref_path


# ---------------------------
# 🔮 TRANSLATE WITH MODEL
# ---------------------------
def generate_hypotheses_for_pair(lang_pair: str, batch_size: int = 8, max_new_tokens: int = 128) -> str:
    """
    Use local fine-tuned model in mass_fine_tuned_models/en-xx
    to translate test.src → test.hyp.
    """
    src, tgt = lang_pair.split("-", 1)
    eval_dir = os.path.join(EVAL_ROOT_DIR, lang_pair)
    ensure_dir(eval_dir)

    test_src_path = os.path.join(eval_dir, "test.src")
    test_hyp_path = os.path.join(eval_dir, "test.hyp")

    if os.path.exists(test_hyp_path):
        print(f"🔁 Reusing existing predictions for {lang_pair} (test.hyp).")
        return test_hyp_path

    src_lines = safe_read_lines(test_src_path)

    model_dir = os.path.join(MODEL_ROOT_DIR, lang_pair)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"❌ No fine-tuned model folder found for {lang_pair}: {model_dir}")

    print(f"🔗 Loading model & tokenizer from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    preds: List[str] = []
    print(f"⚙️ Generating translations for {lang_pair} ({len(src_lines)} sentences)...")

    with torch.no_grad():
        for i in tqdm(range(0, len(src_lines), batch_size)):
            batch = src_lines[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True
            )
            batch_preds = tokenizer.batch_decode(out, skip_special_tokens=True)
            preds.extend(batch_preds)

    write_lines(test_hyp_path, preds)
    print(f"✅ Saved predictions to {test_hyp_path}")
    return test_hyp_path


# ---------------------------
# 📊 METRIC HELPERS
# ---------------------------
def compute_classic_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs])
    ter = sacrebleu.corpus_ter(preds, [refs])
    return {
        "BLEU": bleu.score,
        "CHRF": chrf.score,
        "TER": ter.score,
    }


def compute_meteor_metric(preds: List[str], refs: List[str]) -> float:
    meteor = evaluate.load("meteor")
    result = meteor.compute(predictions=preds, references=refs)
    return float(result["meteor"] * 100.0)


def compute_bertscore_metric(preds: List[str], refs: List[str], tgt_lang: str) -> Dict[str, float]:
    """
    BERTScore: we pass target language code if known,
    else default to 'en' (model is multilingual anyway).
    """
    bertscore = evaluate.load("bertscore")
    lang_code = tgt_lang if tgt_lang in {
        "en", "hi", "ta", "te", "bn", "gu", "kn", "ml", "mr", "or", "pa", "as"
    } else "en"

    result = bertscore.compute(
        predictions=preds,
        references=refs,
        lang=lang_code,
        model_type="microsoft/mdeberta-v3-base"
    )
    p = sum(result["precision"]) / len(result["precision"])
    r = sum(result["recall"]) / len(result["recall"])
    f1 = sum(result["f1"]) / len(result["f1"])

    return {
        "BERTScore_P": p * 100.0,
        "BERTScore_R": r * 100.0,
        "BERTScore_F1": f1 * 100.0,
    }


def compute_comet_metric(src: List[str], preds: List[str], refs: List[str]) -> float:
    comet = evaluate.load("comet")
    model_name = "Unbabel/wmt22-comet-da"
    print(f"🧠 Loading COMET model: {model_name} (may be slow on first run)...")

    result = comet.compute(
        predictions=preds,
        references=refs,
        sources=src,
        model=model_name,
        progress_bar=True
    )
    return float(result["mean_score"])


# ---------------------------
# 🧮 EVALUATE ONE LANG PAIR
# ---------------------------
def evaluate_pair(lang_pair: str) -> Dict[str, float]:
    src, tgt = lang_pair.split("-", 1)
    print(f"\n🌍 Evaluating {lang_pair} ...")

    # 1) Ensure test.src / test.ref exist
    test_src_path, test_ref_path = create_test_files_for_pair(lang_pair)

    # 2) Generate test.hyp via fine-tuned model
    test_hyp_path = generate_hypotheses_for_pair(lang_pair)

    # 3) Load all 3
    src_lines = safe_read_lines(test_src_path)
    ref_lines = safe_read_lines(test_ref_path)
    hyp_lines = safe_read_lines(test_hyp_path)

    n = min(len(src_lines), len(ref_lines), len(hyp_lines))
    src_lines = src_lines[:n]
    ref_lines = ref_lines[:n]
    hyp_lines = hyp_lines[:n]

    print(f"📚 Using {n} aligned samples for metrics.")

    metrics: Dict[str, float] = {}

    # Classic MT metrics
    classic = compute_classic_metrics(hyp_lines, ref_lines)
    metrics.update(classic)
    print(f"  BLEU : {classic['BLEU']:.2f}")
    print(f"  CHRF : {classic['CHRF']:.2f}")
    print(f"  TER  : {classic['TER']:.2f}")

    # METEOR
    meteor_score = compute_meteor_metric(hyp_lines, ref_lines)
    metrics["METEOR"] = meteor_score
    print(f"  METEOR : {meteor_score:.2f}")

    # BERTScore
    bs = compute_bertscore_metric(hyp_lines, ref_lines, tgt_lang=tgt)
    metrics.update(bs)
    print(f"  BERTScore-F1 : {bs['BERTScore_F1']:.2f}")

    # COMET
    comet_score = compute_comet_metric(src_lines, hyp_lines, ref_lines)
    metrics["COMET"] = comet_score
    print(f"  COMET  : {comet_score:.4f}")

    return metrics


# ---------------------------
# 🚀 MAIN
# ---------------------------
def main():
    ensure_dir(EVAL_ROOT_DIR)

    lang_pairs = discover_lang_pairs(FINAL_DATA_DIR)
    if not lang_pairs:
        print("❌ No language pairs found in final_data/. Check your folder structure.")
        return

    print("🌐 Found language pairs:", ", ".join(lang_pairs))

    all_results: List[Dict[str, float]] = []

    for lp in lang_pairs:
        try:
            metrics = evaluate_pair(lp)
            row = {"lang_pair": lp}
            row.update(metrics)
            all_results.append(row)
        except Exception as e:
            print(f"⚠️ Skipping {lp} due to error: {e}")

    if not all_results:
        print("❌ No results to save.")
        return

    # Save summary CSV
    fieldnames = [
        "lang_pair",
        "BLEU", "CHRF", "TER",
        "METEOR",
        "BERTScore_P", "BERTScore_R", "BERTScore_F1",
        "COMET",
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"\n✅ Saved summary metrics to {SUMMARY_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
