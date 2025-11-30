
# pib_text2video2/translate.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import os
from pathlib import Path

# --- Fixed Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent  # Goes to PIB_Projects
FINETUNED_MODELS_DIR = PROJECT_ROOT / "pibb" / "mass_fine_tuned_models"

# Language mapping
MODEL_MAP = {
    "as": "Helsinki-NLP/opus-mt-en-mul",
    "bn": "Helsinki-NLP/opus-mt-en-mul",
    "gu": "Helsinki-NLP/opus-mt-en-mul",
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "kn": "Helsinki-NLP/opus-mt-en-mul",
    "ml": "Helsinki-NLP/opus-mt-en-mul",
    "mr": "Helsinki-NLP/opus-mt-en-mul",
    "or": "Helsinki-NLP/opus-mt-en-mul",
    "pa": "Helsinki-NLP/opus-mt-en-mul",
    "ta": "suriya7/English-to-Tamil",
    "te": "aryaumesh/english-to-telugu",
}

def translate_sentences(sentences, target_lang='hi'):
    """
    Translates a list of English sentences to a target language using your
    locally fine-tuned model.
    """
    if target_lang not in MODEL_MAP:
        raise ValueError(f"No fine-tuned model available for target language: {target_lang}")

    # Local fine-tuned model path
    local_model_path = FINETUNED_MODELS_DIR / f"en-{target_lang}"
    if not local_model_path.exists():
        raise FileNotFoundError(f"Model directory not found at: {local_model_path}")

    print(f"Loading fine-tuned model for '{target_lang}' from: {local_model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(local_model_path)).to(device)

    translations = []
    for sentence in tqdm(sentences, desc=f"Translating to {target_lang}"):
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
        generated_tokens = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
        translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        translations.append(translation)

    return translations

# Simple test function
def test_translation():
    """Test the translation with your fine-tuned models"""
    test_sentences = [
        "The government launched a new education policy.",
        "Prime Minister announced development schemes for farmers.",
        "India celebrates 75 years of independence."
    ]
    
    print("Testing translation with fine-tuned models...")
    
    for lang in ['hi', 'ta', 'te', 'as', 'bn', 'gu', 'kn', 'ml', 'mr', 'or', 'pa']:
        try:
            translated = translate_sentences(test_sentences, target_lang=lang)
            print(f"\n{lang.upper()} Translations:")
            for i, (orig, trans) in enumerate(zip(test_sentences, translated)):
                print(f"  {i+1}. {orig} -> {trans}")
        except Exception as e:
            print(f"❌ Error with {lang}: {e}")

if __name__ == "__main__":
    test_translation()

