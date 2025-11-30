
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import time
from datetime import datetime
import signal
import sys

# --- Graceful Exit Handler ---
def handle_exit(sig, frame):
    print("\n⚠️ Graceful shutdown triggered. Saving progress and exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# --- Custom Dataset ---
class TranslationDataset(Dataset):
    def __init__(self, sources, targets, tokenizer, max_length=256):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        source = str(self.sources[idx])
        target = str(self.targets[idx])
        
        source_enc = self.tokenizer(
            source, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        target_enc = self.tokenizer(
            target, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        labels = target_enc['input_ids'].flatten()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_enc['input_ids'].flatten(),
            'attention_mask': source_enc['attention_mask'].flatten(),
            'labels': labels
        }

# --- Configuration ---
SPLITS_DIR = "data_splits"
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

SAMPLE_SIZE = 5000
VAL_SAMPLE_SIZE = 1000
BATCH_SIZE = 8
NUM_EPOCHS = 1
LEARNING_RATE = 3e-5

# --- Load Data ---
def load_data(lang_pair, split_type):
    file_path = os.path.join(SPLITS_DIR, lang_pair, f"{split_type}.csv")
    if not os.path.exists(file_path):
        print(f"⚠ Warning: {file_path} not found.")
        return None, None
    
    try:
        df = pd.read_csv(
            file_path,
            engine="python",
            on_bad_lines="skip",
            encoding_errors="ignore"
        )
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None, None
    
    source_lang, target_lang = lang_pair.split('-')
    if source_lang not in df.columns or target_lang not in df.columns:
        print(f"⚠ Missing required columns in {file_path}.")
        return None, None

    return df[source_lang].astype(str).tolist(), df[target_lang].astype(str).tolist()

# --- Fine-tuning for Single Language ---
def fine_tune_language(target_lang, model_map):
    lang_pair = f"en-{target_lang}"
    model_name = model_map[target_lang]

    print(f"\n{'='*60}")
    print(f"🚀 Fine-tuning {lang_pair}")
    print(f"📦 Model: {model_name}")
    print(f"{'='*60}")

    train_sources, train_targets = load_data(lang_pair, "train")
    val_sources, val_targets = load_data(lang_pair, "validation")

    if not train_sources or not val_sources:
        print(f"❌ Skipping {lang_pair}: missing data.")
        return None

    train_sources, train_targets = train_sources[:SAMPLE_SIZE], train_targets[:SAMPLE_SIZE]
    val_sources, val_targets = val_sources[:VAL_SAMPLE_SIZE], val_targets[:VAL_SAMPLE_SIZE]

    print(f"📊 Training samples: {len(train_sources)} | Validation: {len(val_sources)}")

    # Load model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"🔧 Using device: {device}")
    except Exception as e:
        print(f"❌ Error loading model for {lang_pair}: {e}")
        return None

    train_loader = DataLoader(
        TranslationDataset(train_sources, train_targets, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        TranslationDataset(val_sources, val_targets, tokenizer),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    losses = []
    for epoch in range(NUM_EPOCHS):
        print(f"\n📚 Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Training {lang_pair}"):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"📉 Avg Training Loss: {avg_loss:.4f}")
        losses.append(avg_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating {lang_pair}"):
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            val_loss += outputs.loss.item()
    val_loss /= len(val_loader)
    print(f"🧪 Validation Loss: {val_loss:.4f}")

    # Save
    output_dir = f"./mass_fine_tuned_models/{lang_pair}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"💾 Saved model to {output_dir}")

    return {
        'lang_pair': lang_pair,
        'train_loss': losses[-1],
        'val_loss': val_loss,
        'path': output_dir
    }

# --- Main Controller ---
def mass_fine_tune():
    print("🎯 MASS FINE-TUNING START")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔢 Languages: {len(MODEL_MAP)}\n")

    results = []
    for lang in MODEL_MAP.keys():
        try:
            res = fine_tune_language(lang, MODEL_MAP)
            if res:
                results.append(res)
        except Exception as e:
            print(f"💥 Error during {lang}: {e}")
        time.sleep(2)

    if results:
        df = pd.DataFrame(results)
        save_path = f"./mass_fine_tuned_models/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(save_path, index=False)
        print(f"\n✅ Results saved to {save_path}")

if __name__ == "__main__":
    mass_fine_tune()
