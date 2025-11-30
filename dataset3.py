import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

from dataset1 import UCF101Dataset
from dataset2 import VideoClassifier

# -------------------------------------------------------
# ⚙️ CONFIGURATION
# -------------------------------------------------------
DATASET_DIR = "C:/Users/Yeshwanth Reddy A R/Downloads/pibb/UCF-101"
CLASSES_FILE = "C:/Users/Yeshwanth Reddy A R/Downloads/pibb/UCF-101/classInd.txt"
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = os.path.join(CHECKPOINT_DIR, "training_log.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# 🧠 DATASET & DATALOADER
# -------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Resize all frames to consistent size
])

print("📦 Loading dataset...")
train_dataset = UCF101Dataset(DATASET_DIR, CLASSES_FILE, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
num_classes = len(train_dataset.classes)

print(f"✅ Loaded {len(train_dataset)} samples across {num_classes} classes.")

# -------------------------------------------------------
# 🧩 MODEL + LOSS + OPTIMIZER
# -------------------------------------------------------
model = VideoClassifier(num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------------------------------------
# 🔁 CHECKPOINT LOADER
# -------------------------------------------------------
def load_latest_checkpoint(model, optimizer):
    latest_ckpt = None
    latest_epoch = 0

    # Find the latest checkpoint file
    for file in os.listdir(CHECKPOINT_DIR):
        if file.startswith("ucf101_epoch") and file.endswith(".pth"):
            try:
                epoch_num = int(file.split("epoch")[1].split(".pth")[0])
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_ckpt = os.path.join(CHECKPOINT_DIR, file)
            except:
                pass

    if not latest_ckpt:
        print("🚀 No checkpoint found, starting fresh from epoch 1.")
        return 1

    print(f"🔄 Found checkpoint: {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location=DEVICE)

    # ✅ Handle both new-format and old-format checkpoints
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resumed_epoch = checkpoint.get("epoch", 0) + 1
        print(f"✅ Resuming training from epoch {resumed_epoch}")
        return resumed_epoch

    else:
        # ⚙️ Handle older .pth files that only store weights
        print("⚠️ Old checkpoint format detected — upgrading automatically.")
        model.load_state_dict(checkpoint)
        upgraded_path = latest_ckpt.replace(".pth", "_fixed.pth")
        torch.save({
            "epoch": latest_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": 0.0
        }, upgraded_path)
        print(f"💾 Upgraded checkpoint saved as: {upgraded_path}")
        print(f"✅ Resuming training from epoch {latest_epoch + 1}")
        return latest_epoch + 1


start_epoch = load_latest_checkpoint(model, optimizer)
print(f"📍 Training will begin from epoch {start_epoch}")

# -------------------------------------------------------
# 🧾 LOG FILE SETUP
# -------------------------------------------------------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "Accuracy (%)"])

# -------------------------------------------------------
# 🎯 TRAINING LOOP
# -------------------------------------------------------
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    running_loss, total, correct = 0.0, 0, 0
    progress_bar = tqdm(train_loader, desc=f"🧠 Epoch {epoch}/{EPOCHS}", unit="batch")

    for frames, labels in progress_bar:
        frames, labels = frames.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        acc = 100 * correct / total
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.2f}%"})

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"✅ Epoch {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # 💾 SAVE CHECKPOINT
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"ucf101_epoch{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }, ckpt_path)
    print(f"💾 Saved checkpoint: {ckpt_path}")

    # 🧾 CRASH-SAFE CSV LOGGING
    for attempt in range(3):
        try:
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_loss, accuracy])
            break
        except PermissionError:
            print("⚠️ training_log.csv is open — retrying in 3s...")
            time.sleep(3)
    else:
        print("❌ Failed to log metrics after 3 retries. Close training_log.csv manually.")

print("\n🎉 Training complete!")
print(f"📊 Logs saved at: {LOG_FILE}")
print(f"💾 Checkpoints saved in: {CHECKPOINT_DIR}")
