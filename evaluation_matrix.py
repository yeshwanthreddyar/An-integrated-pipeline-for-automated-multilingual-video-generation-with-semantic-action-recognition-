# evaluate_ucf101.py

import os
import csv
import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from dataset1 import UCF101Dataset
from dataset2 import VideoClassifier

# -------------------------------------------------------
# ⚙️ CONFIG
# -------------------------------------------------------
DATASET_DIR = r"C:/Users/Yeshwanth Reddy A R/Downloads/pibb/UCF-101"
CLASSES_FILE = r"C:/Users/Yeshwanth Reddy A R/Downloads/pibb/UCF-101/classInd.txt"
CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 4
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLIT_SEED = 42      # 🔒 fixed seed for reproducible split
TEST_FRACTION = 0.2  # 20% test, 80% train

OVERALL_CSV = "evaluation_overall.csv"
PER_CLASS_CSV = "evaluation_per_class.csv"
CONF_MAT_CSV = "confusion_matrix.csv"


# -------------------------------------------------------
# 🧊 FIND LATEST CHECKPOINT
# -------------------------------------------------------
def get_latest_checkpoint(ckpt_dir=CHECKPOINT_DIR):
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoint directory found at: {ckpt_dir}")

    latest_ckpt = None
    latest_epoch = -1

    for f in os.listdir(ckpt_dir):
        if f.startswith("ucf101_epoch") and f.endswith(".pth"):
            try:
                epoch_num = int(f.split("epoch")[1].split(".pth")[0])
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_ckpt = os.path.join(ckpt_dir, f)
            except Exception:
                pass

    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoints like 'ucf101_epochX.pth' found in {ckpt_dir}")

    return latest_ckpt, latest_epoch


# -------------------------------------------------------
# 📚 BUILD DATASET + TEST SPLIT
# -------------------------------------------------------
def build_test_loader():
    print("📦 Loading full UCF101 dataset...")
    full_dataset = UCF101Dataset(DATASET_DIR, CLASSES_FILE)

    num_samples = len(full_dataset)
    num_classes = len(full_dataset.classes)
    print(f"✅ Dataset size: {num_samples} videos | {num_classes} classes")

    indices = list(range(num_samples))
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(indices)

    test_size = int(TEST_FRACTION * num_samples)
    test_indices = indices[:test_size]

    print(f"🧪 Using {test_size} samples for TEST evaluation (≈{TEST_FRACTION*100:.0f}%)")

    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return test_loader, full_dataset.classes, num_classes


# -------------------------------------------------------
# 🧠 LOAD MODEL
# -------------------------------------------------------
def load_model(num_classes):
    ckpt_path, epoch = get_latest_checkpoint()
    print(f"🧠 Loading model from checkpoint: {ckpt_path} (epoch {epoch})")

    model = VideoClassifier(num_classes=num_classes).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("✅ Loaded checkpoint with optimizer + epoch info format.")
    else:
        model.load_state_dict(ckpt)
        print("⚠️ Loaded old-style checkpoint (weights only).")

    model.eval()
    return model


# -------------------------------------------------------
# 📊 EVALUATION LOOP
# -------------------------------------------------------
def evaluate(model, test_loader):
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for frames, labels in test_loader:
            # frames: [B, T, C, H, W]
            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(frames)              # [B, num_classes]
            _, preds = torch.max(outputs, dim=1) # [B]

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # ---- Overall metrics ----
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    print("\n📊 Overall Evaluation Metrics (TEST SET ONLY)")
    print(f"   Accuracy : {accuracy * 100:.2f}%")
    print(f"   Precision: {precision * 100:.2f}%")
    print(f"   Recall   : {recall * 100:.2f}%")
    print(f"   F1 Score : {f1 * 100:.2f}%")

    # ---- Per-class report ----
    print("\n📄 Classification report (per class):")
    print(classification_report(all_labels, all_preds, digits=4))

    # ---- Confusion matrix ----
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n🔢 Confusion matrix shape: {cm.shape}")

    return accuracy, precision, recall, f1, cm, all_labels, all_preds


# -------------------------------------------------------
# 💾 SAVE METRICS TO CSV
# -------------------------------------------------------
def save_overall_csv(accuracy, precision, recall, f1, out_path=OVERALL_CSV):
    print(f"\n💾 Saving overall metrics to {out_path}")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Accuracy", "Precision", "Recall", "F1"])
        writer.writerow([
            f"{accuracy * 100:.4f}",
            f"{precision * 100:.4f}",
            f"{recall * 100:.4f}",
            f"{f1 * 100:.4f}",
        ])


def save_per_class_csv(all_labels, all_preds, class_names, out_path=PER_CLASS_CSV):
    print(f"💾 Saving per-class metrics to {out_path}")
    num_classes = len(class_names)

    # Per-class metrics (no averaging)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ClassID", "ClassName", "Precision", "Recall", "F1", "Support"])

        for cid in range(num_classes):
            cname = class_names[cid] if cid < len(class_names) else f"class_{cid}"
            writer.writerow([
                cid,
                cname,
                f"{precision[cid] * 100:.4f}",
                f"{recall[cid] * 100:.4f}",
                f"{f1[cid] * 100:.4f}",
                int(support[cid]),
            ])


def save_confusion_matrix_csv(cm, out_path=CONF_MAT_CSV):
    print(f"💾 Saving confusion matrix to {out_path}")
    num_classes = cm.shape[0]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        # header: blank + class indices
        header = [""] + [f"class_{i}" for i in range(num_classes)]
        writer.writerow(header)

        for i in range(num_classes):
            row = [f"class_{i}"] + list(cm[i])
            writer.writerow(row)


# -------------------------------------------------------
# 🚀 MAIN
# -------------------------------------------------------
def main():
    # 1) Build test loader from *full* dataset using 80/20 split
    test_loader, class_names, num_classes = build_test_loader()

    # 2) Load latest checkpoint
    model = load_model(num_classes=num_classes)

    # 3) Run evaluation on TEST set only
    accuracy, precision, recall, f1, cm, all_labels, all_preds = evaluate(model, test_loader)

    # 4) Save CSV files
    save_overall_csv(accuracy, precision, recall, f1)
    save_per_class_csv(all_labels, all_preds, class_names)
    save_confusion_matrix_csv(cm)

    print("\n✅ Evaluation complete.")
    print(f"   Overall metrics → {OVERALL_CSV}")
    print(f"   Per-class metrics → {PER_CLASS_CSV}")
    print(f"   Confusion matrix → {CONF_MAT_CSV}")
    print("\n⚠️ NOTE:")
    print("   For *realistic* accuracy (~50–60%), you must train ONLY on the 80% train split")
    print("   and evaluate ONLY on this 20% test split. If you trained on ALL data,")
    print("   test accuracy can still be unrealistically high due to memorization.\n")


if __name__ == "__main__":
    main()
