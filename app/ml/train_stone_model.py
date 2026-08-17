"""One-time, offline training script for the kidney-stone CT screening
model. NOT imported by the Flask app — same relationship train_models.py
has to the rest of the app. Run manually:

    python -m app.ml.train_stone_model

Needs torch/torchvision installed (training-only deps, see
requirements-train.txt — never added to the deployed requirements.txt or
requirements-dev.txt). The deployed app only needs onnxruntime + Pillow to
serve the exported model (see stone_predictor.py) — this keeps
torch/torchvision, which are large, out of the production build entirely.

Dataset: Kaggle "CT KIDNEY DATASET: Normal-Cyst-Tumor-Stone"
(nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone), downloaded via
kagglehub — no Kaggle API credentials needed for this public dataset.
Binarized: Stone -> 1, {Normal, Cyst, Tumor} -> 0 (the app only needs to
answer "is there a stone", not classify all four original categories).
"""
import json
import os
import random

import kagglehub
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 42
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 64
# This machine is a 4-core CPU box with no GPU — num_workers>0 overlaps
# JPEG decode/resize with the forward/backward pass instead of doing them
# serially, which is where most of the wall-clock time goes here (the
# actual matmuls for a frozen-backbone linear probe are cheap; a first run
# with num_workers=0 burned 60+ CPU-minutes without finishing epoch 1).
NUM_WORKERS = 3
# Frozen-backbone transfer learning on a fairly separable task like this
# converges fast — 3 head epochs + 2 fine-tune epochs is enough to check,
# and keeps total wall-clock time reasonable on CPU-only hardware.
HEAD_EPOCHS = 3
FINE_TUNE_EPOCHS = 2
FINE_TUNE_LR = 1e-4
HEAD_LR = 1e-3

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
ONNX_PATH = os.path.join(MODEL_DIR, "kidney_stone.onnx")
META_PATH = os.path.join(MODEL_DIR, "kidney_stone_meta.json")
# Raw torch weights, saved right after training/eval and before the ONNX
# export step — export is a separate concern (dependency versions, opset
# quirks) from training, and a full CPU training run is expensive. If
# export ever fails again, re-run just export_only() against this
# checkpoint instead of retraining from scratch.
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "kidney_stone_checkpoint.pt")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# kagglehub.dataset_download() still hits Kaggle's API on every call to
# check for a newer version, even when the dataset is already cached
# locally — that network round-trip kept failing (WinError 10054,
# connection reset) and crashing runs that otherwise had nothing left to
# download. Once it's been fetched once, go straight to the known cache
# path and skip the network entirely.
_CACHED_ROOT = os.path.join(
    os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
    "nazmul0087", "ct-kidney-dataset-normal-cyst-tumor-and-stone",
    "versions", "1",
    "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone", "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
)


def _dataset_root():
    if os.path.isdir(_CACHED_ROOT):
        return _CACHED_ROOT
    path = kagglehub.dataset_download("nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone")
    # kagglehub unpacks into a nested folder that repeats the dataset name twice.
    nested = os.path.join(path, "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone", "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")
    return nested if os.path.isdir(nested) else path


def _collect_samples(root):
    samples = []  # (filepath, binary_label)
    for class_name in os.listdir(root):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        label = 1 if class_name.lower() == "stone" else 0
        for fname in os.listdir(class_dir):
            samples.append((os.path.join(class_dir, fname), label))
    return samples


class CTStoneDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            if is_train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.detach().numpy().tolist())
            all_preds.extend(logits.argmax(dim=1).detach().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_labels, all_preds, all_probs


def best_threshold_for_f1(labels, probs):
    labels = np.array(labels)
    probs = np.array(probs)
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.1, 0.91, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 2)


def main():
    print("Locating dataset...")
    root = _dataset_root()
    samples = _collect_samples(root)
    labels = [s[1] for s in samples]
    print(f"Total images: {len(samples)}, stone: {sum(labels)}, not-stone: {len(labels) - sum(labels)}")

    train_samples, temp_samples = train_test_split(
        samples, test_size=0.3, stratify=labels, random_state=SEED,
    )
    temp_labels = [s[1] for s in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.5, stratify=temp_labels, random_state=SEED,
    )
    print(f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_loader = DataLoader(CTStoneDataset(train_samples, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)
    val_loader = DataLoader(CTStoneDataset(val_samples, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)
    test_loader = DataLoader(CTStoneDataset(test_samples, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)

    train_labels_arr = np.array([s[1] for s in train_samples])
    class_counts = np.bincount(train_labels_arr, minlength=2)
    class_weights = torch.tensor([1.0 / c if c > 0 else 0.0 for c in class_counts], dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * 2
    print("class weights (no-stone, stone):", class_weights.tolist())

    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\n-- Phase 1: training classifier head (backbone frozen) --")
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=HEAD_LR)
    for epoch in range(HEAD_EPOCHS):
        train_loss, *_ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_labels, val_preds, val_probs = run_epoch(model, val_loader, criterion)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        print(f"epoch {epoch+1}/{HEAD_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

    print("\n-- Phase 2: fine-tuning last backbone block --")
    for p in model.features[-3:].parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=FINE_TUNE_LR,
    )
    for epoch in range(FINE_TUNE_EPOCHS):
        train_loss, *_ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_labels, val_preds, val_probs = run_epoch(model, val_loader, criterion)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        print(f"epoch {epoch+1}/{FINE_TUNE_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

    print("\n-- Tuning decision threshold on validation set --")
    _, val_labels, _, val_probs = run_epoch(model, val_loader, criterion)
    threshold = best_threshold_for_f1(val_labels, val_probs)
    print("chosen threshold:", threshold)

    print("\n-- Final evaluation on held-out test set --")
    _, test_labels, _, test_probs = run_epoch(model, test_loader, criterion)
    test_probs_arr = np.array(test_probs)
    test_preds = (test_probs_arr >= threshold).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(test_labels, test_preds)), 4),
        "precision": round(float(precision_score(test_labels, test_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(test_labels, test_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(test_labels, test_preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(test_labels, test_probs_arr)), 4),
        "chosen_model": "mobilenet_v2_transfer",
    }
    print("test metrics:", metrics)
    print("confusion matrix (rows=true, cols=pred, [no_stone, stone]):")
    print(confusion_matrix(test_labels, test_preds))

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("\n-- Saving checkpoint (safety net before export) --")
    torch.save(
        {"state_dict": model.state_dict(), "threshold": threshold, "metrics": metrics},
        CHECKPOINT_PATH,
    )
    print("saved:", CHECKPOINT_PATH)

    export_onnx(model, threshold, metrics)


def export_onnx(model, threshold, metrics):
    """Exports an already-trained model to ONNX + writes the metadata JSON.
    Split out from main() so a checkpoint-only run (see export_only) can
    retry export without repeating a full training run."""
    print("\n-- Exporting to ONNX --")
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print("saved:", ONNX_PATH, f"({os.path.getsize(ONNX_PATH) / 1e6:.1f} MB)")

    meta = {
        "input_size": IMG_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "threshold": threshold,
        "label_map": {"0": "no_stone", "1": "stone"},
        "metrics": metrics,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print("saved:", META_PATH)


def export_only():
    """Re-run just the ONNX export from a previously saved checkpoint —
    use this if export fails after a training run already completed
    (e.g. an exporter dependency/opset issue), instead of retraining."""
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["state_dict"])
    export_onnx(model, ckpt["threshold"], ckpt["metrics"])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--export-only":
        export_only()
    else:
        main()
