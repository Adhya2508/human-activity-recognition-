# ============================================================
# evaluate.py — Run 0 evaluation
# Run: python3 evaluate.py
# Generates: confusion matrix, ROC, PR curve, training curves
# ============================================================

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from config import Config
from model import MGViT

os.makedirs(Config.RESULTS_DIR, exist_ok=True)

RUN_ID = 1   # ← change to 1 or 2 for other runs


# ── Load checkpoint ───────────────────────────────────────────
def load_checkpoint(run_id=1):
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"best_run{run_id}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"✓ Loaded checkpoint: Run {run_id}")
    print(f"  Best epoch  : {ckpt['epoch']}")
    print(f"  Val Acc     : {ckpt['val_acc']*100:.2f}%")
    print(f"  Val Loss    : {ckpt['val_loss']:.4f}")
    return ckpt


# ── Plot training curves ──────────────────────────────────────
def plot_training_curves(run_id=1):
    log_path = os.path.join(Config.LOG_DIR, f"history_run{run_id}.json")
    if not os.path.exists(log_path):
        print(f"No history file found for run {run_id}, skipping curves.")
        return
    with open(log_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, h["train_loss"], label="Train", color="steelblue", lw=2)
    ax1.plot(epochs, h["val_loss"],   label="Val",   color="orange",    lw=2)
    ax1.set_title("Loss Curve"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a*100 for a in h["train_acc"]], label="Train", color="steelblue", lw=2)
    ax2.plot(epochs, [a*100 for a in h["val_acc"]],   label="Val",   color="orange",    lw=2)
    ax2.set_title("Accuracy Curve"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle(f"MG-ViT Training — Run {run_id}", fontsize=13)
    plt.tight_layout()
    out = os.path.join(Config.RESULTS_DIR, f"training_curves_run{run_id}.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"✓ Training curves saved: {out}")


# ── Confusion matrix ──────────────────────────────────────────
def plot_confusion_matrix(labels, preds, idx_to_class, run_id=1, top_n=20):
    unique, counts = np.unique(labels, return_counts=True)
    top_classes    = unique[np.argsort(-counts)][:top_n]
    mask           = np.isin(labels, top_classes)
    fl, fp         = np.array(labels)[mask], np.array(preds)[mask]
    class_names    = [idx_to_class.get(int(i), str(i)) for i in top_classes]
    remap          = {old: new for new, old in enumerate(top_classes)}
    fl = np.array([remap[l] for l in fl])
    fp = np.array([remap.get(p, top_n) for p in fp])
    valid = fp < top_n
    fl, fp = fl[valid], fp[valid]

    cm      = confusion_matrix(fl, fp, labels=list(range(top_n)))
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(top_n)); ax.set_yticks(range(top_n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — Top {top_n} Classes (Run {run_id})")
    plt.tight_layout()
    out = os.path.join(Config.RESULTS_DIR, f"confusion_matrix_run{run_id}.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"✓ Confusion matrix saved: {out}")


# ── ROC curve ─────────────────────────────────────────────────
def plot_roc(labels, probs, run_id=1, n_classes=101):
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))
    probs_arr  = np.array(probs)

    # Macro AUC
    macro_auc = roc_auc_score(labels_bin, probs_arr, average="macro")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, color in zip(range(10), colors):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs_arr[:, i])
        auc_i = roc_auc_score(labels_bin[:, i], probs_arr[:, i])
        ax.plot(fpr, tpr, color=color, lw=1.5, alpha=0.8,
                label=f"Class {i} (AUC={auc_i:.2f})")
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — Macro AUC = {macro_auc:.4f} (Run {run_id})")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    out = os.path.join(Config.RESULTS_DIR, f"roc_curve_run{run_id}.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"✓ ROC curve saved: {out} | Macro AUC: {macro_auc:.4f}")
    return macro_auc


# ── PR curve ──────────────────────────────────────────────────
def plot_pr(labels, probs, run_id=1, n_classes=101):
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))
    probs_arr  = np.array(probs)
    macro_map  = average_precision_score(labels_bin, probs_arr, average="macro")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, color in zip(range(10), colors):
        prec, rec, _ = precision_recall_curve(labels_bin[:, i], probs_arr[:, i])
        ap = average_precision_score(labels_bin[:, i], probs_arr[:, i])
        ax.plot(rec, prec, color=color, lw=1.5, alpha=0.8,
                label=f"Class {i} (AP={ap:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR Curves — mAP = {macro_map:.4f} (Run {run_id})")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    out = os.path.join(Config.RESULTS_DIR, f"pr_curve_run{run_id}.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"✓ PR curve saved: {out} | mAP: {macro_map:.4f}")
    return macro_map


# ── Get softmax probs from saved checkpoint preds ─────────────
def get_probs_from_checkpoint(ckpt):
    """
    Checkpoint has preds (argmax) not probs.
    We load model and re-run val to get actual softmax probs.
    """
    print("\nReloading model to get softmax probabilities...")
    from dataset import get_dataloaders

    model = MGViT()
    model.load_state_dict(ckpt["model_state"])
    model = model.to(Config.DEVICE)
    model.eval()

    _, val_loader, _ = get_dataloaders(Config.SPLIT_NUMBER)

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for rgb, motion, labels in val_loader:
            rgb    = rgb.to(Config.DEVICE)
            motion = motion.to(Config.DEVICE)
            logits = model(rgb, motion)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    print(f"✓ Got predictions for {len(all_labels)} videos")
    return all_labels, all_preds, all_probs


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"Evaluating Run {RUN_ID}")
    print(f"{'='*50}\n")

    # Load checkpoint
    ckpt         = load_checkpoint(RUN_ID)
    idx_to_class = ckpt["idx_to_class"]

    # Training curves (from saved history)
    plot_training_curves(RUN_ID)

    # Get full softmax probs by re-running val set
    labels, preds, probs = get_probs_from_checkpoint(ckpt)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=4, zero_division=0))

    # All plots
    plot_confusion_matrix(labels, preds, idx_to_class, run_id=RUN_ID)
    macro_auc = plot_roc(labels, probs, run_id=RUN_ID)
    macro_map = plot_pr(labels,  probs, run_id=RUN_ID)

    # Summary
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels) * 100
    print(f"\n{'='*50}")
    print(f"Run {RUN_ID} Summary:")
    print(f"  Val Accuracy : {acc:.2f}%")
    print(f"  Macro AUC    : {macro_auc:.4f}")
    print(f"  mAP          : {macro_map:.4f}")
    print(f"  Best Epoch   : {ckpt['epoch']}")
    print(f"  All plots saved to: {Config.RESULTS_DIR}/")
    print(f"{'='*50}")