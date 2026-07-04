# ============================================================
# train.py — MG-ViT Training Loop (Mac M4 version)
# ============================================================
# M4 specific:
#   - MPS device (no CUDA, no DataParallel)
#   - pin_memory=False (MPS doesn't support it)
#   - No mixed precision (MPS autocast limited support)
#   - Batch size 8 (fits in 16GB unified memory)
# ============================================================

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from config import Config
from dataset import get_dataloaders
from model import MGViT


# ── Reproducibility ──────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── LR Schedule: linear warmup → cosine decay ────────────────
def make_warmup_cosine_schedule(optimizer, warmup_epochs, total_epochs, min_lr_ratio=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return max(cosine, min_lr_ratio)
    return LambdaLR(optimizer, lr_lambda)


# ── Training one epoch ────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (rgb, motion, labels) in enumerate(loader):
        rgb    = rgb.to(device)
        motion = motion.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(rgb, motion)
        loss   = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {correct/total*100:.1f}%")

    return total_loss / len(loader), correct / total


# ── Validation one epoch ──────────────────────────────────────
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for rgb, motion, labels in loader:
            rgb    = rgb.to(device)
            motion = motion.to(device)
            labels = labels.to(device)

            logits = model(rgb, motion)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels


# ── Main Training Loop ────────────────────────────────────────
def train(seed=Config.SEED, run_id=0):
    set_seed(seed)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"\n{'='*55}")
    print(f"Run {run_id} | Seed {seed} | Device: {device}")
    print(f"Epochs: {Config.NUM_EPOCHS} | Patience: {Config.EARLY_STOP_PATIENCE} "
          f"(watching: {Config.EARLY_STOP_METRIC})")
    print(f"Backbone frozen blocks: {Config.FREEZE_BACKBONE_BLOCKS}/12")
    print(f"Cross-attn: {Config.USE_CROSS_ATTN} | Mixup: {Config.USE_MIXUP}")
    print(f"{'='*55}\n")

    # Data — pin_memory=False for MPS
    train_loader, val_loader, idx_to_class = get_dataloaders(Config.SPLIT_NUMBER)

    # Model (single device, no DataParallel on Mac)
    model = MGViT().to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Layer-wise LR optimizer
    param_groups = model.get_param_groups(Config.LEARNING_RATE, Config.BACKBONE_LR_MULT)
    optimizer    = AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)

    # Scheduler
    scheduler = make_warmup_cosine_schedule(
        optimizer,
        warmup_epochs=Config.WARMUP_EPOCHS,
        total_epochs=Config.NUM_EPOCHS
    )

    # Early stopping
    monitor_minimize = (Config.EARLY_STOP_METRIC == "val_loss")
    best_metric   = float("inf") if monitor_minimize else -float("inf")
    best_val_acc  = 0.0
    patience_ctr  = 0
    stopped_epoch = Config.NUM_EPOCHS
    best_ckpt     = os.path.join(Config.CHECKPOINT_DIR, f"best_run{run_id}.pt")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start = time.time()
        print(f"Epoch {epoch}/{Config.NUM_EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, preds, labels = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed     = time.time() - start
        backbone_lr = optimizer.param_groups[0]['lr']
        new_lr      = optimizer.param_groups[1]['lr']

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")
        print(f"  Time: {elapsed:.1f}s | LR backbone/new: {backbone_lr:.6f}/{new_lr:.6f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        current_metric = val_loss if monitor_minimize else val_acc
        improved = (
            (monitor_minimize     and current_metric < best_metric - Config.EARLY_STOP_MIN_DELTA) or
            (not monitor_minimize and current_metric > best_metric + Config.EARLY_STOP_MIN_DELTA)
        )

        if improved:
            best_metric  = current_metric
            patience_ctr = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_acc":      val_acc,
                "val_loss":     val_loss,
                "preds":        preds,
                "labels":       labels,
                "idx_to_class": idx_to_class,
            }, best_ckpt)
            print(f"  ✓ Best saved! Val Acc: {val_acc*100:.2f}% | Val Loss: {val_loss:.4f}")
        else:
            patience_ctr += 1
            print(f"  No improvement ({Config.EARLY_STOP_METRIC}). "
                  f"Patience: {patience_ctr}/{Config.EARLY_STOP_PATIENCE}")
            if patience_ctr >= Config.EARLY_STOP_PATIENCE:
                stopped_epoch = epoch
                print(f"\n  ⏹ Early stopping at epoch {epoch}")
                print(f"  Best val acc: {best_val_acc*100:.2f}%")
                break

        print()

    # Save history
    history["stopped_epoch"] = stopped_epoch
    history["best_val_acc"]  = best_val_acc
    with open(os.path.join(Config.LOG_DIR, f"history_run{run_id}.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Run {run_id} complete!")
    print(f"  Stopped at epoch : {stopped_epoch}/{Config.NUM_EPOCHS}")
    print(f"  Best Val Acc     : {best_val_acc*100:.2f}%")
    return best_val_acc


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    # Start with 1 run — check numbers, then do all 3 for paper
    seeds   = [42, 123, 7]
    results = []

    for run_id, seed in enumerate(seeds):
        acc = train(seed=seed, run_id=run_id)
        results.append(acc * 100)

    print(f"\n{'='*55}")
    print(f"All {len(results)} runs complete!")
    print(f"  Accuracies : {[f'{r:.2f}%' for r in results]}")
    print(f"  Mean       : {np.mean(results):.2f}%")
    print(f"  Std        : {np.std(results):.2f}%")
    print(f"  Paper line : {np.mean(results):.2f} ± {np.std(results):.2f}%")
    print(f"{'='*55}")