# ============================================================
# config.py — Mac M4 Air | Anti-overfitting version
# ============================================================

import torch

class Config:
    # ── Dataset ──────────────────────────────────────────────
    UCF101_ROOT   = "/Users/silverleaf/datasets/UCF-101/UCF-101"
    SPLITS_ROOT   = "/Users/silverleaf/datasets/ucf101_splits/ucfTrainTestlist"
    SPLIT_NUMBER  = 1
    NUM_CLASSES   = 101

    # ── Video Sampling ───────────────────────────────────────
    NUM_FRAMES    = 8
    FRAME_SIZE    = 224

    # ── Augmentation ─────────────────────────────────────────
    USE_AUGMENTATION   = True
    RANDOM_CROP_SCALE  = (0.6, 1.0)   # slightly more aggressive crop
    COLOR_JITTER       = 0.3           # slightly more jitter
    USE_RANDAUGMENT    = True
    RANDAUGMENT_N      = 2
    RANDAUGMENT_M      = 9

    # ── Mixup OFF ─────────────────────────────────────────────
    USE_MIXUP    = False
    MIXUP_ALPHA  = 0.0

    # ── Training ─────────────────────────────────────────────
    BATCH_SIZE        = 8
    NUM_EPOCHS        = 50
    LEARNING_RATE     = 5e-5           # ↓ was 1e-4 — lower LR = slower memorization
    BACKBONE_LR_MULT  = 0.1
    WEIGHT_DECAY      = 1e-1           # ↑ was 5e-2 — stronger L2 regularization
    WARMUP_EPOCHS     = 3
    NUM_WORKERS       = 2

    # ── Early Stopping ────────────────────────────────────────
    EARLY_STOP_METRIC    = "val_acc"
    EARLY_STOP_PATIENCE  = 5           # ↓ was 7 — stop faster when val plateaus
    EARLY_STOP_MIN_DELTA = 1e-3

    # ── Model ────────────────────────────────────────────────
    VIT_MODEL              = "vit_small_patch16_224"
    DROPOUT                = 0.5       # ↑ was 0.3 — more dropout in classifier
    DROP_PATH_RATE         = 0.2       # ↑ was 0.1 — more stochastic depth
    USE_CROSS_ATTN         = True
    FREEZE_BACKBONE_BLOCKS = 9         # ↑ was 6 — freeze 9/12 blocks, train only last 3

    # ── Device ───────────────────────────────────────────────
    DEVICE = (
        "mps"   if torch.backends.mps.is_available() else
        "cuda"  if torch.cuda.is_available()          else
        "cpu"
    )

    USE_MULTI_GPU = False

    # ── Paths ────────────────────────────────────────────────
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR        = "./logs"
    RESULTS_DIR    = "./results"

    # ── Reproducibility ──────────────────────────────────────
    SEED = 42