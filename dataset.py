# ============================================================
# dataset.py — UCF-101 video loader
# Handles: frame sampling + motion map generation + augmentation
# ============================================================

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import Config


# ── Helper: Sample T frames uniformly from a video ──────────
def sample_frames(video_path, num_frames=8):
    """
    Opens a video file and returns num_frames evenly spaced frames.
    Returns list of numpy arrays (H, W, 3) in RGB.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = max(total, 1)

    # Pick indices evenly spaced across the video
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # If read fails, just duplicate last good frame
            frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames  # list of T numpy arrays


# ── Helper: Generate motion maps from frames ─────────────────
def generate_motion_maps(frames):
    """
    Takes list of T RGB frames, returns T-1 motion maps.
    Motion map = absolute difference between consecutive frames (grayscale).
    For T=8 frames → 7 motion maps. We pad one to keep T=8.
    """
    motion_maps = []
    for i in range(1, len(frames)):
        f1 = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY).astype(np.float32)
        f2 = cv2.cvtColor(frames[i],   cv2.COLOR_RGB2GRAY).astype(np.float32)
        diff = np.abs(f2 - f1)
        diff = diff / (diff.max() + 1e-8)
        motion_maps.append(diff)

    motion_maps.append(motion_maps[-1])
    return motion_maps  # list of T grayscale numpy arrays


# ── Transforms ───────────────────────────────────────────────
# TRAIN: augmented (this is the main overfitting fix — the original
# code used the exact same deterministic Resize for train and val,
# so the model saw literally the same crop of the same frame every
# single epoch and could just memorize it).
#
# VAL/TEST: deterministic (Resize + CenterCrop only) — we never want
# augmentation randomness in evaluation, it would make val metrics
# noisy and non-comparable across epochs.

def _build_rgb_train_transform():
    ops = [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(
            Config.FRAME_SIZE,
            scale=Config.RANDOM_CROP_SCALE
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=Config.COLOR_JITTER,
            contrast=Config.COLOR_JITTER,
            saturation=Config.COLOR_JITTER,
        ),
    ]
    if Config.USE_RANDAUGMENT:
        ops.append(transforms.RandAugment(
            num_ops=Config.RANDAUGMENT_N,
            magnitude=Config.RANDAUGMENT_M
        ))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        ),
    ]
    return transforms.Compose(ops)


rgb_train_transform = _build_rgb_train_transform()

rgb_eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((int(Config.FRAME_SIZE * 1.14), int(Config.FRAME_SIZE * 1.14))),
    transforms.CenterCrop(Config.FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    ),
])

# Motion maps: keep flip in sync conceptually, but since motion maps are
# derived per-sample independently of the RGB augmentation object here,
# we just resize/center-crop them consistently (no color jitter — they're
# single-channel magnitude maps, jitter would be meaningless / harmful).
motion_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(Config.FRAME_SIZE, scale=Config.RANDOM_CROP_SCALE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

motion_eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((int(Config.FRAME_SIZE * 1.14), int(Config.FRAME_SIZE * 1.14))),
    transforms.CenterCrop(Config.FRAME_SIZE),
    transforms.ToTensor(),
])


# ── UCF-101 Dataset Class ─────────────────────────────────────
class UCF101Dataset(Dataset):
    """
    Loads UCF-101 videos from official train/test split files.
    Returns: rgb_frames, motion_maps, label
    """
    def __init__(self, split="train", split_number=1):
        self.split = split
        self.samples = []
        self.class_to_idx = {}

        split_tag  = "train" if split == "train" else "test"
        split_file = os.path.join(
            Config.SPLITS_ROOT,
            f"{split_tag}list0{split_number}.txt"
        )

        all_classes = sorted(os.listdir(Config.UCF101_ROOT))
        all_classes = [c for c in all_classes if os.path.isdir(os.path.join(Config.UCF101_ROOT, c))]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel_path = parts[0]
                class_name = rel_path.split("/")[0]
                if class_name not in self.class_to_idx:
                    continue
                full_path = os.path.join(Config.UCF101_ROOT, rel_path)
                if os.path.exists(full_path):
                    self.samples.append((full_path, self.class_to_idx[class_name]))

        # Pick transforms based on split + config flag
        use_aug = Config.USE_AUGMENTATION and split == "train"
        self.rgb_transform    = rgb_train_transform    if use_aug else rgb_eval_transform
        self.motion_transform = motion_train_transform if use_aug else motion_eval_transform

        print(f"[Dataset] {split} split: {len(self.samples)} videos, {len(self.class_to_idx)} classes "
              f"(augmentation={'ON' if use_aug else 'OFF'})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        frames = sample_frames(video_path, Config.NUM_FRAMES)
        motion_maps = generate_motion_maps(frames)

        rgb_tensors    = torch.stack([self.rgb_transform(f) for f in frames])        # (T, 3, 224, 224)
        motion_tensors = torch.stack([
            self.motion_transform((m * 255).astype(np.uint8))
            for m in motion_maps
        ])                                                                            # (T, 1, 224, 224)

        return rgb_tensors, motion_tensors, torch.tensor(label, dtype=torch.long)


# ── DataLoader factory ────────────────────────────────────────
def get_dataloaders(split_number=1):
    train_ds = UCF101Dataset(split="train", split_number=split_number)
    val_ds   = UCF101Dataset(split="test",  split_number=split_number)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False   # pin_memory=False for MPS (it doesn't support it)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False
    )
    return train_loader, val_loader, train_ds.idx_to_class