# ============================================================
# model.py — MG-ViT Architecture (Mac M4 version)
# ============================================================

import torch
import torch.nn as nn
import timm
from config import Config


# ── Cross-Attention Fusion ────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    RGB features attend to motion features via cross-attention.
    Novel contribution — replaces simple element-wise addition.
    """
    def __init__(self, embed_dim=384, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, rgb_feat, motion_feat):
        attended, _ = self.cross_attn(
            query=rgb_feat,
            key=motion_feat,
            value=motion_feat
        )
        x = self.norm1(rgb_feat + attended)
        x = self.norm2(x + self.ffn(x))
        return x


# ── Temporal Attention ────────────────────────────────────────
class TemporalAttention(nn.Module):
    """
    Learns which frames matter more — replaces mean pooling.
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (x * weights).sum(dim=1)               # (B, D)


# ── Motion Projector ──────────────────────────────────────────
class MotionProjector(nn.Module):
    """
    Grayscale motion maps → same embedding dim as ViT.
    """
    def __init__(self, out_dim=384):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 7 * 7, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.proj(self.conv(x).flatten(1))


# ── Main MG-ViT ───────────────────────────────────────────────
class MGViT(nn.Module):
    def __init__(self):
        super().__init__()

        # ViT backbone
        self.vit = timm.create_model(
            Config.VIT_MODEL,
            pretrained=True,
            num_classes=0,
            drop_path_rate=Config.DROP_PATH_RATE,
        )
        embed_dim = self.vit.embed_dim  # 384 for vit_small

        # Freeze early blocks
        self._freeze_backbone_blocks(Config.FREEZE_BACKBONE_BLOCKS)

        # New trainable modules
        self.motion_proj   = MotionProjector(out_dim=embed_dim)
        self.fusion        = CrossAttentionFusion(embed_dim=embed_dim)
        self.temporal_attn = TemporalAttention(embed_dim=embed_dim)
        self.classifier    = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(embed_dim, Config.NUM_CLASSES)
        )

    def _freeze_backbone_blocks(self, n_freeze):
        for p in self.vit.patch_embed.parameters():
            p.requires_grad = False
        for i, block in enumerate(self.vit.blocks):
            if i < n_freeze:
                for p in block.parameters():
                    p.requires_grad = False
        frozen = sum(1 for p in self.vit.parameters() if not p.requires_grad)
        total  = sum(1 for p in self.vit.parameters())
        print(f"[Model] ViT blocks frozen: {n_freeze}/12 | "
              f"Frozen params: {frozen}/{total}")

    def get_param_groups(self, base_lr, backbone_lr_mult):
        backbone_params = [p for p in self.vit.parameters() if p.requires_grad]
        new_params = (
            list(self.motion_proj.parameters()) +
            list(self.fusion.parameters()) +
            list(self.temporal_attn.parameters()) +
            list(self.classifier.parameters())
        )
        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
            {"params": new_params,      "lr": base_lr},
        ]

    def forward(self, rgb, motion):
        B, T, C, H, W = rgb.shape

        # RGB via ViT
        rgb_feat = self.vit(rgb.view(B * T, C, H, W)).view(B, T, -1)

        # Motion via CNN projector
        motion_feat = self.motion_proj(motion.view(B * T, 1, H, W)).view(B, T, -1)

        # Cross-attention fusion
        fused = self.fusion(rgb_feat, motion_feat)

        # Temporal aggregation
        video_feat = self.temporal_attn(fused)

        # Classify
        return self.classifier(video_feat)


# ── Sanity check ─────────────────────────────────────────────
if __name__ == "__main__":
    model = MGViT().to(Config.DEVICE)
    rgb    = torch.randn(2, 8, 3, 224, 224).to(Config.DEVICE)
    motion = torch.randn(2, 8, 1, 224, 224).to(Config.DEVICE)
    out    = model(rgb, motion)
    print(f"Output shape    : {out.shape}")
    print(f"Device          : {Config.DEVICE}")
    total  = sum(p.numel() for p in model.parameters()) / 1e6
    train_ = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params    : {total:.1f}M")
    print(f"Trainable params: {train_:.1f}M")