# ============================================================
# ablation.py — Run ablation experiments
# Each experiment trains with one component removed/changed
# Results saved to results/ablation_results.json
# ============================================================

import json
import os
import numpy as np
from config import Config
from train import train

os.makedirs(Config.RESULTS_DIR, exist_ok=True)

# ── Define ablation experiments ───────────────────────────────
ABLATIONS = [
    {
        "name":        "Full Model (cross-attn + augmentation)",
        "num_frames":  8,
        "use_cross_attn": True,
        "use_augmentation": True,
        "freeze_blocks": 8,
    },
    {
        "name":        "No Augmentation (overfitting repro)",
        "num_frames":  8,
        "use_cross_attn": True,
        "use_augmentation": False,
        "freeze_blocks": 8,
    },
    {
        "name":        "No Backbone Freezing",
        "num_frames":  8,
        "use_cross_attn": True,
        "use_augmentation": True,
        "freeze_blocks": 0,
    },
    {
        "name":        "T=4 Frames",
        "num_frames":  4,
        "use_cross_attn": True,
        "use_augmentation": True,
        "freeze_blocks": 8,
    },
    {
        "name":        "T=16 Frames",
        "num_frames":  16,
        "use_cross_attn": True,
        "use_augmentation": True,
        "freeze_blocks": 8,
    },
    {
        "name":        "Baseline Fusion (simple add, no cross-attn)",
        "num_frames":  8,
        "use_cross_attn": False,
        "use_augmentation": True,
        "freeze_blocks": 8,
    },
]


def patch_config(ablation_cfg):
    """Temporarily patch Config with ablation settings."""
    Config.NUM_FRAMES = ablation_cfg["num_frames"]
    Config.USE_CROSS_ATTN = ablation_cfg["use_cross_attn"]
    Config.USE_AUGMENTATION = ablation_cfg["use_augmentation"]
    Config.FREEZE_BACKBONE_BLOCKS = ablation_cfg["freeze_blocks"]


def run_ablations():
    results = {}

    for i, ablation in enumerate(ABLATIONS):
        print(f"\n{'='*60}")
        print(f"Ablation {i+1}/{len(ABLATIONS)}: {ablation['name']}")
        print(f"{'='*60}")

        patch_config(ablation)

        accs = []
        for seed in [42, 123, 7]:
            acc = train(seed=seed, run_id=f"ablation_{i}_seed{seed}")
            accs.append(acc * 100)

        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)

        results[ablation["name"]] = {
            "accuracies": accs,
            "mean": round(mean_acc, 2),
            "std":  round(std_acc, 2),
            "report": f"{mean_acc:.2f} ± {std_acc:.2f}%"
        }

        print(f"\nResult: {results[ablation['name']]['report']}")

    out_path = os.path.join(Config.RESULTS_DIR, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Experiment':<45} {'Accuracy':>15}")
    print("-" * 62)
    for name, res in results.items():
        print(f"{name:<45} {res['report']:>15}")

    return results


if __name__ == "__main__":
    run_ablations()