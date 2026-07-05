#!/bin/bash
# ============================================================
# UCF-101 Dataset Setup Script (macOS)
# ============================================================

echo "=== MG-ViT Dataset Setup ==="
echo "Downloading UCF-101 and train/test splits..."
echo ""

# Create folders
mkdir -p ~/datasets/UCF-101
mkdir -p ~/datasets/ucf101_splits

# ============================================================
# STEP 1: Download UCF-101
# ============================================================
echo ">>> Downloading UCF-101..."
cd ~/datasets/UCF-101 || exit

wget --no-check-certificate -c \
"https://www.crcv.ucf.edu/data/UCF101/UCF101.rar" \
-O UCF101.rar

if [ ! -f UCF101.rar ]; then
    echo "Download failed!"
    exit 1
fi

echo ">>> Extracting UCF-101..."
unar UCF101.rar

# ============================================================
# STEP 2: Download train/test splits
# ============================================================
echo ""
echo ">>> Downloading train/test splits..."
cd ~/datasets/ucf101_splits || exit

wget --no-check-certificate -c \
"https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"

unzip -o UCF101TrainTestSplits-RecognitionTask.zip

echo ""
echo "========================================"
echo "Dataset setup completed!"
echo "========================================"
echo "Dataset:"
echo "~/datasets/UCF-101/UCF-101"
echo ""
echo "Splits:"
echo "~/datasets/ucf101_splits/ucfTrainTestlist"