#!/bin/bash

# 사용법: ./scripts/train_eval/kfold.sh [MODEL] [RUN_NAME]
MODEL=${1:-"SSD_FE"}
RUN_NAME=${2:-"default"}

echo "🚀 >>> [학습] K-Fold 진행..."
python train.py --model "$MODEL" --run_name "$RUN_NAME"

echo ""
echo "🎯 >>> [평가] K-Fold 진행..."
python evaluate.py --model "$MODEL" --run_name "$RUN_NAME"
