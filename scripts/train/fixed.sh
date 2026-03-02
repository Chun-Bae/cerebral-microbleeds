#!/bin/bash

# 사용법: ./scripts/train/fixed.sh [MODEL] [RUN_NAME]
MODEL=${1:-"SSD_FE"}
RUN_NAME=${2:-"default"}

echo "🚀 >>> [학습] Fixed Split 진행..."
python train.py --model "$MODEL" --run_name "$RUN_NAME" --fixed_split
