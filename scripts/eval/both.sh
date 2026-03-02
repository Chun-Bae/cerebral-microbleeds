#!/bin/bash

# 사용법: ./scripts/eval/both.sh [MODEL] [RUN_NAME]
MODEL=${1:-"SSD_FE"}
RUN_NAME=${2:-"default"}

echo "🎯 >>> [평가] Fixed Split 진행..."
python evaluate.py --model "$MODEL" --run_name "$RUN_NAME"

echo ""
echo "🎯 >>> [평가] K-Fold 진행..."
python evaluate.py --model "$MODEL" --run_name "$RUN_NAME"
