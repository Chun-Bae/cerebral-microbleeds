#!/bin/bash

# 사용법: ./scripts/eval/fixed_conf_brute_force.sh [MODEL] [RUN_NAME]
MODEL=${1:-"SSD_FE"}
RUN_NAME=${2:-"default"}

# 0.05부터 0.40까지 0.05 단위로 평가 진행
for CONF in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95
do
    echo "🎯 >>> [평가] Fixed Split 진행 (Confidence: $CONF)..."
    python evaluate.py --model "$MODEL" --run_name "$RUN_NAME" --fixed_split --conf "$CONF"
done

