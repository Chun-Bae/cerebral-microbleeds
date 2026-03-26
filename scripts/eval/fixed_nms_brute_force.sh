#!/bin/bash

# 사용법: ./scripts/eval/fixed_nms_brute_force.sh [MODEL] [RUN_NAME] [CONF]
MODEL=${1:-"SSD_FE"}
RUN_NAME=${2:-"default"}

echo "🚀 >>> NMS_THRESH 수집 시작... (CONF=$CONF)"

# for NMS in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90
for NMS in 0.000001 0.00001 0.0001 0.001 0.01
do
    echo "🎯 >>> NMS_THRESH: $NMS 평가 중..."
    python evaluate.py --model "$MODEL" --run_name "$RUN_NAME" --fixed_split --conf 0.02 --nms_thresh "$NMS" --no_vis
done