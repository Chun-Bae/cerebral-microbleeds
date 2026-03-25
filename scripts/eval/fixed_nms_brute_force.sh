#!/bin/bash

# 사용법: ./scripts/eval/fixed_nms_brute_force.sh [MODEL] [RUN_NAME] [CONF]
MODEL=${1:-"SSD_FE"}
RUN_NAME=${2:-"default"}
CONF=${3:-0.001}

DATA_FILE="froc_nms_data_${MODEL}.csv"
echo "NMS_Thresh,TP,FP,FN,FP_CBM,Precision,Recall,F1,AP" > "$DATA_FILE"

echo "🚀 >>> NMS_THRESH 수집 시작... (CONF=$CONF)"

for NMS in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90
do
    echo "🎯 >>> NMS_THRESH: $NMS 평가 중..."

    RESULT=$(python evaluate.py --model "$MODEL" --run_name "$RUN_NAME" --fixed_split --conf "$CONF" --nms_thresh "$NMS" --no_vis)

    TP=$(echo "$RESULT"        | grep -w "TP"        | awk -F': ' '{print $2}' | sed 's/\x1b\[[0-9;]*m//g' | xargs)
    FP=$(echo "$RESULT"        | grep -w "FP"        | grep -v "/" | awk -F': ' '{print $2}' | sed 's/\x1b\[[0-9;]*m//g' | xargs)
    FN=$(echo "$RESULT"        | grep -w "FN"        | awk -F': ' '{print $2}' | sed 's/\x1b\[[0-9;]*m//g' | xargs)
    FP_CBM=$(echo "$RESULT"    | grep "FP/CBM"       | awk -F': ' '{print $2}' | sed 's/\x1b\[[0-9;]*m//g' | xargs)
    PRECISION=$(echo "$RESULT" | grep -w "Precision" | awk -F': ' '{print $2}' | sed 's/\x1b\[[0-9;]*m//g' | xargs)
    RECALL=$(echo "$RESULT"    | grep -w "Recall"    | awk -F': ' '{print $2}' | sed 's/\x1b\[[0-9;]*m//g' | xargs)
    F1=$(echo "$RESULT"        | grep -w "F1"        | awk -F': ' '{print $2}' | sed 's/\x1b\[[0-9;]*m//g' | xargs)
    AP=$(echo "$RESULT"        | grep "AP@"          | awk -F': ' '{print $2}' | sed 's/\x1b\[[0-9;]*m//g' | xargs)

    echo "$NMS,$TP,$FP,$FN,$FP_CBM,$PRECISION,$RECALL,$F1,$AP" >> "$DATA_FILE"
    echo "   ✅ 수집 완료: Recall=$RECALL, Precision=$PRECISION, F1=$F1, FP=$FP"
done

echo "📊 >>> 그래프 생성 중..."

python3 -c "
import matplotlib.pyplot as plt
import pandas as pd
try:
    df = pd.read_csv('$DATA_FILE').sort_values('NMS_Thresh')

    metrics = [
        ('Recall',    '#2ed573'),
        ('Precision', '#1e90ff'),
        ('F1',        '#ffa502'),
        ('AP',        '#a29bfe'),
        ('FP',        '#ff4757'),
        ('FP_CBM',    '#ff6b81'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, (col, color) in zip(axes, metrics):
        ax.plot(df['NMS_Thresh'], df[col], 'o-', linewidth=2, markersize=6, color=color)
        ax.set_xlabel('NMS Threshold')
        ax.set_ylabel(col)
        ax.set_title(f'{col} vs NMS Threshold')
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle('NMS Threshold Sweep ($MODEL)', fontsize=14)
    plt.tight_layout()
    plt.savefig('nms_sweep_curve.png', dpi=150)
    print('✨ 그래프 저장 완료: nms_sweep_curve.png')
except Exception as e:
    print(f'❌ 그래프 생성 실패: {e}')
"

echo "✅ >>> 완료: $DATA_FILE"
