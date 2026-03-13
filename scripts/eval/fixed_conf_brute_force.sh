#!/bin/bash

# 사용법: ./scripts/eval/fixed_conf_brute_force.sh [MODEL] [RUN_NAME]
MODEL=${1:-"SSD_FE"}
RUN_NAME=${2:-"default"}

DATA_FILE="froc_data_${MODEL}.csv"
echo "Conf,Recall,FP" > "$DATA_FILE"

echo "🚀 >>> Recall vs FP 수집 시작..."

for CONF in 0.0001 0.0003 0.0005 0.0007 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.10 0.15 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95
do
    echo "🎯 >>> Confidence: $CONF 평가 중..."
    

    RESULT=$(python evaluate.py --model "$MODEL" --run_name "$RUN_NAME" --fixed_split --conf "$CONF" --no_vis)
    
    # RECALL 추출: 'Recall' 뒤에 오는 숫자만 정확히 추출
# RECALL 추출: 숫자만 뽑고 바로 특수문자 제거
RECALL=$(echo "$RESULT" | grep -w "Recall" | awk -F': ' '{print $2}' | sed 's/\x1b\[[0-9;]*m//g' | xargs)

# FP 추출: 'FP/CBM' 제외하고 숫자만 뽑은 뒤 특수문자 제거
FP=$(echo "$RESULT" | grep -w "FP" | grep -v "/" | awk -F': ' '{print $2}' | sed 's/\x1b\[[0-9;]*m//g' | xargs)
    # 파일에 저장
    echo "$CONF,$RECALL,$FP" >> "$DATA_FILE"
    echo "   ✅ 수집 완료: Recall=$RECALL, FP=$FP"
done

echo "📊 >>> 그래프 생성 중..."

python3 -c "
import matplotlib.pyplot as plt
import pandas as pd
try:
    df = pd.read_csv('$DATA_FILE').sort_values('FP')
    plt.figure(figsize=(10, 6))
    plt.plot(df['FP'], df['Recall'], 'o-', linewidth=2, markersize=8, color='#ff4757', label='FROC Curve')
    plt.xlabel('Total False Positives (FP)')
    plt.ylabel('Recall (Sensitivity)')
    plt.title('Recall vs FP Curve ($MODEL)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('combined_froc_curve.png')
    print('✨ 그래프 저장 완료: combined_froc_curve.png')
except Exception as e:
    print(f'❌ 그래프 생성 실패: {e}')
"
