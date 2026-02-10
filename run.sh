#!/bin/bash

# 에러 발생 시 중단
set -e

# 설정
WEIGHTS="weights/latest_ssd_fold_0.pth"

# 평가 함수 정의
run_eval() {
    echo "----------------------------------------------------------------"
    echo "📊 평가 실행 (Model: $WEIGHTS)"
    echo "----------------------------------------------------------------"
    
    # 1. 특정 환자 (VK049) 평가 (Training Set 사용)
    echo "👉 1. Patient VK049 Check (Train Set Fold 0)"
    python evaluate.py --patient VK049 --lmdb_path data/lmdb/fold_0/train.lmdb --model "$WEIGHTS"
    
    # 2. 전체 평가 (기본 Holdout Test 사용)
    echo "👉 2. General Evaluation (Holdout Test)"
    python evaluate.py --model "$WEIGHTS"
}

# --- Step 1: Epoch 50 ---
echo "================================================================"
echo "🚀 [Step 1] Training up to Epoch 50"
echo "================================================================"
# args.weights로 경로를 지정하면, 파일이 있으면 로드(Resume), 없으면 새로 시작합니다.
python main.py --epochs 50 --weights "$WEIGHTS" --fold 0

run_eval

# --- Step 2: Epoch 100 ---
echo "================================================================"
echo "🚀 [Step 2] Training up to Epoch 100"
echo "================================================================"
# 이전 단계에서 저장된 WEIGHTS 파일을 로드하여 이어 학습합니다.
python main.py --epochs 100 --weights "$WEIGHTS" --fold 0

run_eval

# --- Step 3: Epoch 200 ---
echo "================================================================"
echo "🚀 [Step 3] Training up to Epoch 200"
echo "================================================================"
python main.py --epochs 200 --weights "$WEIGHTS" --fold 0

run_eval

echo "✅ 모든 작업 완료!"
