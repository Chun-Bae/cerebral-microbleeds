"""
evaluate.py - CMB 탐지 모델 평가 및 시각화 통합 모듈

이 모듈은 모델 평가와 결과 시각화를 하나의 함수로 통합합니다.
trainer.py (학습 후 평가)와 inference.py (독립 평가) 모두에서 사용합니다.

주요 기능:
1. mAP, FP/CMB 지표 계산
2. FROC 곡선 생성
3. Confusion Matrix 생성
4. 결과 저장
"""

import os
from calculate import calculate_mAP_precision, calculate_fp_cmb
from plot import plot_froc, plot_confusion_matrix


def evaluate_model(model, train_loader, test_loader, device, save_dir, prefix=""):
    """
    모델 평가 및 결과 시각화 통합 함수

    Train/Test 세트에 대해 mAP, FP/CMB를 계산하고
    FROC 곡선과 Confusion Matrix를 저장합니다.

    Args:
        model: 평가할 모델 (이미 가중치가 로드된 상태)
        train_loader: Train 데이터 로더
        test_loader: Test 데이터 로더
        device: 연산 디바이스 (cuda/cpu)
        save_dir: 결과 저장 디렉토리
        prefix: 파일명 접두사 (예: "train", "inf_train")

    Returns:
        dict: 평가 결과 딕셔너리
            - train_mAP: Train mAP
            - test_mAP: Test mAP
            - train_fp_cmb: Train FP/CMB (추후 확장용)
            - test_fp_cmb: Test FP/CMB (추후 확장용)
    """
    results = {}

    # 결과 저장 폴더 생성
    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # [1] 평가 지표 계산
    # ==========================================
    print("\n" + "=" * 40)
    print("       모델 평가 (Evaluation)       ")
    print("=" * 40)

    # Train Set 평가
    print("\n>>> [1] Train Set 평가")
    train_mAP, _, train_recall, _ = calculate_mAP_precision(
        model, train_loader, device, mode="Train", return_metrics=True
    )
    calculate_fp_cmb(model, train_loader, device, mode="Train")
    results['train_mAP'] = train_mAP
    results['train_recall'] = train_recall

    # Test Set 평가
    print("\n>>> [2] Test Set 평가")
    test_mAP, _, test_recall, _ = calculate_mAP_precision(
        model, test_loader, device, mode="Test", return_metrics=True
    )
    calculate_fp_cmb(model, test_loader, device, mode="Test")
    results['test_mAP'] = test_mAP
    results['test_recall'] = test_recall

    # ==========================================
    # [2] 결과 시각화 저장
    # ==========================================
    print("\n" + "=" * 40)
    print("       결과 시각화 저장 (Visualization)       ")
    print("=" * 40)

    # 파일명 접두사 처리
    train_prefix = f"{prefix}_train" if prefix else "train"
    test_prefix = f"{prefix}_test" if prefix else "test"

    # FROC 곡선 저장
    plot_froc(model, train_loader, device,
              save_dir=save_dir, prefix=train_prefix)
    plot_froc(model, test_loader, device,
              save_dir=save_dir, prefix=test_prefix)

    # Confusion Matrix 저장
    plot_confusion_matrix(model, train_loader, device,
                          save_dir=save_dir, prefix=train_prefix)
    plot_confusion_matrix(model, test_loader, device,
                          save_dir=save_dir, prefix=test_prefix)

    print(f"\n=== 평가 완료: 결과가 '{save_dir}'에 저장되었습니다. ===")

    return results
