"""
calculate.py - CMB 탐지 모델 평가 지표 계산 모듈

이 모듈은 CMB 탐지 모델의 성능을 평가하는 함수들을 정의합니다.

주요 지표:
1. mAP (mean Average Precision): 픽셀 단위 분류 정확도
2. FP/CMB (False Positives per CMB): CMB당 오탐지 수
3. Precision: 정밀도

평가 방식:
- 픽셀 단위 평가 (Semantic Segmentation 방식)
- sklearn의 average_precision_score 사용
"""

import time
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_score


# ==========================================
# mAP 계산 함수 (픽셀 단위)
# ==========================================
def calculate_mAP_precision(model, loader, device, mode="Test", return_metrics=False, iou_threshold=0.1):
    """
    mAP (mean Average Precision) 계산 - 픽셀 단위

    각 픽셀을 배경(0) 또는 병변(1)으로 분류하는 성능을 측정합니다.
    sklearn의 average_precision_score를 사용합니다.

    Args:
        model: 평가할 모델
        loader: 데이터 로더
        device: 연산 디바이스 (cuda/cpu)
        mode: 평가 모드 이름 ("Train"/"Test")
        return_metrics: True시 지표 반환
        iou_threshold: (미사용, 호환성 유지용)

    Returns:
        mAP: mean Average Precision (픽셀 단위)
        mAP: (동일값 반환, 호환성)
        precision: Precision
        0.0: (호환성용 더미값)
    """
    model.eval()
    start_time = time.time()
    print(f"\n[{mode}] 평가 지표 계산 중 ({len(loader)} 배치)...")

    all_preds = []   # 모든 예측 확률값
    all_labels = []  # 모든 GT 라벨

    with torch.no_grad():
        for swi_images, roi_masks, _ in tqdm(loader, desc=f"{mode} Metrics", leave=False):
            swi_images = swi_images.to(device)
            roi_masks = roi_masks.to(device)

            # ---- 순전파 (Anchor 버전: 3개 출력) ----
            outputs = model(swi_images)
            if len(outputs) == 3:
                _, scores, _ = outputs  # anchor 버전
            else:
                _, scores = outputs  # 기존 버전

            # ---- GT 마스크를 출력 크기에 맞게 리사이즈 ----
            B = scores.shape[0]
            H, W = scores.shape[-2:]
            roi_masks_resized = F.interpolate(
                roi_masks.unsqueeze(1).float(),
                size=(H, W),
                mode="nearest"
            ).squeeze(1)
            roi_masks_resized = (roi_masks_resized > 0).long()

            # ---- CMB 클래스 확률 추출 ----
            # Anchor 버전: scores = (B, num_anchors, H, W), Sigmoid 사용
            # 기존 버전: scores = (B, 2, H, W), Softmax 사용
            if scores.shape[1] <= 3:  # Anchor 버전 (1-3 anchors)
                probs = torch.sigmoid(scores).max(dim=1)[0]  # 최대 anchor 확률
            else:
                probs = torch.softmax(scores, dim=1)[:, 1, :, :]  # 기존 방식

            # 평탄화하여 저장
            all_preds.append(probs.cpu().numpy().ravel())
            all_labels.append(roi_masks_resized.cpu().numpy().ravel())

    # 전체 데이터 결합
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # ---- mAP 계산 (sklearn) ----
    mAP = average_precision_score(all_labels, all_preds)

    # ---- Precision 계산 ----
    binary_preds = (all_preds > 0.5).astype(int)
    precision = precision_score(all_labels, binary_preds, zero_division=0)

    # 소요 시간 계산
    elapsed = time.time() - start_time

    # 디버그 정보 출력
    print(
        f"   Scores - Min: {np.min(all_preds):.4f}, Max: {np.max(all_preds):.4f}, Mean: {np.mean(all_preds):.4f}")
    print(
        f"   >>> mAP: {mAP:.4f} | Precision: {precision:.4f} (⏱ {elapsed:.1f}초)")

    if return_metrics:
        return mAP, mAP, precision, 0.0

    return mAP, mAP, precision, 0.0


# ==========================================
# FP/CMB 계산 함수
# ==========================================
def calculate_fp_cmb(model, test_loader, device, mode="Eval"):
    """
    False Positives per CMB (FP/CMB) 계산

    CMB 탐지 모델의 오탐지율을 평가합니다.
    CMB 하나당 평균 몇 개의 오탐지가 발생하는지 측정합니다.

    Args:
        model: 평가할 모델
        test_loader: 데이터 로더
        device: 연산 디바이스
        mode: 평가 모드 이름

    출력:
        FP/CMB 값 (낮을수록 좋음)

    계산 방식:
        FP/CMB = 총 False Positives / 총 CMB 픽셀 수

    여기서:
        - FP: 병변이 없는데 있다고 예측한 픽셀 수
        - CMB: GT 마스크에서 병변인 픽셀 수
    """
    model.eval()
    start_time = time.time()
    fp_count, cmb_count = 0, 0

    print(f"\n[{mode}] False Positives per CMB (FP/CMB) 계산 중...")

    with torch.no_grad():
        for swi_images, roi_masks, _ in test_loader:
            swi_images = swi_images.to(device)
            roi_masks = roi_masks.to(device)

            # 순전파 (Anchor 버전: 3개 출력)
            outputs = model(swi_images)
            if len(outputs) == 3:
                _, scores, _ = outputs
            else:
                _, scores = outputs

            B = scores.shape[0]
            H, W = scores.shape[-2:]

            # GT 마스크를 출력 크기에 맞게 리사이즈
            roi_masks_resized = F.interpolate(
                roi_masks.unsqueeze(1).float(),
                size=(H, W),
                mode="nearest"
            ).squeeze(1)
            roi_masks_resized = (roi_masks_resized > 0).float()

            # 예측값을 확률로 변환 후 이진화
            if scores.shape[1] <= 3:  # Anchor 버전
                probs = torch.sigmoid(scores).max(dim=1)[0]
            else:
                probs = torch.softmax(scores, dim=1)[:, 1, :, :]
            preds = (probs > 0.5).float()

            # FP 카운트: 예측=1 이고 GT=0 인 픽셀 수
            fp_count += ((preds == 1) & (roi_masks_resized == 0)).sum().item()

            # CMB 카운트: GT=1 인 픽셀 수
            cmb_count += (roi_masks_resized == 1).sum().item()

    # 결과 출력
    elapsed = time.time() - start_time
    if cmb_count == 0:
        print(f"   ⚠️ [{mode}] CMB 수가 0이므로 FP/CMB 계산 불가 (⏱ {elapsed:.1f}초)")
        return

    fp_per_cmb = fp_count / cmb_count
    print(
        f"   ➡️ [{mode}] False Positives per CMB : {fp_per_cmb:.4f} (⏱ {elapsed:.1f}초)")
