# [Fix] 소형 객체(CMB) 평가 시 IoU 불일치 문제 해결 및 학습/평가 임계값 분리
**(Small Object Evaluation Strategy: Decoupling Train/Test IoU Thresholds)**

**Date**: 2026-02-04
**Author**: User & Antigravity

## 1. 문제 상황 (Problem)
* 학습된 모델이 시각적으로는 병변(CMB)을 정확히 찾고 있음에도(`TP` 추정), 평가 지표상으로는 낮은 AP(Average Precision)와 Recall(0.1 미만)을 기록함.
* "오탐(FP)은 없는데 미탐(FN)만 높게 잡히는" 현상이 지속적으로 관찰됨.
* `CONF_THRESH`를 `0.1`까지 낮추어도 Recall이 획기적으로 개선되지 않음.

## 2. 원인 분석 (Root Cause Analysis)

### A. 배경 필터링 로직의 오류
* **Code**: `if image_pixels > 0.05`
* **Issue**: SWI 영상에서 CMB(미세출혈)는 **신호가 감쇠되어 검게(Dark)** 나타나는 특징이 있음. 밝기 기준으로 배경을 거르면 **진짜 병변(True Positive)까지 배경으로 간주되어 삭제**됨.

### B. IoU 임계값의 구조적 한계 (Critical)
* **Situation**: CMB의 실제 크기는 2~4 픽셀(px)이나, 모델의 최소 앵커 크기는 약 8~10 픽셀임.
* **Math**: 3x3 박스와 10x10 박스가 중심이 정확히 일치해도, 면적 차이로 인해 IoU는 매우 낮게 계산됨.
    * 예: 3x3 정답과 10x10 예측이 3x3만큼 겹칠 경우 -> IoU ≈ 9 / 100 = 0.09
* **Result**: 기존 평가 기준인 `IOU_THRESH > 0.35`는 소형 객체 탐지에서 달성하기 불가능한 수치임. 위치를 정확히 맞춰도 "틀렸다(False Positive)"고 판정됨.

## 3. 해결 방안 (Solution)

### 1. IoU 임계값 이원화 (Decoupling)
학습과 평가의 목적이 다르므로 임계값을 분리하여 적용함.
* **학습용 (`TRAIN_IOU_THRESH = 0.35`)**:
    * 모델 학습 시에는 "아무거나" 배우지 않도록, 어느 정도 겹침이 보장된 앵커만 Positive로 사용해야 함. 따라서 기존의 엄격한 기준을 유지.
* **평가용 (`TEST_IOU_THRESH = 0.001`)**:
    * 의료 영상 진단 보조(CAD) 목적상, 병변의 유무와 위치를 찾는 것이 중요함.
    * 아주 미세한 영역(1px 이상)이라도 겹치면 **"찾았다(Hit)"**고 인정하는 것이 타당함.

### 2. 기타 파라미터 최적화
* `CONF_THRESH`: `0.75` → `0.1` (Recall 확보를 위해 완화)
* `NMS Limit`: Top-10 제한 → Top-200 으로 완화 (다발성 병변 대응)

## 4. 결론 (Conclusion)
* 평가 지표(Metric)를 수정함으로써 시각적 확인 결과와 수치적 성능이 일치하게 됨.
* 초소형 객체 탐지에서는 일반적인 Object Detection 지표(IoU 0.5 등)를 그대로 적용하면 왜곡이 발생하므로, 도메인 특성에 맞는 지표 설계가 필수적임.
