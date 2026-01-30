"""
trainer.py - CMB(뇌미세출혈) 탐지 모델 학습 모듈 (Anchor 기반)

이 모듈은 SSD-FE 모델의 학습과 검증을 담당합니다.
주요 기능:
- Anchor 기반 다중 객체 탐지 학습
- IoU 기반 anchor-GT 매칭
- Focal Loss (클래스 불균형 해결)
- 예측 vs GT 시각화 (10에폭마다)
"""

import time
import datetime
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from calculate import calculate_mAP_precision


# ==========================================
# Anchor 기반 유틸리티 함수들
# ==========================================

def compute_iou(boxes1, boxes2):
    """
    두 박스 세트 간의 IoU(Intersection over Union) 계산

    Args:
        boxes1: (N, 4) - [cx, cy, w, h] 형식
        boxes2: (M, 4) - [cx, cy, w, h] 형식

    Returns:
        iou: (N, M) - 각 박스 쌍의 IoU
    """
    # cx, cy, w, h -> x1, y1, x2, y2 변환
    boxes1_xyxy = torch.stack([
        boxes1[:, 0] - boxes1[:, 2] / 2,  # x1
        boxes1[:, 1] - boxes1[:, 3] / 2,  # y1
        boxes1[:, 0] + boxes1[:, 2] / 2,  # x2
        boxes1[:, 1] + boxes1[:, 3] / 2,  # y2
    ], dim=1)

    boxes2_xyxy = torch.stack([
        boxes2[:, 0] - boxes2[:, 2] / 2,
        boxes2[:, 1] - boxes2[:, 3] / 2,
        boxes2[:, 0] + boxes2[:, 2] / 2,
        boxes2[:, 1] + boxes2[:, 3] / 2,
    ], dim=1)

    # 교집합 계산
    N, M = boxes1.shape[0], boxes2.shape[0]

    # (N, 1, 4) vs (1, M, 4) -> (N, M)
    b1 = boxes1_xyxy.unsqueeze(1).expand(N, M, 4)
    b2 = boxes2_xyxy.unsqueeze(0).expand(N, M, 4)

    inter_x1 = torch.max(b1[:, :, 0], b2[:, :, 0])
    inter_y1 = torch.max(b1[:, :, 1], b2[:, :, 1])
    inter_x2 = torch.min(b1[:, :, 2], b2[:, :, 2])
    inter_y2 = torch.min(b1[:, :, 3], b2[:, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # 합집합 계산
    area1 = boxes1[:, 2] * boxes1[:, 3]  # w * h
    area2 = boxes2[:, 2] * boxes2[:, 3]

    area1 = area1.unsqueeze(1).expand(N, M)
    area2 = area2.unsqueeze(0).expand(N, M)

    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-6)


def match_anchors_to_gt(anchors, gt_boxes, pos_thresh=0.3, neg_thresh=0.1):
    """
    Anchor와 GT bbox 매칭 (CMB 탐지용)

    규칙:
    - IoU > pos_thresh: Positive (해당 GT 담당)
    - IoU < neg_thresh: Negative (배경)
    - 그 사이: Ignore (학습에서 제외)
    - 각 GT의 max IoU anchor는 무조건 Positive

    Args:
        anchors: (num_anchors, 4) - [cx, cy, w, h]
        gt_boxes: (num_gt, 4) - [x_min, y_min, x_max, y_max] 정규화 좌표
        pos_thresh: Positive 판정 IoU 임계값
        neg_thresh: Negative 판정 IoU 임계값

    Returns:
        labels: (num_anchors,) - 1=positive, 0=negative, -1=ignore
        matched_gt_idx: (num_anchors,) - 매칭된 GT 인덱스 (-1=미매칭)
    """
    num_anchors = anchors.shape[0]
    num_gt = gt_boxes.shape[0]
    device = anchors.device

    # GT가 없으면 모든 anchor = negative
    if num_gt == 0:
        labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        matched_gt_idx = torch.full(
            (num_anchors,), -1, dtype=torch.long, device=device)
        return labels, matched_gt_idx

    # GT를 cx, cy, w, h 형식으로 변환
    gt_cxcywh = torch.stack([
        (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,  # cx
        (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2,  # cy
        gt_boxes[:, 2] - gt_boxes[:, 0],        # w
        gt_boxes[:, 3] - gt_boxes[:, 1],        # h
    ], dim=1)

    # IoU 계산
    iou_matrix = compute_iou(anchors, gt_cxcywh)  # (num_anchors, num_gt)

    # 각 anchor의 최대 IoU와 해당 GT 인덱스
    max_iou, matched_gt_idx = iou_matrix.max(dim=1)

    # 라벨 초기화 (-1 = ignore)
    labels = torch.full((num_anchors,), -1, dtype=torch.long, device=device)

    # Positive: IoU > pos_thresh
    labels[max_iou >= pos_thresh] = 1

    # Negative: IoU < neg_thresh
    labels[max_iou < neg_thresh] = 0

    # 각 GT에 대해 max IoU anchor는 무조건 Positive (작은 GT도 매칭 보장)
    for gt_idx in range(num_gt):
        best_anchor_idx = iou_matrix[:, gt_idx].argmax()
        labels[best_anchor_idx] = 1
        matched_gt_idx[best_anchor_idx] = gt_idx

    return labels, matched_gt_idx


def encode_bbox_offset(anchors, gt_boxes, matched_gt_idx):
    """
    GT bbox를 anchor 기준 오프셋으로 인코딩

    Args:
        anchors: (num_anchors, 4) - [cx, cy, w, h]
        gt_boxes: (num_gt, 4) - [x_min, y_min, x_max, y_max]
        matched_gt_idx: (num_anchors,) - 매칭된 GT 인덱스

    Returns:
        offsets: (num_anchors, 4) - [dx, dy, dw, dh]
    """
    device = anchors.device
    num_anchors = anchors.shape[0]

    offsets = torch.zeros(num_anchors, 4, device=device)

    valid_mask = matched_gt_idx >= 0
    if not valid_mask.any():
        return offsets

    matched_gt = gt_boxes[matched_gt_idx[valid_mask]]
    valid_anchors = anchors[valid_mask]

    # GT를 cx, cy, w, h로 변환
    gt_cx = (matched_gt[:, 0] + matched_gt[:, 2]) / 2
    gt_cy = (matched_gt[:, 1] + matched_gt[:, 3]) / 2
    gt_w = matched_gt[:, 2] - matched_gt[:, 0]
    gt_h = matched_gt[:, 3] - matched_gt[:, 1]

    # 오프셋 계산 (anchor 기준)
    dx = (gt_cx - valid_anchors[:, 0]) / (valid_anchors[:, 2] + 1e-6)
    dy = (gt_cy - valid_anchors[:, 1]) / (valid_anchors[:, 3] + 1e-6)
    dw = torch.log(gt_w / (valid_anchors[:, 2] + 1e-6) + 1e-6)
    dh = torch.log(gt_h / (valid_anchors[:, 3] + 1e-6) + 1e-6)

    offsets[valid_mask, 0] = dx
    offsets[valid_mask, 1] = dy
    offsets[valid_mask, 2] = dw
    offsets[valid_mask, 3] = dh

    return offsets


class FocalLoss(nn.Module):
    """
    Focal Loss - 클래스 불균형 해결
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: (N,) - sigmoid 적용 전 logits
            target: (N,) - 0 또는 1
            mask: (N,) - True인 위치만 loss 계산 (ignore 제외)
        """
        pred_sigmoid = torch.sigmoid(pred)

        # BCE 계산
        bce = F.binary_cross_entropy_with_logits(
            pred, target.float(), reduction='none')

        # Focal weight
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)

        focal_loss = alpha_weight * focal_weight * bce

        if mask is not None:
            focal_loss = focal_loss[mask]

        return focal_loss.mean() if focal_loss.numel() > 0 else torch.tensor(0.0, device=pred.device)


def visualize_predictions(image, gt_boxes, pred_boxes, pred_scores, save_path, threshold=0.5):
    """
    예측 결과와 GT를 한 이미지에 시각화

    Args:
        image: (3, H, W) 텐서 또는 (H, W, 3) numpy
        gt_boxes: GT bbox 리스트 [[x1, y1, x2, y2], ...]
        pred_boxes: 예측 bbox 리스트
        pred_scores: 예측 confidence 리스트
        save_path: 저장 경로
        threshold: 시각화할 최소 confidence
    """
    # 텐서 -> numpy 변환
    if torch.is_tensor(image):
        image = image.cpu().permute(1, 2, 0).numpy()
        image = ((image * 0.5) + 0.5) * 255  # 역정규화
        image = np.clip(image, 0, 255).astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    H, W = image.shape[:2]

    # GT 박스 그리기 (초록색)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, 'GT', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 예측 박스 그리기 (빨간색)
    for box, score in zip(pred_boxes, pred_scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, f'{score:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(save_path, image)


def decode_bbox_from_anchor(anchors, offsets):
    """
    Anchor + offset -> 실제 bbox 좌표 디코딩

    Args:
        anchors: (N, 4) - [cx, cy, w, h]
        offsets: (N, 4) - [dx, dy, dw, dh]

    Returns:
        boxes: (N, 4) - [x1, y1, x2, y2]
    """
    cx = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
    cy = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]
    w = anchors[:, 2] * torch.exp(offsets[:, 2])
    h = anchors[:, 3] * torch.exp(offsets[:, 3])

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression

    Args:
        boxes: (N, 4) - [x1, y1, x2, y2]
        scores: (N,)
        iou_threshold: NMS IoU 임계값

    Returns:
        keep: 유지할 인덱스
    """
    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    # torchvision NMS 사용 가능하면 사용, 아니면 직접 구현
    try:
        from torchvision.ops import nms as tv_nms
        return tv_nms(boxes, scores, iou_threshold)
    except ImportError:
        pass

    # 직접 구현
    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # 나머지와 IoU 계산
        others = order[1:]
        ious = compute_iou(
            boxes[i:i+1].unsqueeze(0).squeeze(0),
            boxes[others]
        ).squeeze(0)

        # IoU < threshold인 것만 유지
        mask = ious < iou_threshold
        order = others[mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# wandb (optional - 설치되지 않았으면 비활성화)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device, result_dir, eval_interval):
    """
    Anchor 기반 다중 객체 탐지 학습 함수

    Args:
        model: SSD_FE 모델 인스턴스 (Anchor 기반)
        train_loader: 학습 데이터 로더
        test_loader: 테스트 데이터 로더
        criterion: (사용 안 함 - Focal Loss 사용)
        optimizer: 옵티마이저 (Adam)
        scheduler: 학습률 스케줄러 (현재 None)
        num_epochs: 총 학습 에폭 수
        device: 학습 디바이스 (cuda/cpu)
        result_dir: 결과 저장 디렉토리 경로
        eval_interval: mAP 평가 주기 (에폭 단위, 기본값: 10)

    Returns:
        None (결과는 파일로 저장됨)
    """

    # ==========================================
    # [1] 초기화
    # ==========================================

    # Anchor 기반 손실 함수
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    bbox_criterion = torch.nn.SmoothL1Loss(reduction='none')

    # 시각화 저장 폴더
    viz_dir = os.path.join(result_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # 체크포인트 저장 디렉토리 생성
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(
        checkpoint_dir, "checkpoint_latest.pth")

    # 학습 상태 변수 초기화
    start_epoch = 0
    train_loss_history = []

    # 학습 히스토리 딕셔너리
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # ==========================================
    # [2] 체크포인트 로드 (학습 재개)
    # ==========================================

    if os.path.exists(latest_checkpoint_path):
        print(f"\n🔄 체크포인트 발견: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)

        # 모델 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])

        # 옵티마이저 상태 로드 (학습률, 모멘텀 등 유지)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 학습 상태 복원
        start_epoch = checkpoint['epoch'] + 1
        prev_loss = checkpoint['loss']

        if 'train_loss_history' in checkpoint:
            train_loss_history = checkpoint['train_loss_history']

        if 'history' in checkpoint:
            history = checkpoint['history']

        print(f"   마지막 에폭: {checkpoint['epoch']+1}")
        print(f"   마지막 손실: {prev_loss:.4f}")
        print(f"   에폭 {start_epoch+1}부터 학습 재개.\n")
    else:
        print("\n🆕 새로운 학습을 시작합니다.\n")

    # ==========================================
    # [3] 학습 시작
    # ==========================================

    total_start_time = time.time()
    print(f"학습 시작 (총 {num_epochs} 에폭)...")

    for epoch in range(start_epoch, num_epochs):
        # 학습 모드 전환
        model.train()
        total_loss = 0.0  # 에폭 내 총 손실 누적

        # 프로그레스 바 생성
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        # ----------------------------------------
        # [3-1] 학습 루프 (Anchor 기반)
        # ----------------------------------------
        for batch_idx, (swi_images, roi_masks, gt_boxes_list) in enumerate(pbar):

            # 데이터를 GPU로 이동
            swi_images = swi_images.to(device)
            roi_masks = roi_masks.to(device)

            # 그래디언트 초기화
            optimizer.zero_grad()

            # ---- 1) 순전파 (Forward Pass) ----
            locs, scores, anchors = model(swi_images, gt_mask=roi_masks)
            # locs: (B, num_anchors*4, H', W') - bbox 오프셋 예측
            # scores: (B, num_anchors, H', W') - CMB 확률 (Sigmoid 전)
            # anchors: (num_anchors_total, 4) - [cx, cy, w, h]

            B = swi_images.shape[0]
            num_anchors_per_loc = model.num_anchors
            _, _, H, W = scores.shape
            num_anchors_total = H * W * num_anchors_per_loc

            # 출력 reshape: (B, num_anchors, H, W) -> (B, num_anchors_total)
            scores_flat = scores.permute(
                0, 2, 3, 1).reshape(B, num_anchors_total)
            locs_flat = locs.permute(0, 2, 3, 1).reshape(
                B, num_anchors_total, 4)

            # ---- 2) 배치별 타겟 생성 및 손실 계산 ----
            batch_cls_loss = 0.0
            batch_bbox_loss = 0.0

            for b in range(B):
                gt_boxes = gt_boxes_list[b].to(device)  # (num_gt, 4)

                # Anchor-GT 매칭
                labels, matched_gt_idx = match_anchors_to_gt(
                    anchors, gt_boxes, pos_thresh=0.3, neg_thresh=0.1
                )

                # 분류 타겟 및 마스크
                pos_mask = (labels == 1)
                neg_mask = (labels == 0)
                valid_mask = pos_mask | neg_mask  # ignore 제외

                # Focal Loss (분류)
                cls_target = pos_mask.float()
                cls_loss = focal_loss(scores_flat[b], cls_target, valid_mask)
                batch_cls_loss += cls_loss

                # Bbox 손실 (positive만)
                if pos_mask.sum() > 0:
                    # 오프셋 타겟 계산
                    offset_targets = encode_bbox_offset(
                        anchors, gt_boxes, matched_gt_idx)

                    # SmoothL1Loss (positive anchor만)
                    bbox_loss_per_anchor = bbox_criterion(
                        locs_flat[b], offset_targets)
                    bbox_loss = bbox_loss_per_anchor[pos_mask].mean()
                    batch_bbox_loss += bbox_loss

            # 배치 평균
            cls_loss_avg = batch_cls_loss / B
            bbox_loss_avg = batch_bbox_loss / B

            # 총 손실
            loss = cls_loss_avg + bbox_loss_avg

            # ---- 3) 역전파 및 가중치 업데이트 ----
            loss.backward()
            optimizer.step()

            # 손실 누적 및 표시
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 에폭 평균 학습 손실 계산
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        train_loss_history.append(avg_train_loss)

        # ----------------------------------------
        # [3-2] 검증 루프 (Anchor 기반)
        # ----------------------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for swi_images, roi_masks, gt_boxes_list in test_loader:
                swi_images = swi_images.to(device)
                roi_masks = roi_masks.to(device)

                # 순전파
                locs, scores, anchors = model(swi_images, gt_mask=roi_masks)

                B = swi_images.shape[0]
                num_anchors_per_loc = model.num_anchors
                _, _, H, W = scores.shape
                num_anchors_total = H * W * num_anchors_per_loc

                scores_flat = scores.permute(
                    0, 2, 3, 1).reshape(B, num_anchors_total)
                locs_flat = locs.permute(0, 2, 3, 1).reshape(
                    B, num_anchors_total, 4)

                batch_val_loss = 0.0
                for b in range(B):
                    gt_boxes = gt_boxes_list[b].to(device)

                    labels, matched_gt_idx = match_anchors_to_gt(
                        anchors, gt_boxes, pos_thresh=0.3, neg_thresh=0.1
                    )

                    pos_mask = (labels == 1)
                    neg_mask = (labels == 0)
                    valid_mask = pos_mask | neg_mask

                    cls_target = pos_mask.float()
                    cls_loss = focal_loss(
                        scores_flat[b], cls_target, valid_mask)

                    bbox_loss = torch.tensor(0.0, device=device)
                    if pos_mask.sum() > 0:
                        offset_targets = encode_bbox_offset(
                            anchors, gt_boxes, matched_gt_idx)
                        bbox_loss_per = bbox_criterion(
                            locs_flat[b], offset_targets)
                        bbox_loss = bbox_loss_per[pos_mask].mean()

                    batch_val_loss += cls_loss + bbox_loss

                val_loss += batch_val_loss.item() / B

        avg_val_loss = val_loss / len(test_loader)
        history['val_loss'].append(avg_val_loss)

        # 에폭 결과 출력
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.12f} | Val Loss: {avg_val_loss:.12f}")

        # wandb 로깅
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            })

        # ----------------------------------------
        # [3-3] 스케줄러 및 체크포인트
        # ----------------------------------------
        if scheduler:
            scheduler.step()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'history': history,
            'train_loss_history': train_loss_history
        }, latest_checkpoint_path)

        # ----------------------------------------
        # [3-4] 주기적 평가 및 시각화
        # ----------------------------------------
        if (epoch + 1) % eval_interval == 0:
            print(f"\n📊 [Epoch {epoch+1}] Test 평가 및 시각화 중...")

            # mAP 평가 (기존 함수 사용 - 호환성 문제 시 스킵)
            try:
                mAP, _, precision, _ = calculate_mAP_precision(
                    model, test_loader, device, mode="Test", return_metrics=True
                )
                print(f"   Test mAP: {mAP:.4f} | Precision: {precision:.4f}")
            except Exception as e:
                print(f"   mAP 계산 스킵 (anchor 버전 미호환): {e}")

            # 예측 시각화 (첫 몇 개 샘플)
            model.eval()
            viz_count = 0
            with torch.no_grad():
                for swi_images, roi_masks, gt_boxes_list in test_loader:
                    if viz_count >= 5:
                        break

                    swi_images = swi_images.to(device)
                    locs, scores, anchors = model(swi_images)

                    B = swi_images.shape[0]
                    num_anchors_per_loc = model.num_anchors
                    _, _, H, W = scores.shape
                    num_anchors_total = H * W * num_anchors_per_loc

                    scores_flat = scores.permute(
                        0, 2, 3, 1).reshape(B, num_anchors_total)
                    locs_flat = locs.permute(0, 2, 3, 1).reshape(
                        B, num_anchors_total, 4)

                    for b in range(B):
                        if viz_count >= 5:
                            break

                        # Sigmoid 적용하여 확률 계산
                        probs = torch.sigmoid(scores_flat[b])

                        # Top-k 또는 Threshold 이상인 anchor 선택
                        # (학습 초기에도 시각화되도록 낮은 threshold 사용)
                        conf_threshold = 0.3
                        conf_mask = probs > conf_threshold

                        # 탐지된 것이 없어도 GT는 시각화
                        if conf_mask.sum() == 0:
                            # 탐지 없음 - GT만 보여주기
                            gt_boxes = gt_boxes_list[b].numpy()
                            save_path = os.path.join(
                                viz_dir, f"epoch{epoch+1:03d}_sample{viz_count}.png"
                            )
                            visualize_predictions(
                                swi_images[b], gt_boxes, [], [
                                ], save_path, threshold=0.0
                            )
                            viz_count += 1
                            continue

                        # Bbox 디코딩
                        pred_boxes = decode_bbox_from_anchor(
                            anchors[conf_mask], locs_flat[b][conf_mask]
                        )
                        pred_scores = probs[conf_mask]

                        # NMS 적용
                        keep = nms(pred_boxes, pred_scores, iou_threshold=0.5)
                        pred_boxes = pred_boxes[keep].cpu().numpy()
                        pred_scores = pred_scores[keep].cpu().numpy()

                        # GT boxes
                        gt_boxes = gt_boxes_list[b].numpy()

                        # 시각화 저장
                        save_path = os.path.join(
                            viz_dir, f"epoch{epoch+1:03d}_sample{viz_count}.png"
                        )
                        visualize_predictions(
                            swi_images[b], gt_boxes, pred_boxes, pred_scores, save_path, threshold=0.0
                        )
                        viz_count += 1

            print(f"   ✅ 시각화 저장 완료: {viz_dir}")
            model.train()

        # 경과 시간 출력
        elapsed = time.time() - total_start_time
        print(f"⏱️ 경과 시간: {str(datetime.timedelta(seconds=int(elapsed)))}")

    # ==========================================
    # [4] 학습 완료 후 처리
    # ==========================================

    end_time = time.time()
    elapsed_time = end_time - total_start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\n총 학습 시간: {int(hours)}시간 {int(minutes)}분 {int(seconds)}초")

    # ----------------------------------------
    # [4-1] 학습 곡선 시각화
    # ----------------------------------------
    # 최소 손실 및 해당 에폭 계산
    best_loss = min(history['train_loss'])
    best_epoch = history['train_loss'].index(best_loss) + 1

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'],
             'b-', linewidth=2, label='Training Loss')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'],
             'g-', linewidth=2, label='Validation Loss')
    plt.axhline(y=best_loss, color='r', linestyle='--',
                label=f'Best Train Loss: {best_loss:.8f} (Epoch {best_epoch})')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_plot_path = os.path.join(result_dir, "training_loss_curve.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"✅ 손실 그래프 저장: {loss_plot_path}")

    # ----------------------------------------
    # [4-2] 최종 모델 평가 (통합 함수 사용)
    # ----------------------------------------
    from evaluate import evaluate_model
    evaluate_model(model, train_loader, test_loader, device, result_dir)

    print(f"=== 학습 완료: {result_dir} 에 결과 저장됨 ===")
