import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torchvision.ops import nms
from src.utils import decode, jaccard
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset import CMBsDatasetLMDB, collate_fn, BBOX_JSON_PATH, normalize_16bit
from src.model import SSD_FE
from src.utils import Logger
import sys
import platform

# 한글 폰트 설정 (NanumGothic이 없으면 기본 폰트 사용)
import matplotlib.font_manager as fm

import config


available_fonts = [f.name for f in fm.fontManager.ttflist]
if "NanumGothic" in available_fonts:
    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False


def get_ignored_fn(pred_loc, pred_score, anchors, gt_boxes):
    """
    GT와 매칭되었지만(위치 정확), 점수가 낮아서(배경 오분류) 버려진 앵커들을 찾음
    -> 각 GT별로 가장 IoU가 높은 앵커 1개씩을 추적하여,
       그 앵커의 점수가 CONF_THRESH 미만이면 '파란색 박스' 후보로 리턴
    """
    if len(gt_boxes) == 0:
        return np.array([]), np.array([])

    # 1. 모든 앵커 Decode (시각화용이므로 속도보다 정확성)
    # 메모리 절약을 위해 모든 앵커를 다 하기보다, IoU 계산을 먼저 Default Anchor로 약식 진행할 수도 있으나,
    # 정확한 '자리 잡음'을 보려면 Decode를 해야 함. (6만개 정도는 괜찮음)

    # Softmax
    probs = F.softmax(pred_score, dim=-1)
    scores = probs[:, 1]  # (N_anchors,)

    # 2. Decode
    decoded_boxes = decode(pred_loc, anchors)  # (N_anchors, 4) [cx,cy,w,h]

    # [cx, cy, w, h] -> [x1, y1, x2, y2]
    decoded_boxes_xyxy = torch.zeros_like(decoded_boxes)
    decoded_boxes_xyxy[:, 0] = decoded_boxes[:, 0] - decoded_boxes[:, 2] / 2
    decoded_boxes_xyxy[:, 1] = decoded_boxes[:, 1] - decoded_boxes[:, 3] / 2
    decoded_boxes_xyxy[:, 2] = decoded_boxes[:, 0] + decoded_boxes[:, 2] / 2
    decoded_boxes_xyxy[:, 3] = decoded_boxes[:, 1] + decoded_boxes[:, 3] / 2

    # GT도 변환
    gt_boxes_xyxy = torch.zeros_like(gt_boxes)
    gt_boxes_xyxy[:, 0] = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
    gt_boxes_xyxy[:, 1] = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
    gt_boxes_xyxy[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
    gt_boxes_xyxy[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3] / 2

    # 3. IoU 계산 (GT vs All Decoded Anchors)
    # jaccard는 [x1,y1,x2,y2] 기준 동작 (utils.py 확인 결과 jaccard 내부에서 변환하지만,
    # utils.jaccard는 [cx,cy,w,h] 입력을 가정함.
    # 따라서 decoded_boxes(cx,cy,w,h)를 그대로 넣어야 함.)

    ious = jaccard(gt_boxes, decoded_boxes)  # (num_gt, num_anchors)

    ignored_boxes = []
    ignored_scores = []

    # 각 GT별로 가장 잘 맞는 앵커 확인
    for i in range(len(gt_boxes)):
        # i번째 GT와 가장 IoU가 높은 앵커 인덱스
        best_iou_val, best_idx = ious[i].max(dim=0)

        # IoU가 어느정도(0.3) 이상인 경우에만 "자리를 잡았다"고 인정
        if best_iou_val > 0.3:
            # 해당 앵커의 점수가 임계값보다 낮으면 -> 억울한 FN (파란색 대상)
            if scores[best_idx] < config.CONF_THRESH:
                ignored_boxes.append(decoded_boxes[best_idx].cpu().numpy())
                ignored_scores.append(scores[best_idx].item())

    return np.array(ignored_boxes), np.array(ignored_scores)


def post_process(
    pred_locs,
    pred_scores,
    anchors,
    mask_map=None,  # [New] 배경 필터링용 마스크 (H, W)
):
    """
    단일 이미지 후처리
    CONF_THRESH, NMS_THRESH 사용
    """
    # 1. Softmax → 병변 클래스 확률만 추출
    probs = F.softmax(pred_scores, dim=-1)
    scores = probs[:, 1]

    # 2. Confidence 필터링
    mask = scores > config.CONF_THRESH
    # 병변이 없으면 예측 x
    if mask.sum() == 0:
        return torch.zeros((0, 4), device=anchors.device), torch.zeros(
            (0,), device=anchors.device
        )

    # mask 되는 샘플은 다같이 지워주기
    filtered_scores = scores[mask]
    filtered_locs = pred_locs[mask]
    filter_anchors = anchors[mask]

    # 3. Offset → 실제 좌표 디코딩
    # decode: 앵커 + offset → 실제 bbox [cx, cy, w, h]
    boxes = decode(filtered_locs, filter_anchors)

    # --- [Background Filtering] Pre-NMS ---
    # 중요: NMS 전에 배경 박스를 지워야, 진짜 병변이 NMS로 인해 삭제되는 것을 방지함.
    # [Fix] 단순 Image Intensity > 0.05로 필터링하면 어두운 병변(CMB)까지 지워질 수 있음.
    # 따라서 명시적인 ROI Mask(Brain Mask)를 사용해야 함.
    if mask_map is not None:
        H, W = mask_map.shape
        keep_indices = []

        # 앵커 위치 확인을 위해 CPU로 (이미 filtered_anchors가 있음)
        anchors_cpu = filter_anchors.detach().cpu().numpy()

        for idx, anchor in enumerate(anchors_cpu):
            cx, cy, _, _ = anchor
            px = int(cx * W)
            py = int(cy * H)
            px = max(0, min(W - 1, px))
            py = max(0, min(H - 1, py))

            # 뇌 마스크(ROI) 내부인지 확인
            # mask_map은 Binary (0 or 1) 가정 -> 0보다 크면 Brain
            # 만약 mask_map이 이미지 자체라면 위험함. (수정됨)
            if mask_map[py, px] > 0:
                keep_indices.append(idx)

        # 필터링 적용
        if len(keep_indices) != len(boxes):
            keep_indices = torch.as_tensor(
                keep_indices, device=boxes.device, dtype=torch.long
            )
            boxes = boxes[keep_indices]
            filtered_scores = filtered_scores[keep_indices]
            # 만약 다 지워졌다면 빈 텐서 반환
            if len(boxes) == 0:
                return torch.zeros((0, 4), device=anchors.device), torch.zeros(
                    (0,), device=anchors.device
                )

    # 4. [cx, cy, w, h] → [x1, y1, x2, y2] 변환
    boxes_xyxy = torch.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    # 5. NMS (Non-Maximum Suppression)
    # 이제 유령 박스가 사라졌으므로 안전하게 NMS 수행
    keep = nms(boxes_xyxy, filtered_scores, config.NMS_THRESH)
    if keep.size(0) > 200:
        # 점수 순 정렬되어 있다고 가정하거나, 다시 sort
        # nms는 보통 점수 순으로 keep을 반환함 (torchvision 기준)
        keep = keep[:200]

    # 6. 최종 결과 (cx, cy, w, h 유지)
    final_boxes = boxes[keep]
    final_scores = filtered_scores[keep]

    return final_boxes, final_scores


def post_process_batch(
    pred_locs,
    pred_scores,
    anchors,
    roi_masks=None,  # [Fix] Images -> ROI Masks
):
    """
    배치 단위 후처리
    """
    batch_size = pred_locs.size(0)
    all_boxes = []
    all_scores = []

    for b in range(batch_size):
        # 마스크 추출
        mask_map = None
        if roi_masks is not None:
            # (B, 1, H, W) -> (H, W)
            # 이미 Binary(0 or 1)로 로드되어 있음 (dataset.py 참조)
            mask_map = roi_masks[b, 0].cpu().numpy()

        boxes, scores = post_process(
            pred_locs[b],
            pred_scores[b],
            anchors,
            mask_map=mask_map,  # Pass roi_mask
        )
        all_boxes.append(boxes)
        all_scores.append(scores)

    return all_boxes, all_scores


def match_boxes(pred_boxes, gt_boxes):
    """
    예측 vs 정답 매칭 → TP, FP, FN 계산

    Args:
        pred_boxes: (M, 4) - 모델 예측
        gt_boxes: (K, 4) - 정답

    Returns:
        tp: True Positive 개수
        fp: False Positive 개수
        fn: False Negative 개수
    """

    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)  # 모두 놓침

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0  # 모두 오탐

    ious = jaccard(gt_boxes, pred_boxes)

    # 매칭 (Greedy 방식)
    matched_gt = set()
    tp = 0

    for pred_idx in range(len(pred_boxes)):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx in range(len(gt_boxes)):
            if gt_idx in matched_gt:
                continue

            if ious[gt_idx, pred_idx] > best_iou:
                best_iou = ious[gt_idx, pred_idx]
                best_gt_idx = gt_idx

        if best_iou >= config.TEST_IOU_THRESH:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn


def get_all_detections(all_preds):
    """
    AP 계산을 위해 모든 예측 결과를 Score 순으로 정렬하여 반환
    """
    all_detections = []
    # tqdm은 상위 함수에서 제어하거나 여기서는 생략 (너무 자주 출력되면 시끄러움)
    for boxes, scores, img_id in all_preds:
        for box, score in zip(boxes, scores):
            all_detections.append((score, box, img_id))

    # 내림차순 정렬
    all_detections.sort(key=lambda x: x[0], reverse=True)
    return all_detections


def calculate_ap(all_detections, all_gts, iou_threshold=0.3):
    """
    정렬된 detection 리스트를 받아 AP 계산
    """
    # 2. 전체 GT 개수
    total_gt = sum(len(gt) for gt in all_gts.values())

    if total_gt == 0:
        return 0.0

    matched = {img_id: set() for img_id in all_gts.keys()}

    # 3. 하나씩 추가하면서 TP/FP 평가
    tp_list = []
    fp_list = []

    # 이미 정렬된 상태로 들어옴
    for score, box, img_id in all_detections:
        gt_boxes = all_gts.get(img_id, [])

        # GT 아닌데 긍정 예측
        if len(gt_boxes) == 0:
            tp_list.append(0)
            fp_list.append(1)
            continue

        # 가장 IoU 높은 GT 찾기
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            # 이미 매칭
            if gt_idx in matched[img_id]:
                continue

            # 단일 IoU
            iou = jaccard(gt_box.unsqueeze(0), box.unsqueeze(0))[0, 0].item()
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            # TP
            tp_list.append(1)
            fp_list.append(0)
            matched[img_id].add(best_gt_idx)
        else:
            # FP
            tp_list.append(0)
            fp_list.append(1)

    # 4. 누적 TP, FP 계산
    tp_cumsum = torch.cumsum(torch.tensor(tp_list), dim=0)
    fp_cumsum = torch.cumsum(torch.tensor(fp_list), dim=0)

    # Precision, Recall 계산
    total_pred = tp_cumsum + fp_cumsum

    if len(total_pred) > 0 and total_pred[-1] > 0:
        precisions = tp_cumsum / total_pred
    else:
        precisions = torch.zeros_like(tp_cumsum)

    if total_gt > 0:
        recalls = tp_cumsum / total_gt
    else:
        recalls = torch.zeros_like(tp_cumsum)

    # 5. AP 계산 (All-point interpolation)
    precisions = torch.cat([torch.tensor([1.0]), precisions])
    recalls = torch.cat([torch.tensor([0.0]), recalls])

    # Precision을 monotonically decreasing으로 만들기
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 면적 계산
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap.item()


def compute_ap(all_preds, all_gts):
    """
    기존 호환성 유지용 Wrapper (Global TEST_IOU_THRESH 사용)
    """
    all_detections = get_all_detections(all_preds)
    return calculate_ap(all_detections, all_gts, config.TEST_IOU_THRESH)


def plot_froc(fps_per_image, sensitivities, save_path):
    """
    FROC 곡선 그리기 (Style matched to user reference)
    """
    plt.figure(figsize=(8, 6))

    # User style: Line with markers, legend
    plt.plot(
        fps_per_image,
        sensitivities,
        "o-",
        linewidth=1.5,
        markersize=4,
        label="test FROC Curve",
    )

    plt.xlabel("Average False Positives per Image", fontsize=12)
    plt.ylabel(
        "True Positive Rate (Sensitivity)", fontsize=12
    )  # Label matched to image y-axis
    plt.title(
        "FROC Curve (test) - CMB Detection", fontsize=14
    )  # Title matched to image

    # Axis limits considering the data spread
    if len(fps_per_image) > 0:
        plt.xlim([0, max(fps_per_image) * 1.05])
    plt.ylim([0, 1.05])

    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ FROC 곡선 저장: {save_path}")


def plot_confusion_matrix_bar(tp, fp, fn, save_path):
    """
    Confusion Matrix 그리기 (Bar Chart 형태 - 기존 유지)
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # 2x2가 아닌 바 형태로 표시
    categories = [
        "True Positive\n(정탐)",
        "False Positive\n(오탐)",
        "False Negative\n(미탐)",
    ]
    values = [tp, fp, fn]
    colors = ["#4CAF50", "#F44336", "#FF9800"]  # 초록, 빨강, 주황

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=1.2)

    # 값 표시
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Detection Results Help", fontsize=14)

    # Precision, Recall 표시
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    info_text = f"Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}"
    ax.text(
        0.98,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Confusion Matrix (Bar) 저장: {save_path}")


def plot_confusion_matrix_heatmap(tp, fp, fn, save_path):
    """
    2x2 Confusion Matrix 그리기 (Heatmap 형태)
    Object Detection에서는 TN을 정의할 수 없으므로 N/A로 표시
    """
    # 2x2 행렬 데이터: [[TP, FN], [FP, TN=0]]
    # Row: True Class (Positive, Negative)
    # Col: Pred Class (Positive, Negative)
    cm_data = np.array([[tp, fn], [fp, 0]])

    fig, ax = plt.subplots(figsize=(6, 5))

    # 0인 TN 부분을 제외하고 색상 매핑을 위해 마스킹하거나 그냥 그림
    # TN이 0이지만 여기서는 의미없는 0이므로 색상에서 제외하고 싶을 수 있으나,
    # 간단하게 전체를 그린다.
    im = ax.imshow(cm_data, interpolation="nearest", cmap="Blues")

    # 컬러바
    ax.figure.colorbar(im, ax=ax)

    # 축 라벨 설정
    classes_x = ["Predicted CMB", "Predicted Background"]
    classes_y = ["Actual CMB", "Actual Background"]

    ax.set(
        xticks=np.arange(cm_data.shape[1]),
        yticks=np.arange(cm_data.shape[0]),
        xticklabels=classes_x,
        yticklabels=classes_y,
        title="Confusion Matrix (2x2)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # 텍스트 주석
    thresh = cm_data.max() / 2.0
    for i in range(cm_data.shape[0]):
        for j in range(cm_data.shape[1]):
            # 값 텍스트
            if i == 1 and j == 1:
                val_text = "N/A"  # TN 위치
            else:
                val_text = f"{cm_data[i, j]}"

            ax.text(
                j,
                i,
                val_text,
                ha="center",
                va="center",
                color="white" if cm_data[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Confusion Matrix (Heatmap) 저장: {save_path}")


def compute_froc_data(all_preds, all_gts, num_images):
    """
    FROC 곡선용 데이터 계산

    Returns:
        fps_per_image: 이미지당 평균 FP 리스트
        sensitivities: sensitivity 리스트
    """
    # 모든 예측을 confidence 순으로 정렬
    all_detections = []
    for boxes, scores, img_id in tqdm(all_preds, desc="FROC 데이터 수집"):
        for box, score in zip(boxes, scores):
            all_detections.append((score.item(), box, img_id))

    all_detections.sort(key=lambda x: x[0], reverse=True)

    total_gt = sum(len(gt) for gt in all_gts.values())
    matched = {img_id: set() for img_id in all_gts.keys()}

    fps_per_image = []
    sensitivities = []

    tp_count = 0
    fp_count = 0

    for score, box, img_id in tqdm(all_detections, desc="FROC 계산"):
        gt_boxes = all_gts.get(img_id, [])

        # IoU 확인
        is_tp = False
        if len(gt_boxes) > 0:
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched[img_id]:
                    continue
                iou = jaccard(gt_box.unsqueeze(0), box.unsqueeze(0))[0, 0].item()
                if iou >= config.TEST_IOU_THRESH:
                    is_tp = True
                    matched[img_id].add(gt_idx)
                    break

        if is_tp:
            tp_count += 1
        else:
            fp_count += 1

        # 현재 시점의 FP/image, sensitivity
        fps_per_image.append(fp_count / num_images)
        sensitivities.append(tp_count / total_gt if total_gt > 0 else 0)

    return fps_per_image, sensitivities


def visualize_predictions(
    img,
    gt_boxes,
    pred_boxes,
    pred_scores,
    save_path,
    ignored_boxes=None,
    ignored_scores=None,
    iou_threshold=0.3,
    min_box_size=8,
):
    H, W = img.shape[:2]

    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

        # GT 그리기 (초록색)
    for box in gt_boxes:
        cx, cy, w, h = map(float, box)
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)

        # 안전하게 정수 변환 및 범위 제한
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

    # Pred 그리기 (빨간색)
    for box, score in zip(pred_boxes, pred_scores):
        cx, cy, w, h = map(float, box)

        # 유효하지 않은 값(NaN, Inf) 또는 크기가 0인 박스 제외
        if not all(np.isfinite([cx, cy, w, h])) or w <= 0 or h <= 0:
            continue

        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)

        # 최소 크기 보장 및 정수 변환
        try:
            ix1, iy1 = int(max(0, x1)), int(max(0, y1))
            ix2, iy2 = int(min(W - 1, x2)), int(min(H - 1, y2))

            # OpenCV 함수 호출 전 좌표 재검증
            cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (0, 0, 255), 1)
            cv2.putText(
                vis,
                f"{float(score):.2f}",
                (ix1, max(15, iy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
            )
        except (ValueError, OverflowError):
            continue

    # Ignored FN 그리기 (파란색) - 위치는 잡았으나 점수가 낮아 버려진 녀석들
    if ignored_boxes is not None and len(ignored_boxes) > 0:
        for box, score in zip(ignored_boxes, ignored_scores):
            cx, cy, w, h = map(float, box)

            if not all(np.isfinite([cx, cy, w, h])) or w <= 0 or h <= 0:
                continue

            x1 = int((cx - w / 2) * W)
            y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W)
            y2 = int((cy + h / 2) * H)

            try:
                ix1, iy1 = int(max(0, x1)), int(max(0, y1))
                ix2, iy2 = int(min(W - 1, x2)), int(min(H - 1, y2))

                # 파란색 (BGR: 255, 0, 0)
                cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (255, 0, 0), 1)
                # 점수 표시 (파란색)
                cv2.putText(
                    vis,
                    f"{float(score):.2f}",
                    (ix1, max(15, iy2 + 15)),  # 박스 아래쪽에 표시
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 0, 0),
                    1,
                )
            except (ValueError, OverflowError):
                continue

    cv2.imwrite(save_path, vis)


def evaluate(model, testloader, dataset, device, save_dir):
    """
    전체 평가: TP/FP/FN, mAP, FROC, Confusion Matrix, 시각화
    """
    import gc

    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    total_tp, total_fp, total_fn = 0, 0, 0

    # mAP, FROC용 데이터 수집
    all_preds = []  # [(boxes, scores, img_id), ...]
    all_gts = {}  # {img_id: gt_boxes}

    model.eval()
    img_id = 0
    vis_count = 0

    with torch.no_grad():
        pbar = tqdm(testloader, desc="평가+시각화")
        for batch_img, batch_lesion_mask, batch_roi_mask, batch_bboxes in pbar:
            batch_img = batch_img.to(device)
            batch_img = normalize_16bit(batch_img).to(device)

            # 1. 추론
            # model returns 3 items
            pred_locs, pred_scores, anchors = model(batch_img)

            # 2. 후처리
            # [Fix] 이미지 대신 ROI Mask 전달
            pred_boxes_list, pred_scores_list = post_process_batch(
                pred_locs, pred_scores, anchors, roi_masks=batch_roi_mask
            )

            # 3. 배치 내 각 이미지 처리
            for b in range(len(batch_bboxes)):
                pred_boxes = pred_boxes_list[b]
                pred_score = pred_scores_list[b]

                gt_boxes = batch_bboxes[b].to(device)

                # TP, FP, FN
                tp, fp, fn = match_boxes(pred_boxes, gt_boxes)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                # mAP, FROC용 (CPU로 이동하여 저장)
                all_preds.append((pred_boxes.cpu(), pred_score.cpu(), img_id))
                all_gts[img_id] = gt_boxes.cpu()

                # 시각화 저장 (MRI: 16-bit grayscale → 8-bit)
                img_np = batch_img[b].detach().cpu().numpy()
                if len(img_np.shape) == 3:
                    img_np = img_np.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
                    # 3채널(Prev, Curr, Next)이면 중앙(Curr, index 1) 사용
                    if img_np.shape[2] == 3:
                        img_np = img_np[:, :, 1]  # 두 번째 채널(Center) 사용
                    elif img_np.shape[2] == 1:
                        img_np = img_np[:, :, 0]

                    # Normalized (-1~1) → 8-bit (0~255)
                    img_np = ((img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

                gt_np = gt_boxes.cpu().numpy()
                pred_np = (
                    pred_boxes.cpu().numpy() if len(pred_boxes) > 0 else np.array([])
                )
                pred_s_np = (
                    pred_score.cpu().numpy() if len(pred_score) > 0 else np.array([])
                )

                # "배경으로 오분류된 FN" 찾기 (파란색 박스용)
                # 현재 배치의 pred_locs[b], pred_scores[b] 사용
                ign_boxes_np, ign_scores_np = get_ignored_fn(
                    pred_locs[b],
                    pred_scores[b],
                    anchors,
                    gt_boxes,
                )

                # 파일명 결정
                # [Fix] 병변이 있거나(GT) 예측된 병변이 있는 경우만 저장
                if len(gt_boxes) > 0 or len(pred_boxes) > 0:
                    filename = (
                        dataset.idx_to_name[img_id]
                        if hasattr(dataset, "idx_to_name")
                        else f"img_{img_id:05d}.png"
                    )
                    save_path = os.path.join(vis_dir, filename)

                    visualize_predictions(
                        img_np,
                        gt_np,
                        pred_np,
                        pred_s_np,
                        save_path,
                        ignored_boxes=ign_boxes_np,
                        ignored_scores=ign_scores_np,
                    )
                    vis_count += 1

                img_id += 1

            # tqdm 업데이트
            pbar.set_postfix(
                {
                    "TP": total_tp,
                    "FP": total_fp,
                    "FN": total_fn,
                    "pred_boxes": len(pred_boxes),
                    "vis": vis_count,
                }
            )

            # GPU 메모리 정리
            del pred_locs, pred_scores, batch_img
            torch.cuda.empty_cache()

    # 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()

    # 4. 지표 계산
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # 5. AP 계산
    # 5-1. 사용자 지정 IoU (기존 방식)
    # 정렬은 한 번만 수행
    print("\n  📐 AP 계산을 위한 데이터 정렬 중...")
    all_detections = get_all_detections(all_preds)

    ap = calculate_ap(all_detections, all_gts, config.TEST_IOU_THRESH)

    # 5-2. COCO Style mAP (0.5 ~ 0.95, step 0.05)
    coco_thresholds = np.arange(0.5, 1.0, 0.05)
    coco_aps = []

    print("  📐 COCO Style mAP 계산 중 (0.5 ~ 0.95)...")
    for thr in coco_thresholds:
        val = calculate_ap(all_detections, all_gts, thr)
        coco_aps.append(val)

    mAP_coco = np.mean(coco_aps)

    # 6. FROC 데이터 계산 & 플롯
    fps_per_image, sensitivities = compute_froc_data(all_preds, all_gts, img_id)
    if len(fps_per_image) > 0:
        plot_froc(
            fps_per_image, sensitivities, os.path.join(save_dir, "froc_curve.png")
        )

    # 7. Confusion Matrix 플롯
    # (1) 기존 Bar Chart
    plot_confusion_matrix_bar(
        total_tp, total_fp, total_fn, os.path.join(save_dir, "confusion_matrix_bar.png")
    )
    # (2) 2x2 Heatmap
    plot_confusion_matrix_heatmap(
        total_tp, total_fp, total_fn, os.path.join(save_dir, "confusion_matrix.png")
    )

    # 8. 결과 저장
    total_gt_count = total_tp + total_fn
    fp_per_cbm = total_fp / total_gt_count if total_gt_count > 0 else 0.0

    results = {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "FP/CBM": fp_per_cbm,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        f"AP@{config.TEST_IOU_THRESH} (Target)": ap,
        "mAP@[.5:.95]": mAP_coco,
        "AP@0.50": coco_aps[0],
        "AP@0.75": coco_aps[5] if len(coco_aps) > 5 else 0,
    }

    # 로그 출력
    print("\n" + "=" * 50)
    print("  📊 평가 결과")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 50)
    print(f"  📁 시각화 저장: {vis_count}개 → {vis_dir}")
    print("=" * 50)

    # 결과 파일로 저장
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    # 최종 메모리 정리
    del all_preds, all_gts
    gc.collect()

    return results


if __name__ == "__main__":
    import datetime
    import argparse
    from torch.utils.data import Subset

    parser = argparse.ArgumentParser(description="CMB Detection Evaluation")
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Specific patient ID to evaluate (e.g., VK049)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="weights/00009(latest_ssd_fold_0_200).pth",
        help="Path to model weights",
    )
    parser.add_argument(
        "--lmdb_path",
        type=str,
        default="data/lmdb/holdout_test.lmdb",
        help="Path to LMDB dataset",
    )
    args = parser.parse_args()

    # ==================== 설정 ====================
    MODEL_PATH = args.model
    LMDB_PATH = args.lmdb_path
    BATCH_SIZE = 32
    NUM_WORKERS = 8

    # 타임스탬프 폴더 생성
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d(%Hh-%Mm-%Ss)")
    if args.patient:
        SAVE_DIR = os.path.join("results", f"eval_{args.patient}_{timestamp}")
    else:
        SAVE_DIR = os.path.join("results", f"eval_{timestamp}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    # ===============================================

    # 로거 설정

    sys.stdout = Logger(os.path.join(SAVE_DIR, "log.txt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Evaluation Config:")
    print(f"  CONF_THRESH: {config.CONF_THRESH}")
    print(f"  IOU_THRESH:  {config.TEST_IOU_THRESH}")
    print(f"  NMS_THRESH:  {config.NMS_THRESH}")

    # 모델 로드
    model = SSD_FE(num_classes=2).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"✅ 모델 로드: {MODEL_PATH}")

    # 데이터 로드
    dataset = CMBsDatasetLMDB(LMDB_PATH, BBOX_JSON_PATH)

    # 환자 필터링
    if args.patient:
        print(f"🔍 환자 '{args.patient}' 검색 중...")
        target_indices = []
        new_idx_to_name = {}

        # 필터링 및 이름 매핑 재구성
        new_idx = 0
        for i in range(len(dataset)):
            name = dataset.idx_to_name.get(i, "")
            if args.patient in name:
                target_indices.append(i)
                new_idx_to_name[new_idx] = name
                new_idx += 1

        if len(target_indices) == 0:
            print(f"❌ Error: 환자 '{args.patient}' 데이터를 찾을 수 없습니다.")
            sys.exit(1)

        print(f"✅ 환자 '{args.patient}' 데이터 발견: {len(target_indices)}장")

        # Subset 생성
        original_dataset = dataset
        dataset = Subset(original_dataset, target_indices)
        # Subset에 idx_to_name 속성 추가 (evaluate 함수 호환성)
        dataset.idx_to_name = new_idx_to_name

    testloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"✅ 데이터 로드: {len(dataset)} samples")

    # 평가 실행
    results = evaluate(
        model,
        testloader,
        dataset,
        device,
        SAVE_DIR,
    )
