import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torchvision.ops import nms
from src.box_ops import BoxOps
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset import CMBsDatasetLMDB, collate_fn, BBOX_JSON_PATH, normalize_16bit
from src.model import SSD_FE
from src.logger import Logger
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
    decoded_boxes = BoxOps.decode(pred_loc, anchors)  # (N_anchors, 4) [cx,cy,w,h]

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

    ious = BoxOps.jaccard(gt_boxes, decoded_boxes)  # (num_gt, num_anchors)

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
    boxes = BoxOps.decode(filtered_locs, filter_anchors)

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

    ious = BoxOps.jaccard(gt_boxes, pred_boxes)

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


def precompute_matches(all_preds, all_gts):
    """
    AP, FROC 계산을 위해 각 예측값에 대해 Best IoU와 GT Index를 미리 계산
    Iterating one-by-one is slow, so we batch process per image.
    """
    detection_list = []

    # all_preds: list of (pred_boxes, pred_scores, img_id)
    for pred_boxes, pred_scores, img_id in all_preds:
        if len(pred_boxes) == 0:
            continue

        gt_boxes = all_gts.get(img_id, torch.zeros(0, 4))

        # 예측은 있고 GT가 없는 경우: 무조건 FP (IoU=0, gt_idx=-1)
        if len(gt_boxes) == 0:
            for i in range(len(pred_scores)):
                detection_list.append(
                    {
                        "score": pred_scores[i].item(),
                        "iou": 0.0,
                        "gt_idx": -1,
                        "img_id": img_id,
                    }
                )
            continue

        # Batch IoU Calculation
        # ious: (num_gt, num_pred)
        ious = BoxOps.jaccard(gt_boxes, pred_boxes)

        # 각 예측 박스별로 가장 높은 IoU를 가진 GT 찾기
        # best_ious, best_gt_indices: (num_pred,)
        best_ious, best_gt_indices = ious.max(dim=0)

        for i in range(len(pred_scores)):
            detection_list.append(
                {
                    "score": pred_scores[i].item(),
                    "iou": best_ious[i].item(),
                    "gt_idx": best_gt_indices[i].item(),
                    "img_id": img_id,
                }
            )

    # 점수 기준 내림차순 정렬
    detection_list.sort(key=lambda x: x["score"], reverse=True)
    return detection_list


def calculate_ap(detection_list, all_gts, iou_threshold=0.3):
    """
    최적화된 AP 계산: 미리 계산된 매칭 정보 사용
    """
    total_gt = sum(len(gt) for gt in all_gts.values())
    if total_gt == 0:
        return 0.0

    matched = {img_id: set() for img_id in all_gts.keys()}
    tp_list = []
    fp_list = []

    for item in detection_list:
        img_id = item["img_id"]
        gt_idx = item["gt_idx"]
        iou = item["iou"]

        # 이미 매칭되었거나 IoU가 낮은 경우 -> FP
        # (혹은 GT가 없는 이미지였던 경우)
        if gt_idx != -1 and iou >= iou_threshold:
            if gt_idx not in matched[img_id]:
                # TP
                tp_list.append(1)
                fp_list.append(0)
                matched[img_id].add(gt_idx)
            else:
                # 이미 발견된 GT -> 중복 검출 (FP)
                tp_list.append(0)
                fp_list.append(1)
        else:
            # IoU 미달 또는 GT 없음 -> FP
            tp_list.append(0)
            fp_list.append(1)

    # 4. 누적 TP, FP 계산 및 AP 도출 (이하 동일)
    tp_cumsum = torch.cumsum(torch.tensor(tp_list), dim=0)
    fp_cumsum = torch.cumsum(torch.tensor(fp_list), dim=0)

    total_pred = tp_cumsum + fp_cumsum

    if len(total_pred) > 0 and total_pred[-1] > 0:
        precisions = tp_cumsum / total_pred
    else:
        precisions = torch.zeros_like(tp_cumsum)

    if total_gt > 0:
        recalls = tp_cumsum / total_gt
    else:
        recalls = torch.zeros_like(tp_cumsum)

    precisions = torch.cat([torch.tensor([1.0]), precisions])
    recalls = torch.cat([torch.tensor([0.0]), recalls])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap.item()


def compute_ap(all_preds, all_gts):
    """Wrapper"""
    detection_list = precompute_matches(all_preds, all_gts)
    return calculate_ap(detection_list, all_gts, config.TEST_IOU_THRESH)


# (FROC 그래프 등은 시각화 함수이므로 기존 유지, 데이터 계산 함수만 최적화)
def compute_froc_data(all_preds, all_gts, num_images):
    """
    FROC 곡선용 데이터 계산 (최적화 버전)
    """
    detection_list = precompute_matches(all_preds, all_gts)

    total_gt = sum(len(gt) for gt in all_gts.values())
    matched = {img_id: set() for img_id in all_gts.keys()}

    fps_per_image = []
    sensitivities = []

    tp_count = 0
    fp_count = 0

    for item in tqdm(detection_list, desc="FROC 계산"):
        img_id = item["img_id"]
        gt_idx = item["gt_idx"]
        iou = item["iou"]

        is_tp = False
        if gt_idx != -1 and iou >= config.TEST_IOU_THRESH:
            if gt_idx not in matched[img_id]:
                is_tp = True
                matched[img_id].add(gt_idx)

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
            # 정규화
            batch_img = normalize_16bit(batch_img)

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
        default="weights/latest_ssd_fold_0.pth",
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
