import torch
from src.utils.bbox import BBox
import config
from tqdm import tqdm


def match_boxes(pred_boxes, gt_boxes):
    """예측 vs 정답 매칭 → TP, FP, FN 계산"""
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    ious = BBox.jaccard(gt_boxes, pred_boxes)
    matched_gt = set()
    tp = 0

    for pred_idx in range(len(pred_boxes)):
        best_iou, best_gt_idx = 0, -1
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
    """AP, FROC 계산을 위해 각 예측값에 대해 Best IoU와 GT Index를 미리 계산"""
    detection_list = []
    for pred_boxes, pred_scores, img_id in all_preds:
        if len(pred_boxes) == 0:
            continue
        gt_boxes = all_gts.get(img_id, torch.zeros(0, 4))
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

        ious = BBox.jaccard(gt_boxes, pred_boxes)
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
    detection_list.sort(key=lambda x: x["score"], reverse=True)
    return detection_list


def calculate_ap(detection_list, all_gts, iou_threshold=0.3):
    total_gt = sum(len(gt) for gt in all_gts.values())
    if total_gt == 0:
        return 0.0

    matched = {img_id: set() for img_id in all_gts.keys()}
    tp_list, fp_list = [], []

    for item in detection_list:
        if item["gt_idx"] != -1 and item["iou"] >= iou_threshold:
            if item["gt_idx"] not in matched[item["img_id"]]:
                tp_list.append(1)
                fp_list.append(0)
                matched[item["img_id"]].add(item["gt_idx"])
            else:
                tp_list.append(0)
                fp_list.append(1)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cumsum = torch.cumsum(torch.tensor(tp_list), dim=0)
    fp_cumsum = torch.cumsum(torch.tensor(fp_list), dim=0)

    total_pred = tp_cumsum + fp_cumsum
    precisions = (
        tp_cumsum / total_pred
        if len(total_pred) > 0 and total_pred[-1] > 0
        else torch.zeros_like(tp_cumsum)
    )
    recalls = tp_cumsum / total_gt if total_gt > 0 else torch.zeros_like(tp_cumsum)

    precisions = torch.cat([torch.tensor([1.0]), precisions])
    recalls = torch.cat([torch.tensor([0.0]), recalls])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap.item()


def compute_ap(all_preds, all_gts):
    detection_list = precompute_matches(all_preds, all_gts)
    return calculate_ap(detection_list, all_gts, config.TEST_IOU_THRESH)


def compute_froc_data(all_preds, all_gts, num_images):
    detection_list = precompute_matches(all_preds, all_gts)
    total_gt = sum(len(gt) for gt in all_gts.values())
    matched = {img_id: set() for img_id in all_gts.keys()}
    fps_per_image, sensitivities = [], []
    tp_count, fp_count = 0, 0

    for item in tqdm(detection_list, desc="FROC 계산"):
        is_tp = False
        if item["gt_idx"] != -1 and item["iou"] >= config.TEST_IOU_THRESH:
            if item["gt_idx"] not in matched[item["img_id"]]:
                is_tp = True
                matched[item["img_id"]].add(item["gt_idx"])

        if is_tp:
            tp_count += 1
        else:
            fp_count += 1

        fps_per_image.append(fp_count / num_images)
        sensitivities.append(tp_count / total_gt if total_gt > 0 else 0)

    return fps_per_image, sensitivities
