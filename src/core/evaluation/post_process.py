import torch
import torch.nn.functional as F
from torchvision.ops import nms
from src.utils.bbox import BBox
import numpy as np
import config


def get_ignored_fn(pred_loc, pred_score, anchors, gt_boxes):
    """
    GT와 매칭되었지만(위치 정확), 점수가 낮아서(배경 오분류) 버려진 앵커들을 찾음
    """
    if len(gt_boxes) == 0:
        return np.array([]), np.array([])

    probs = F.softmax(pred_score, dim=-1)
    scores = probs[:, 1]

    decoded_boxes = BBox.decode(pred_loc, anchors)
    ious = BBox.jaccard(gt_boxes, decoded_boxes)

    ignored_boxes = []
    ignored_scores = []

    for i in range(len(gt_boxes)):
        best_iou_val, best_idx = ious[i].max(dim=0)
        if best_iou_val > 0.3:
            if scores[best_idx] < config.CONF_THRESH:
                ignored_boxes.append(decoded_boxes[best_idx].cpu().numpy())
                ignored_scores.append(scores[best_idx].item())

    return np.array(ignored_boxes), np.array(ignored_scores)


def post_process(pred_locs, pred_scores, anchors, mask_map=None):
    """단일 이미지 후처리: Confidence 컷 -> NMS"""
    probs = F.softmax(pred_scores, dim=-1)
    scores = probs[:, 1]
    mask = scores > config.CONF_THRESH

    if mask.sum() == 0:
        return torch.zeros((0, 4), device=anchors.device), torch.zeros(
            (0,), device=anchors.device
        )

    filtered_scores = scores[mask]
    filtered_locs = pred_locs[mask]
    filter_anchors = anchors[mask]

    boxes = BBox.decode(filtered_locs, filter_anchors)

    if mask_map is not None:
        H, W = mask_map.shape
        keep_indices = []
        anchors_cpu = filter_anchors.detach().cpu().numpy()

        for idx, anchor in enumerate(anchors_cpu):
            cx, cy, _, _ = anchor
            px = max(0, min(W - 1, int(cx * W)))
            py = max(0, min(H - 1, int(cy * H)))
            if mask_map[py, px] > 0:
                keep_indices.append(idx)

        if len(keep_indices) != len(boxes):
            keep_indices = torch.as_tensor(
                keep_indices, device=boxes.device, dtype=torch.long
            )
            boxes = boxes[keep_indices]
            filtered_scores = filtered_scores[keep_indices]
            if len(boxes) == 0:
                return torch.zeros((0, 4), device=anchors.device), torch.zeros(
                    (0,), device=anchors.device
                )

    boxes_xyxy = torch.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    keep = nms(boxes_xyxy, filtered_scores, config.NMS_THRESH)
    if keep.size(0) > 200:
        keep = keep[:200]

    return boxes[keep], filtered_scores[keep]


def post_process_batch(pred_locs, pred_scores, anchors, roi_masks=None):
    """배치 단위 후처리"""
    batch_size = pred_locs.size(0)
    all_boxes = []
    all_scores = []
    for b in range(batch_size):
        mask_map = roi_masks[b, 0].cpu().numpy() if roi_masks is not None else None
        boxes, scores = post_process(
            pred_locs[b], pred_scores[b], anchors, mask_map=mask_map
        )
        all_boxes.append(boxes)
        all_scores.append(scores)
    return all_boxes, all_scores
