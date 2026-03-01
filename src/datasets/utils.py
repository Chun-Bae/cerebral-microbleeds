import torch
import os
import json
import numpy as np


def load_bbox_json(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"⚠️ BBox JSON 파일 없음: {json_path}")
        return {}


def generate_lesion_mask(image_tensor, bboxes):
    """
    Segmentation Ground Truth 생성 (Binary Mask)
    image_tensor: (1, H, W) - Shape 참고용
    bboxes: (N, 4) [cx, cy, w, h] (0~1 scale)
    """
    _, H, W = image_tensor.shape
    mask = torch.zeros((1, H, W), dtype=torch.float32, device=image_tensor.device)

    for box in bboxes:
        cx, cy, w, h = box
        # 좌표 변환
        x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # 병변 영역 1로 채우기
        mask[:, y1:y2, x1:x2] = 1.0

    return mask


def filter_dataset_by_patient(dataset, patient_id):
    """지정된 환자(patient_id)의 데이터만 남도록 데이터셋을 필터링(Subset)합니다."""
    if not patient_id:
        return dataset

    target_indices = []
    new_idx_to_name = {}
    new_idx = 0

    for i in range(len(dataset)):
        name = dataset.idx_to_name.get(i, "")
        if patient_id in name:
            target_indices.append(i)
            new_idx_to_name[new_idx] = name
            new_idx += 1

    subset_ds = torch.utils.data.Subset(dataset, target_indices)
    subset_ds.idx_to_name = new_idx_to_name
    return subset_ds


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)  # (B, C, H, W)
    lesion_masks = torch.stack([item[1] for item in batch], dim=0)  # (B, 1, H, W)
    roi_masks = torch.stack([item[2] for item in batch], dim=0)  # (B, 1, H, W)
    bboxes = [item[3] for item in batch]  # 리스트

    return images, lesion_masks, roi_masks, bboxes
