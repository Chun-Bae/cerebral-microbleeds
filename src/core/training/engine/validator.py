import torch
from tqdm import tqdm
from src.datasets import normalize_16bit


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """
    학습 중 모델의 일반화 성능을 평가(검증)합니다.
    """
    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0

    pbar = tqdm(val_loader, desc="Validating")
    for batch_img, batch_lesion_mask, batch_roi_mask, batch_bboxes in pbar:
        # GPU
        batch_img = batch_img.to(device)
        batch_roi_mask = batch_roi_mask.to(device)

        # 정규화
        batch_img = normalize_16bit(batch_img)

        # 1. forward
        pred_locs, pred_scores, anchors = model(batch_img)

        # 2. GT 준비
        gt_labels = [
            torch.ones(len(bboxes)).long().to(device) for bboxes in batch_bboxes
        ]

        # 3. Validation Loss 계산
        loss, cls_loss, loc_loss = criterion(
            pred_locs, pred_scores, anchors, batch_bboxes, gt_labels, batch_roi_mask
        )

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_loc_loss += loc_loss.item()

    n = max(1, len(val_loader))
    return total_loss / n, total_cls_loss / n, total_loc_loss / n
