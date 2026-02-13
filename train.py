import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import os
import cv2
import numpy as np
from src.dataset import normalize_16bit, get_transforms

torch.backends.cudnn.benchmark = True


def masks_to_bboxes(masks):
    """
    증강된 마스크(B, 1, H, W)에서 다시 BBox를 추출
    return: List [Tensor(N, 4), Tensor(M, 4), ...]
    """
    batch_size, _, H, W = masks.shape
    new_bboxes_list = []

    # CPU numpy로 변환 (cv2 사용을 위해)
    masks_np = masks.detach().cpu().numpy()

    for i in range(batch_size):
        mask_np = masks_np[i, 0]  # (H, W)

        # 1. 단순 이진화 (0 or 255)
        binary = (mask_np > 0).astype(np.uint8) * 255

        # 2. 윤곽선 찾기
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bboxes = []
        for cnt in contours:
            # 3. bbox 계산
            x, y, w, h = cv2.boundingRect(cnt)

            if w <= 0 or h <= 0:
                continue

            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H

            bboxes.append([cx, cy, nw, nh])

        if len(bboxes) > 0:
            new_bboxes_list.append(torch.tensor(bboxes, dtype=torch.float32))
        else:
            new_bboxes_list.append(torch.zeros((0, 4), dtype=torch.float32))

    return new_bboxes_list


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    transform,
    epoch,
    num_epochs,
    scaler,
):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0

    pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{num_epochs}]")
    for batch_img, batch_lesion_mask, batch_roi_mask, batch_bboxes in pbar:
        # GPU
        batch_img = batch_img.to(device)
        batch_lesion_mask = batch_lesion_mask.to(device)
        batch_roi_mask = batch_roi_mask.to(device)

        # 1. 전처리 및 정규화
        # 이미지, 병변마스크, 뇌마스크를 함께 변환 (3개 리턴)
        batch_img, batch_lesion_mask, batch_roi_mask = transform(
            batch_img, batch_lesion_mask, batch_roi_mask
        )

        # 2. 증강된 마스크에서 BBox 다시 추출
        # Kornia로 인해 위치가 변했을 수 있으므로 마스크에서 역산하여 BBox 업데이트
        batch_bboxes = masks_to_bboxes(batch_lesion_mask)

        # 입력 이미지 정규화
        batch_img = normalize_16bit(batch_img)

        try:
            # 3. forward (AMP autocast)
            with torch.amp.autocast("cuda"):
                # forward(x, gt_image, gt_mask) - 학습 시 FE 적용
                # batch_lesion_mask 증강된 것을 넣어야 함
                pred_locs, pred_scores, anchors = model(
                    batch_img, batch_img, batch_lesion_mask
                )

                # 4. GT 준비 (bboxes and label)
                gt_labels = [
                    torch.ones(len(bboxes)).long().to(device) for bboxes in batch_bboxes
                ]

                # 5. MultiBox Loss 계산
                loss, cls_loss, loc_loss = criterion(
                    pred_locs,
                    pred_scores,
                    anchors,
                    batch_bboxes,
                    gt_labels,
                    batch_roi_mask,
                )

            # 6. Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_loc_loss += loc_loss.item()

        except Exception as e:
            print(f"Error during training step: {e}")
            continue

        # GPU 메모리 사용량 (used/total GB)
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_reserved() / (1024**3)
            gpu_mem_str = f"{gpu_used:.1f}GB"
        else:
            gpu_mem_str = "N/A"

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "cls": f"{cls_loss.item():.4f}",
                "loc": f"{loc_loss.item():.4f}",
                "GPU": gpu_mem_str,
            }
        )

    n = max(1, len(train_loader))
    return total_loss / n, total_cls_loss / n, total_loc_loss / n


def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for batch_img, batch_lesion_mask, batch_roi_mask, batch_bboxes in pbar:
            # GPU
            batch_img = batch_img.to(device)
            batch_roi_mask = batch_roi_mask.to(device)
            # Validation 시에는 증강을 안하므로 batch_bboxes 그대로 사용

            # 정규화
            batch_img = normalize_16bit(batch_img)

            # 1. forward
            # Validation에서는 FE 적용 안 함 (테스트 모드)
            pred_locs, pred_scores, anchors = model(batch_img)

            # 2. GT
            gt_labels = [
                torch.ones(len(bboxes)).long().to(device) for bboxes in batch_bboxes
            ]

            loss, cls_loss, loc_loss = criterion(
                pred_locs, pred_scores, anchors, batch_bboxes, gt_labels, batch_roi_mask
            )

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_loc_loss += loc_loss.item()

    n = max(1, len(val_loader))
    return total_loss / n, total_cls_loss / n, total_loc_loss / n


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    fold_idx=0,
    start_epoch=1,
    loss_history=None,
):
    # 1. Transform 생성
    train_transform, _ = get_transforms(device)

    # AMP GradScaler 생성
    scaler = torch.amp.GradScaler("cuda")

    # Loss 히스토리 초기화
    if loss_history is None:
        loss_history = {
            "train_loss": [],
            "val_loss": [],
            "train_cls_loss": [],
            "train_loc_loss": [],
            "val_cls_loss": [],
            "val_loc_loss": [],
        }

    for epoch in range(start_epoch, num_epochs + 1):
        # 2. 학습 실행
        train_loss, train_cls, train_loc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train_transform,
            epoch,
            num_epochs,
            scaler,
        )

        # 3. 검증 실행 (10 Epoch마다)
        validation_interval = 10
        if epoch % validation_interval == 0 or epoch == num_epochs:
            val_loss, val_cls, val_loc = validate(model, val_loader, criterion, device)

            # 4. 로그 출력
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train: {train_loss:.4f} (cls:{train_cls:.4f}, loc:{train_loc:.4f}) | "
                f"Val: {val_loss:.4f} (cls:{val_cls:.4f}, loc:{val_loc:.4f}) | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
        else:
            val_loss, val_cls, val_loc = 0.0, 0.0, 0.0

        # 3-1. Loss 히스토리에 추가
        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_loss)
        loss_history["train_cls_loss"].append(train_cls)
        loss_history["train_loc_loss"].append(train_loc)
        loss_history["val_cls_loss"].append(val_cls)
        loss_history["val_loc_loss"].append(val_loc)

        # 5. Checkpoint 저장 (매 epoch 최신 상태 덮어쓰기)
        os.makedirs("weights", exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "loss_history": loss_history,
            "fold_idx": fold_idx,
            "config": {
                "num_epochs": num_epochs,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "batch_size": train_loader.batch_size,
            },
        }
        # 항상 최신 저장
        torch.save(checkpoint, f"weights/latest_ssd_fold_{fold_idx}.pth")

    print(f"\n학습 완료!")
