import time
import torch
from tqdm import tqdm
from src.datasets import normalize_16bit, get_transforms
from src.utils.logger import log
from src.utils.bbox import masks_to_bboxes
import config


def _adjust_learning_rate(optimizer, cur_iteration):
    """현재 이터레이션에 맞게 Learning Rate를 조정합니다."""
    if cur_iteration < 80000:
        lr = 1e-3
    elif cur_iteration < 100000:
        lr = 1e-4
    else:
        lr = 1e-5

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def _forward_backward_step(
    model,
    criterion,
    optimizer,
    scaler,
    batch_img,
    batch_bboxes,
    batch_roi_mask,
    device,
):
    """단일 미니배치에 대한 Forward & Backward 핵심 연산 수행"""
    # 3. forward (AMP autocast)
    device_str = device.type if hasattr(device, "type") else "cuda"
    with torch.amp.autocast(device_type=device_str):
        pred_locs, pred_scores, anchors = model(batch_img, batch_img, batch_bboxes)

    # Convert to strictly float32 and ensure no existing NaN/Inf poisons the loss
    pred_locs = torch.nan_to_num(
        pred_locs.float(), nan=0.0, posinf=65500.0, neginf=-65500.0
    )
    pred_scores = torch.nan_to_num(
        pred_scores.float(), nan=0.0, posinf=65500.0, neginf=-65500.0
    )

    # 4. GT 준비 (bboxes and label)
    gt_labels = [torch.ones(len(bboxes)).long().to(device) for bboxes in batch_bboxes]

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

    # Gradient Clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    scaler.step(optimizer)
    scaler.update()

    return loss.item(), cls_loss.item(), loc_loss.item()


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    transform,
    epoch,
    max_iterations,
    scaler,
    cur_iteration,
    train_start_time,
    train_start_step,
):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        train_loader, desc=f"[Epoch {epoch}] Iter {cur_iteration}/{max_iterations}"
    )
    for batch_img, batch_lesion_mask, batch_roi_mask, batch_bboxes in pbar:
        # lr scheduler
        lr = _adjust_learning_rate(optimizer, cur_iteration)

        # GPU
        batch_img = batch_img.to(device)
        batch_lesion_mask = batch_lesion_mask.to(device)
        batch_roi_mask = batch_roi_mask.to(device)

        # 1. 전처리 및 정규화
        batch_img, batch_lesion_mask, batch_roi_mask = transform(
            batch_img, batch_lesion_mask, batch_roi_mask
        )

        # 2. 증강된 마스크에서 BBox 다시 추출
        batch_bboxes = masks_to_bboxes(batch_lesion_mask)

        # 입력 이미지 정규화
        batch_img = normalize_16bit(batch_img)

        try:
            # 3. 모델 전파 및 역전파
            loss_val, cls_val, loc_val = _forward_backward_step(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                batch_img=batch_img,
                batch_bboxes=batch_bboxes,
                batch_roi_mask=batch_roi_mask,
                device=device,
            )

            total_loss += loss_val
            total_cls_loss += cls_val
            total_loc_loss += loc_val

        except Exception as e:
            log.error(f"Error during training step: {e}")
            continue

        # GPU 메모리 사용량 (used/total GB)
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_reserved() / (1024**3)
            gpu_mem_str = f"{gpu_used:.1f}GB"
        else:
            gpu_mem_str = "N/A"

        cur_iteration += 1
        num_batches += 1

        elapsed_time = time.time() - train_start_time
        steps_done = cur_iteration - train_start_step
        if steps_done > 0:
            avg_time_per_step = elapsed_time / steps_done
            steps_left = max_iterations - cur_iteration
            eta_seconds = avg_time_per_step * steps_left
            import datetime

            eta_time = datetime.datetime.fromtimestamp(time.time() + eta_seconds)
            eta_str = eta_time.strftime("%m/%d %H:%M")
        else:
            eta_str = "N/A"

        pbar.set_postfix(
            {
                "loss": f"{loss_val:.4f}",
                "cls": f"{cls_val:.4f}",
                "loc": f"{loc_val:.4f}",
                "LR": f"{lr:.0e}",
                "ETA": eta_str,
                "GPU": gpu_mem_str,
            }
        )

        if cur_iteration >= max_iterations:
            break

    n = max(1, num_batches)
    return total_loss / n, total_cls_loss / n, total_loc_loss / n, cur_iteration
