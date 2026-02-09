import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import os
import cv2
import numpy as np
from src.utils import jaccard, encode, decode
from src.dataset import normalize_16bit, get_transforms
from torch.optim.lr_scheduler import CosineAnnealingLR


def masks_to_bboxes(masks):
    """
    증강된 마스크(B, 1, H, W)에서 다시 BBox를 추출
    return: List [Tensor(N, 4), Tensor(M, 4), ...]
    """
    batch_size, _, H, W = masks.shape
    new_bboxes_list = []

    # CPU numpy로 변환 (cv2 사용을 위해)
    masks_np = masks.detach().cpu().numpy()

    # 디버깅: 전체 배치 중 BBox가 발견된 샘플 수 카운트
    found_count = 0

    for i in range(batch_size):
        mask_np = masks_np[i, 0]  # (H, W)

        # 1. 단순 이진화 (0 or 255)
        binary = (mask_np > 0).astype(np.uint8) * 255

        # 2. 윤곽선 찾기 (Dilation 없음)
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
            found_count += 1
        else:
            new_bboxes_list.append(torch.zeros((0, 4), dtype=torch.float32))

    # 디버깅 로그 (필요시 활성화)
    # if found_count == 0 and batch_size > 0:
    #     print("No bboxes found in batch")

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

    pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/ {num_epochs}]")
    for i, (batch_img, batch_lesion_mask, batch_roi_mask, batch_bboxes) in enumerate(
        pbar
    ):
        # GPU
        batch_img = batch_img.to(device)
        batch_lesion_mask = batch_lesion_mask.to(device)
        batch_roi_mask = batch_roi_mask.to(device)

        # 1. 전처리 및 정규화
        # 이미지, 병변마스크, 뇌마스크를 함께 변환 (3개 리턴)
        # data_keys=["image", "mask", "mask"] 설정 덕분에 brain_mask는 밝기/블러 영향 안 받음
        batch_img, batch_lesion_mask, batch_roi_mask = transform(
            batch_img, batch_lesion_mask, batch_roi_mask
        )

        # 2. 증강된 마스크에서 BBox 다시 추출
        # Kornia로 인해 위치가 변했을 수 있으므로 마스크에서 역산
        batch_bboxes = masks_to_bboxes(batch_lesion_mask)

        # --- [시각화 디버깅] 증강 결과 확인 (100 배치마다 1번) ---
        # if i % 100 == 0:  # 매 배치 실행은 너무 느리므로 빈도 조절
        if False:
            os.makedirs("data/aug_bbox_valid", exist_ok=True)

            # 첫 번째 샘플 가져오기
            viz_img = (
                batch_img[0].detach().cpu().numpy().transpose(1, 2, 0)
            )  # (H, W, 3)
            viz_mask = batch_mask[0, 0].detach().cpu().numpy()  # (H, W)
            viz_bboxes = batch_bboxes[0]  # (N, 4)

            # 1. Image: 정규화 안 된 상태라면 0~1 또는 0~65535일 것임.
            # 하지만 모델 입력 전이라 normalize_16bit 호출 전 단계임 (위 코드 순서상)
            # 만약 normalize_16bit 후라면 denormalize 해야함.
            # 현재 위치는 normalize_16bit 전이므로 값 확인 필요.
            # 보통 Kornia Augmentation 출력은 0~1 사이 (float) 유지함.

            # 0~1 -> 0~255로 변환
            viz_img = (viz_img * 255).astype(np.uint8).copy()
            # BGR로 변환 (OpenCV용)
            viz_img = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)

            # 2. Mask: 빨간색 윤곽선 그리기 (더 선명하게)
            binary_viz_mask = (viz_mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                binary_viz_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(viz_img, contours, -1, (0, 0, 255), 2)

            # 3. BBox: 초록색 사각형
            H, W = viz_img.shape[:2]
            for box in viz_bboxes:
                cx, cy, w, h = box.tolist()
                x1 = int((cx - w / 2) * W)
                y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W)
                y2 = int((cy + h / 2) * H)
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 저장
            cv2.imwrite("data/aug_bbox_valid/augmented_sample.png", viz_img)
        # -----------------------------------------------------

        # 정규화 전 이미지 저장 (FE Layer용)
        batch_img_raw = batch_img.clone()
        batch_img = normalize_16bit(batch_img)

        # 2. forward (AMP autocast)
        with torch.amp.autocast("cuda"):
            # model returns 3 items
            # forward(x, input_image, bboxes_list) - 학습 시 FE 적용
            pred_locs, pred_scores, anchors = model(
                batch_img, batch_img_raw, batch_bboxes
            )

            # 3. GT 준비 (bboxes and label)
            # Anchor 별 매칭
            gt_labels = [
                torch.ones(len(bboxes)).long().to(device) for bboxes in batch_bboxes
            ]

            # 4. MultiBox Loss 계산 (use batch_roi_mask)
            loss, cls_loss, loc_loss = criterion(
                pred_locs,
                pred_scores,
                anchors,
                batch_bboxes,
                gt_labels,
                batch_roi_mask,
            )

        # 5. Backward (AMP GradScaler)
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient Clipping (NaN 방지)
        # 뇌 바깥 영역은 loss를 무시하기 때문에 가중치 조정 시 학습되지 않았던
        # 부분에 대해서 기울기 폭발이 일어나는 것으로 보임.
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_loc_loss += loc_loss.item()

        # GPU 메모리 사용량 (used/total GB)
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_reserved() / (1024**3)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_mem_str = f"{gpu_used:.1f}/{gpu_total:.0f}GB"
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

        # --- [앵커 추적 시각화] ---
        # 매 100 배치(스텝)마다 고정된 샘플에 대한 앵커 변화를 그립니다.
        # 주의: 이 기능을 위해 'fixed_debug_sample'이 필요하므로, train_one_epoch 인자에 추가하거나
        # 전역변수/클래스 멤버로 관리해야 하지만, 여기서는 함수 내에서 static처럼 처리하기 위해
        # 함수 속성(attribute)을 활용하는 트릭을 씁니다.

        if not hasattr(train_one_epoch, "fixed_sample"):
            # 처음 한 번만 고정 샘플을 저장 (현재 배치의 첫 번째 샘플 사용)
            # 조건: 병변이 있어야 함
            if len(batch_bboxes[0]) > 0:
                train_one_epoch.fixed_sample = {
                    "img": batch_img[0].clone().detach(),  # (1, 512, 512) 정규화됨
                    "mask": batch_lesion_mask[0].clone().detach(),  # (1, 512, 512)
                    "bbox": batch_bboxes[0].clone().detach(),  # (N, 4) 정답
                    "brain_mask": batch_roi_mask[0].detach()
                    if "batch_roi_mask" in locals()
                    else None,
                }
                print(f"📌 앵커 추적용 고정 샘플 설정 완료 (Batch {i})")

                # 앵커 고정 (Top-5 IoU 앵커 선택)
                with torch.no_grad():
                    # Note: We pass only image because we just need anchors.
                    # forward(x, input_image=None, bboxes_list=None) -> 테스트 모드
                    _, _, anchors_temp = model(batch_img[0:1])

                gt_box = batch_bboxes[0].to(device)
                ious = jaccard(gt_box, anchors_temp)  # (num_gt, num_anchors)

                # 가장 큰 놈부터 5개 선택
                # GT가 여러개면 그중에서도 best GT 기준
                best_gt_idx = ious.max(dim=1)[0].argmax().item()

                topk_val, topk_idx = ious[best_gt_idx].topk(5)

                train_one_epoch.target_anchor_indices = topk_idx.tolist()
                train_one_epoch.target_anchor_ious = topk_val.tolist()  # IoU 저장
                train_one_epoch.target_gt_idx = best_gt_idx

                print(
                    f"📌 추적할 Top-5 앵커 인덱스: {train_one_epoch.target_anchor_indices}"
                )
                print(
                    f"📌 해당 앵커들의 초기 IoU: {[f'{x:.4f}' for x in train_one_epoch.target_anchor_ious]}"
                )

        # 10 스텝마다 그리기 (샘플이 설정된 경우만)
        if hasattr(train_one_epoch, "fixed_sample") and i % 1000 == 0:
            fs = train_one_epoch.fixed_sample

            with torch.no_grad():
                inp_img = fs["img"].unsqueeze(0).to(device)
                inp_mask = fs["mask"].unsqueeze(0).to(device)
                pred_locs, pred_scores, anchors = model(inp_img, inp_mask)

                # 정답 좌표
                gt_box = fs["bbox"][train_one_epoch.target_gt_idx].to(device)

            # 이미지 복원
            viz_img = (fs["img"].detach().cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
            viz_img = (viz_img * 255).clip(0, 255).astype(np.uint8).copy()
            if viz_img.shape[2] == 1:
                viz_img = cv2.cvtColor(viz_img, cv2.COLOR_GRAY2BGR)
            else:
                viz_img = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)

            H, W = viz_img.shape[:2]

            def draw_box(img, box, color, thickness=1, label=None):
                cx, cy, w, h = box
                x1 = int((cx - w / 2) * W)
                y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W)
                y2 = int((cy + h / 2) * H)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                if label:
                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

            # 🟩 정답 (초록) - 한 번만 그림
            draw_box(viz_img, gt_box.cpu(), (0, 255, 0), 2, "GT")

            # 🔁 Top-5 앵커 Loop + 로그 출력
            log_msg = [f"🔍 [Step {i}] Top-5 Anchors Tracking:"]

            for rank, a_idx in enumerate(train_one_epoch.target_anchor_indices):
                # 1. 앵커 원본
                anchor_box = anchors[a_idx]
                iou_val = train_one_epoch.target_anchor_ious[rank]

                # 2. 예측
                pred_loc = pred_locs[0, a_idx]
                decoded_box = decode(pred_loc.unsqueeze(0), anchor_box.unsqueeze(0))[0]
                pred_conf = torch.sigmoid(pred_scores[0, a_idx, 1]).item()

                # 로그에 추가
                log_msg.append(
                    f"   #{rank + 1}: Conf={pred_conf:.4f}, Initial_IoU={iou_val:.4f}"
                )

                # 그리기
                # 🟥 예측 (빨강)
                label = f"#{rank + 1} Cf:{pred_conf:.2f} IoU:{iou_val:.2f}"
                draw_box(viz_img, decoded_box.cpu(), (0, 0, 255), 2, label)

                # 화살표 (앵커 중심 -> 예측 중심)
                acx, acy = int(anchor_box[0] * W), int(anchor_box[1] * H)
                pcx, pcy = int(decoded_box[0] * W), int(decoded_box[1] * H)
                cv2.arrowedLine(viz_img, (acx, acy), (pcx, pcy), (0, 255, 255), 1)

            # 로그 출력
            print("\n".join(log_msg))

            # 저장
            save_path = f"data/train_debug/track_ep{epoch}_step{i}.png"
            os.makedirs("data/train_debug", exist_ok=True)
            cv2.imwrite(save_path, viz_img)

    n = len(train_loader)
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
            batch_lesion_mask = batch_lesion_mask.to(device)  # Now available
            batch_roi_mask = batch_roi_mask.to(device)

            # (Note: Now we have roi_mask from loader, so no need to create it from image)

            # 정규화
            batch_img = normalize_16bit(batch_img)

            # 2. forward
            # Validation에서는 FE 적용 안 함 (테스트 모드)
            pred_locs, pred_scores, anchors = model(batch_img)

            # 3. GT
            gt_labels = [
                torch.ones(len(bboxes)).long().to(device) for bboxes in batch_bboxes
            ]

            loss, cls_loss, loc_loss = criterion(
                pred_locs, pred_scores, anchors, batch_bboxes, gt_labels, batch_roi_mask
            )

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_loc_loss += loc_loss.item()

    n = len(val_loader)
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

    # Scheduler 생성 (CosineAnnealingLR)
    # T_max: 전체 에포크 수, eta_min: 최소 학습률 (1e-6 정도)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    best_val_loss = float("inf")

    # Loss 히스토리 초기화 (이어서 학습 시 기존 히스토리 유지)
    if loss_history is None:
        loss_history = {
            "train_loss": [],
            "val_loss": [],
            "train_cls_loss": [],
            "train_loc_loss": [],
            "val_cls_loss": [],
            "val_loc_loss": [],
            "train_dice_loss": [],
            "val_dice_loss": [],
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

        # 3. 검증 실행 (속도를 위해 5 Epoch마다 실행)
        if epoch % 200 == 0 or epoch == num_epochs:
            val_loss, val_cls, val_loc = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_cls, val_loc = 0.0, 0.0, 0.0

        # 3-1. Loss 히스토리에 추가
        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_loss)
        loss_history["train_cls_loss"].append(train_cls)
        loss_history["train_loc_loss"].append(train_loc)
        loss_history["val_cls_loss"].append(val_cls)
        loss_history["val_loc_loss"].append(val_loc)

        # 4. 로그 출력
        if epoch % 200 == 0 or epoch == num_epochs:
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train: {train_loss:.4f} (cls:{train_cls:.4f}, loc:{train_loc:.4f}) | "
                f"Val: {val_loss:.4f} (cls:{val_cls:.4f}, loc:{val_loc:.4f})"
            )
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train: {train_loss:.4f} (cls:{train_cls:.4f}, loc:{train_loc:.4f}) | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        # 스케줄러 업데이트
        scheduler.step()

        # 5. Checkpoint 저장
        os.makedirs("weights", exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),  # 스케줄러 상태 저장
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "loss_history": loss_history,
            "fold_idx": fold_idx,
            "config": {
                "num_epochs": num_epochs,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "batch_size": train_loader.batch_size,
                "num_workers": train_loader.num_workers,
                "amp_enabled": scaler.is_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
        }
        torch.save(checkpoint, f"weights/latest_ssd_fold_{fold_idx}.pth")
        print(f"  💾 모델 저장: weights/latest_ssd_fold_{fold_idx}.pth (epoch {epoch})")

    print(f"\n학습 완료! Best Val Loss: {best_val_loss:.4f}")
