import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import os
import cv2
import numpy as np
from utils import jaccard, encode, decode
from dataset import normalize_16bit, get_transforms


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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Positive 샘플에 대한 가중치 (0.25가 일반적)
        self.gamma = gamma  # 쉬운 샘플을 얼마나 줄일 것인가 (2.0이 일반적)
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, num_classes) - 모델의 예측 logits
        # targets: (N) - 실제 클래스 ID (0: 배경, 1: 병변)

        # Cross Entropy Loss 계산 (reduction='none'으로 각 샘플별 Loss 계산)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Softmax를 통해 예측 확률 pt 계산
        # targets에 해당하는 클래스의 예측 확률을 가져옴
        # log(pt) = -ce_loss 이므로 pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # Focal Loss 계산
        # α * (1 - pt)^γ * CE(pt)

        # alpha factor 적용: target이 1 (positive)이면 alpha, 0 (negative)이면 (1-alpha)
        alpha_factor = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)

        # modulating factor 적용: (1 - pt)^gamma
        focal_loss = alpha_factor * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def ciou_loss(pred_boxes, gt_boxes, eps=1e-7):
    """
    CIoU Loss 계산
    pred_boxes: (N, 4) - [cx, cy, w, h] (decoded)
    gt_boxes: (N, 4) - [cx, cy, w, h]
    """
    # [cx, cy, w, h] -> [x1, y1, x2, y2]
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    gt_x1 = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
    gt_y1 = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
    gt_x2 = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
    gt_y2 = gt_boxes[:, 1] + gt_boxes[:, 3] / 2

    # 1. IoU 계산
    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
    gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
    union_area = pred_area + gt_area - inter_area + eps

    iou = inter_area / union_area

    # 2. 중심점 거리 (DIoU term)
    cw_x1 = torch.min(pred_x1, gt_x1)
    cw_y1 = torch.min(pred_y1, gt_y1)
    cw_x2 = torch.max(pred_x2, gt_x2)
    cw_y2 = torch.max(pred_y2, gt_y2)

    cw_w = (cw_x2 - cw_x1).clamp(min=0)
    cw_h = (cw_y2 - cw_y1).clamp(min=0)
    c2 = cw_w**2 + cw_h**2 + eps  # 대각선 거리 제곱

    rho2 = (pred_boxes[:, 0] - gt_boxes[:, 0]) ** 2 + (
        pred_boxes[:, 1] - gt_boxes[:, 1]
    ) ** 2

    # 3. 종횡비 고려 (CIoU term)
    v = (4 / (math.pi**2)) * torch.pow(
        torch.atan(gt_boxes[:, 2] / (gt_boxes[:, 3] + eps))
        - torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + eps)),
        2,
    )

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2) - (alpha * v)
    loss = 1.0 - ciou
    return loss.sum()


class MultiBoxLoss(nn.Module):
    def __init__(
        self, num_classes=2, iou_threshold=0.5, alpha=0.25, alpha_loc=2.0, gamma=2.0
    ):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.focal_criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction="sum")
        self.alpha_loc = alpha_loc

    def forward(self, pred_loc, pred_score, anchors, gt_bboxes, gt_labels):
        """
        pred_loc: (B, total_anchors, 4)
        pred_score: (B, total_anchors, num_classes)
        anchors: (total_anchors, 4)
        gt_bboxes: list of tensors (B개)
        gt_labels: list of tensors (B개)
        """

        batch_size = pred_loc.size(0)
        num_anchors = anchors.size(0)
        device = pred_loc.device

        # 1. 매칭 (Anchors <-> GT bboxes)
        # 병변이 작아서 IoU 조금만 겹쳐도 Positive로 간주
        conf_t = torch.zeros(batch_size, num_anchors).long().to(device)

        # CIoU Loss를 위해 타겟 좌표를 저장할 텐서 (decoded [cx,cy,w,h])
        target_boxes = torch.zeros(batch_size, num_anchors, 4).to(device)

        for b in range(batch_size):
            # 병변 없는 구역
            if len(gt_bboxes[b]) == 0:
                continue

            # GPU로 이동
            gt_boxes_b = gt_bboxes[b].to(device)
            gt_labels_b = gt_labels[b].to(device)

            # 1. IoU 계산
            ious = jaccard(gt_boxes_b, anchors)

            # Top-K 매칭 (단순화): 각 GT별로 IoU 상위 K개의 앵커 무조건 선택
            # 작은 객체가 IoU가 낮아도 학습 기회를 갖도록 강제함
            K = 9
            _, topk_idx = ious.topk(K, dim=1)  # (num_gt, K)

            # 인덱스 정리
            anchor_indices = topk_idx.view(-1)  # (num_gt * K)
            gt_indices = torch.arange(len(gt_boxes_b)).to(device).repeat_interleave(K)

            # 할당 (나중 GT가 덮어쓰는 방식)
            conf_t[b, anchor_indices] = gt_labels_b[gt_indices]
            target_boxes[b, anchor_indices] = gt_boxes_b[gt_indices]

        # Localization Loss (Positive 앵커)
        pos_mask = conf_t > 0

        if pos_mask.sum() == 0:
            loc_loss = torch.tensor(0.0, device=device)
        else:
            # 1. 예측된 오프셋(pred_loc)을 실제 좌표로 디코딩
            anchors_expanded = anchors.unsqueeze(0).expand_as(pred_loc)

            decoded_pred_boxes = decode(pred_loc[pos_mask], anchors_expanded[pos_mask])

            # 2. CIoU Loss 계산
            loc_loss = ciou_loss(decoded_pred_boxes, target_boxes[pos_mask])

        # 가중치 적용
        loc_loss = self.alpha_loc * loc_loss

        # Classification Loss (모든 앵커에 대해 Focal Loss 적용)
        cls_loss = self.focal_criterion(
            pred_score.view(-1, self.num_classes), conf_t.view(-1)
        )

        # Loss Normalization
        N = max(1, pos_mask.sum().item())

        return (loc_loss + cls_loss) / N, cls_loss / N, loc_loss / N


def train_one_epoch(
    model, train_loader, optimizer, criterion, device, transform, epoch, scaler
):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for i, (batch_img, batch_mask, batch_bboxes) in enumerate(pbar):
        # GPU
        batch_img = batch_img.to(device)
        batch_mask = batch_mask.to(device)

        # 1. 전처리 및 정규화
        # 이미지와 마스크를 함께 변환
        batch_img, batch_mask = transform(batch_img, batch_mask)

        # 2. 증강된 마스크에서 BBox 다시 추출
        # Kornia로 인해 위치가 변했을 수 있으므로 마스크에서 역산
        batch_bboxes = masks_to_bboxes(batch_mask)

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

        batch_img = normalize_16bit(batch_img)

        # 2. forward (AMP autocast)
        with torch.amp.autocast("cuda"):
            pred_locs, pred_scores, anchors = model(batch_img, batch_mask)

            # 3. GT 준비 (bboxes and label)
            # Anchor 별 매칭
            gt_labels = [
                torch.ones(len(bboxes)).long().to(device) for bboxes in batch_bboxes
            ]

            # 4. MultiBox Loss 계산
            loss, cls_loss, loc_loss = criterion(
                pred_locs, pred_scores, anchors, batch_bboxes, gt_labels
            )

        # 5. Backward (AMP GradScaler)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
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

    n = len(train_loader)
    return total_loss / n, total_cls_loss / n, total_loc_loss / n


def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for batch_img, batch_mask, batch_bboxes in pbar:
            # GPU
            batch_img = batch_img.to(device)

            # 1. 정규화
            batch_img = normalize_16bit(batch_img)

            # 2. forward
            pred_locs, pred_scores, anchors = model(batch_img)

            # 3. GT
            gt_labels = [
                torch.ones(len(bboxes)).long().to(device) for bboxes in batch_bboxes
            ]

            loss, cls_loss, loc_loss = criterion(
                pred_locs, pred_scores, anchors, batch_bboxes, gt_labels
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
            scaler,
        )

        # 3. 검증 실행
        val_loss, val_cls, val_loc = validate(model, val_loader, criterion, device)

        # 3-1. Loss 히스토리에 추가
        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_loss)
        loss_history["train_cls_loss"].append(train_cls)
        loss_history["train_loc_loss"].append(train_loc)
        loss_history["val_cls_loss"].append(val_cls)
        loss_history["val_loc_loss"].append(val_loc)

        # 4. 로그 출력
        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train: {train_loss:.4f} (cls:{train_cls:.4f}, loc:{train_loc:.4f}) | "
            f"Val: {val_loss:.4f} (cls:{val_cls:.4f}, loc:{val_loc:.4f})"
        )

        # 5. Checkpoint 저장
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
                "num_workers": train_loader.num_workers,
                "amp_enabled": scaler.is_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
        }
        torch.save(checkpoint, f"weights/latest_ssd_fold_{fold_idx}.pth")
        print(f"  💾 모델 저장: weights/latest_ssd_fold_{fold_idx}.pth (epoch {epoch})")

    print(f"\n학습 완료! Best Val Loss: {best_val_loss:.4f}")
