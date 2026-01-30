import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from utils import jaccard, encode
from dataset import normalize_16bit, get_transforms


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes=2, iou_threshold=0.35, neg_pos_ratio=3):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio

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
        loc_t = torch.zeros(batch_size, num_anchors, 4).to(device)

        for b in range(batch_size):
            # 병변 없는 구역
            if len(gt_bboxes[b]) == 0:
                continue

            # GPU로 이동
            gt_boxes_b = gt_bboxes[b].to(device)
            gt_labels_b = gt_labels[b].to(device)

            # 1. IoU 계산
            ious = jaccard(gt_boxes_b, anchors)

            # 2. 각 앵커별 가장 높은 IoU인 GT 매칭
            best_gt_iou, best_gt_idx = ious.max(0)

            # 임계값 이상이면 postive 설정
            pos_idx = best_gt_iou > self.iou_threshold
            conf_t[b, pos_idx] = gt_labels_b[best_gt_idx[pos_idx]]

            # 3. 좌표 인코딩
            loc_t[b, pos_idx] = encode(
                gt_boxes_b[best_gt_idx[pos_idx]], anchors[pos_idx]
            )

        # Localization Loss (Positive 앵커)
        pos_mask = conf_t > 0
        num_pos = pos_mask.sum(dim=1, keepdim=True)

        post_idx_expanded = pos_mask.unsqueeze(-1).expand_as(pred_loc)
        loc_loss = F.smooth_l1_loss(
            pred_loc[post_idx_expanded], loc_t[post_idx_expanded], reduction="sum"
        )

        # Hard Negative Mining
        # 배경 앵커 중 손실이 큰 것만 골라 학습하여 class imbalence 문제 해결

        loss_c = F.cross_entropy(
            pred_score.view(-1, self.num_classes), conf_t.view(-1), reduction="none"
        )
        loss_c = loss_c.view(batch_size, num_anchors)
        
        # 정답 위치 예외
        loss_c[pos_mask] = 0

        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(-1)

        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=num_anchors - 1)
        neg_mask = idx_rank < num_neg

        # Classification Loss (Positive + Selected Negative)
        cls_mask = pos_mask | neg_mask
        cls_loss = F.cross_entropy(
            pred_score[cls_mask], conf_t[cls_mask], reduction="sum"
        )

        N = max(1, num_pos.sum().item())

        return (loc_loss + cls_loss) / N, cls_loss / N, loc_loss / N


def train_one_epoch(
    model, train_loader, optimizer, criterion, device, transform, epoch, scaler
):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_img, batch_mask, batch_bboxes in pbar:
        # GPU
        batch_img = batch_img.to(device)
        batch_mask = batch_mask.to(device)

        # 1. 전처리 및 정규화
        batch_img, batch_mask = transform(batch_img, batch_mask)
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
):
    # 1. Transform 생성
    train_transform, _ = get_transforms(device)

    # AMP GradScaler 생성
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
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

        # 4. 로그 출력
        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train: {train_loss:.4f} (cls:{train_cls:.4f}, loc:{train_loc:.4f}) | "
            f"Val: {val_loss:.4f} (cls:{val_cls:.4f}, loc:{val_loc:.4f})"
        )

        # 5. 모델 저장 (매 에폭)
        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), f"weights/latest_ssd_fold_{fold_idx}.pth")
        print(f"  💾 모델 저장: weights/latest_ssd_fold_{fold_idx}.pth")

    print(f"\n학습 완료! Best Val Loss: {best_val_loss:.4f}")
