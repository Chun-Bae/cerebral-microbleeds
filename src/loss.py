import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import encode, jaccard, decode


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction="mean"):
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

        # modulating factor 적용: (1 - pt)^gammaf
        focal_loss = alpha_factor * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


import math


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
        self,
        num_classes,
        iou_threshold=0.5,
        alpha=0.25,
        gamma=2.0,
        neg_pos_ratio=10,
    ):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio

        # Focal Loss
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction="none")

    def forward(
        self,
        pred_loc,
        pred_score,
        anchors,
        gt_bboxes,
        gt_labels,
        brain_masks=None,
    ):
        """
        pred_loc:   (B, num_anchors, 4)
        pred_score: (B, num_anchors, num_classes)
        anchors:    (num_anchors, 4)
        gt_bboxes:  List of Tensors [(N_obj, 4), ...]
        gt_labels:  List of Tensors [(N_obj,), ...]
        brain_masks: (B, 1, H, W) - Brain Mask (0: Background, 1: Brain)
        """
        batch_size = pred_loc.size(0)
        num_anchors = pred_loc.size(1)
        device = pred_loc.device

        # 타겟 텐서 초기화
        conf_t = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=device)
        target_boxes = torch.zeros(
            batch_size, num_anchors, 4, dtype=torch.float32, device=device
        )

        # --- [Background Ignoring logic] ---
        # 검은 배경(Pixel=0)에 있는 앵커는 Loss 계산에서 제외 (Mask Out)
        # 효과: 뇌 바깥의 쉬운 배경(Easy Negative) 학습 방지 -> 뇌 안쪽 학습 집중
        valid_anchor_mask = torch.ones(
            batch_size, num_anchors, dtype=torch.bool, device=device
        )

        if brain_masks is not None:
            with torch.no_grad():
                # 1. Brain Mask는 이미 (B, 1, H, W) 형태의 0/1 텐서임

                # 2. 앵커 중심점(cx, cy) 추출 및 (x, y) 좌표계 변환
                # grid_sample은 [-1, 1] 좌표계를 사용하므로 변환 필요: (0~1) -> (-1~1)
                # cx, cy는 (0~1) 범위임.
                anchor_centers = anchors[:, :2]  # (N, 2)
                grid = anchor_centers.view(1, num_anchors, 1, 2) * 2 - 1  # (1, N, 1, 2)
                grid = grid.expand(batch_size, -1, -1, -1)  # (B, N, 1, 2)

                # 3. 각 앵커 위치의 마스크 값 샘플링
                # brain_masks가 (B, 3, H, W)이면 2번째 채널(Index 1), 1채널이면 Index 0 사용
                if brain_masks.size(1) == 3:
                    brain_masks_1ch = brain_masks[:, 1:2, :, :]
                else:
                    brain_masks_1ch = brain_masks[:, 0:1, :, :]

                # (B, 1, H, W)에서 샘플링 -> (B, 1, N, 1) -> (B, N)
                sampled_mask = F.grid_sample(brain_masks_1ch, grid, align_corners=False)
                sampled_mask = sampled_mask.view(batch_size, num_anchors)

                # 4. 값이 0.01 미만(완전 배경)인 앵커만 무효 처리
                # 0.5로 하면 경계선에 걸친 앵커가 버려질 수 있으므로, 조금이라도 뇌에 걸치면(>0.01) 살림.
                valid_anchor_mask = sampled_mask > 0.01

        from src.utils import (
            jaccard,
        )  # Local import to avoid circular dependency if any

        for b in range(batch_size):
            # 병변 없는 구역 (Negative Sample)
            if len(gt_bboxes[b]) == 0:
                # GT가 없으므로 모두 Background(0)
                # conf_t, target_boxes는 이미 0으로 초기화됨
                # valid_anchor_mask는 위에서 Brain Mask로 초기화됨
                continue

            # GPU로 이동
            gt_boxes_b = gt_bboxes[b].to(device)
            gt_labels_b = gt_labels[b].to(device)

            # 1. IoU 계산
            ious = jaccard(gt_boxes_b, anchors)

            # 각 앵커별로 가장 높은 IoU를 가진 GT 선택 (Max over GTs)
            # values: (num_anchors,), gt_indices_max: (num_anchors)
            values, gt_indices_max = ious.max(dim=0)

            # 1. Positive: IoU > 0.4 (작은 객체니까 0.5 대신 0.4로 완화)
            # 또는 GT별로 가장 높은 IoU를 가진 앵커(Best Match)는 무조건 포함
            pos_mask_b = values > self.iou_threshold

            # (Best Match 보장 로직)
            # 각 GT별로 가장 높은 IoU를 가진 앵커 인덱스를 찾아서 추가
            best_val, best_idx = ious.max(dim=1)
            pos_mask_b[best_idx] = True

            if pos_mask_b.any():
                conf_t[b, pos_mask_b] = gt_labels_b[gt_indices_max[pos_mask_b]]
                target_boxes[b, pos_mask_b] = gt_boxes_b[gt_indices_max[pos_mask_b]]

                # Positive 앵커는 배경 마스크를 뚫고 살아남아야 함
                valid_anchor_mask[b, pos_mask_b] = True

        # -----------------------------------------------------
        # Loss 계산
        # -----------------------------------------------------

        # 1. Localization Loss (Smooth L1)
        # Positive 앵커에 대해서만 계산
        pos_mask = conf_t > 0  # (B, num_anchors)
        num_pos = pos_mask.sum(dim=1, keepdim=True)  # (B, 1)

        # Encode target boxes
        # target_boxes: (B, N, 4) -> xywh format expected by smooth_l1
        # BUT target_boxes above is raw GT boxes (cx, cy, w, h).
        # We need to encode them to offsets relative to anchors before calculating loss.
        # However, MultiBoxLoss implementation usually expects encoded targets OR encodes them here.
        # Let's check original code. Original code used 'encode' function in train? NO, wait.
        # Reading original implementation from train.py...
        pass  # Placeholder for code inspection

        # Re-implementing logic based on my understanding of standard SSD
        # Need to encode first!
        # The logic in original train.py loop:
        # target_boxes[b, pos_mask_b] = gt_boxes_b[gt_indices_max[pos_mask_b]]
        # This assigns RAW boxes. So we must encode them.

        loc_loss = torch.tensor(0.0, device=device)
        if pos_mask.any():
            # (Batch, Num_Anchors, 4)
            # anchors: (Num_Anchors, 4)
            # We need to expand anchors to match batch
            expanded_anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)

            # Select positive samples
            pos_target_boxes = target_boxes[pos_mask]
            pos_anchors = expanded_anchors[pos_mask]
            pos_pred_loc = pred_loc[pos_mask]

            # CIoU Loss (Replaces Smooth L1)
            # 1. Decode predicted locations to boxes
            pos_decoded_pred = decode(pos_pred_loc, pos_anchors)
            # 2. Use decoded targets directly (which are pos_target_boxes)
            #    No, wait. pos_target_boxes are GT boxes (cx, cy, w, h).
            #    pos_decode_pred are also (cx, cy, w, h).
            #    Perfect for CIoU.

            loc_loss = ciou_loss(pos_decoded_pred, pos_target_boxes, eps=1e-7)

        # 2. Classification Loss (Focal Loss)
        # (B, N, num_classes) -> (B*N, num_classes)
        conf_p = pred_score.view(-1, self.num_classes)
        conf_t_flat = conf_t.view(-1)
        valid_mask_flat = valid_anchor_mask.view(-1)

        # 배경이지만 valid하지 않은(뇌 밖의) 앵커는 Loss 계산에서 제외
        # -> valid_mask_flat이 False인 애들은 무시
        # 하지만 Pytorch Loss는 ignore_index를 지원하지 않는 경우가 많음 (Focal Loss Custom 구현 시 처리 필요)
        # 여기서는 valid_mask_flat인 애들만 모아서 계산.

        if valid_mask_flat.sum() > 0:
            conf_p_valid = conf_p[valid_mask_flat]
            conf_t_valid = conf_t_flat[valid_mask_flat]

            # Focal Loss 적용
            # Shape: (N_valid,)
            cls_losses = self.focal_loss(conf_p_valid, conf_t_valid)
        else:
            cls_losses = torch.tensor(0.0, device=device)

        # -----------------------------------------------------
        # Hard Negative Mining (Focal Loss를 쓰더라도 일부 적용 가능, 여기서는 Focal Loss 자체에 맡기거나 병행)
        # Focal Loss는 Easy Negative의 가중치를 줄여주므로 HNM이 필수는 아님.
        # 하지만 Background가 압도적으로 많으므로 HNM을 섞어주면 더 안정적일 수 있음.
        # 여기서는 모든 Valid Anchor에 대해 Focal Loss를 구하고,
        # Background 중 Loss가 높은 상위 K개(Negative Mining)만 역전파하는 방식도 가능하지만,
        # Focal Loss의 철학은 "모든 샘플을 다 쓰되 가중치를 조절"하는 것임.
        #
        # ★ 지금 코드 구조상:
        # valid_anchor_mask로 뇌 영역만 추림 -> 전체 앵커의 30~40% 정도?
        # 그 중에서 Positive는 극소수. Negative는 여전히 많음.
        # Focal Loss가 처리해주길 기대하며 전체 평균(Sum)을 사용?
        #
        # 기존 코드의 HNM 로직을 살려보자.
        # Focal Loss 값을 기준으로 정렬해서 Top-K Negative만 사용.
        # -----------------------------------------------------

        # Loss를 Tensor로 가지고 있어야 함 (reduction='none' in Focal Loss)
        # 위에서 cls_losses는 이미 reduction='none'으로 호출됨 (mean/sum 아님) -> Class init 확인 필요
        # Focal Loss init에서 reduction='none'으로 설정함. OK.

        # cls_losses: (N_valid,)
        # 이걸 다시 (B, N) 형태로 매핑하기엔 valid_mask 때문에 인덱스가 깨짐.
        # 간편하게:
        # 1. 모든 Anchor에 대해 Loss 계산 (invalid는 0으로)
        # 2. HNM 수행

        # 다시 전체에 대해 계산
        cls_loss_all = self.focal_loss(conf_p, conf_t_flat)  # (B*N,)
        cls_loss_all = cls_loss_all.view(batch_size, num_anchors)  # (B, N)

        # Invalid Anchor는 Loss 0으로 마스킹
        cls_loss_all = cls_loss_all * valid_anchor_mask.float()

        # Positive와 Negative 분리
        pos_mask_float = pos_mask.float()
        neg_mask_float = 1.0 - pos_mask_float  # (B, N)

        # 각 이미지별 Positive 개수
        num_pos = pos_mask.sum(dim=1, keepdim=True)  # (B, 1)

        # Positive Loss (얘네는 무조건 포함)
        pos_loss = (cls_loss_all * pos_mask_float).sum()

        # Negative Loss Mining
        loss_neg = cls_loss_all * neg_mask_float  # (B, N) - Positive 자리는 0

        # 내림차순 정렬 (높은 Loss = Hard Negative가 앞쪽으로)
        _, loss_idx = loss_neg.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        # 가져올 Negative 개수: Positive * neg_pos_ratio
        # 만약 Positive가 0개(Negative Sample)면 최소 100개를 학습하여
        # "아무것도 없는 이미지"도 배경이라고 확실히 학습시킴.
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, min=100)  # (B, 1)

        # 등수(idx_rank)가 num_neg보다 작은 애들(Top-K)만 True
        neg_mask = idx_rank < num_neg.expand_as(idx_rank)

        # 최종 Negative Loss
        neg_loss = (loss_neg * neg_mask.float()).sum()

        # 최종 Loss 합산
        # Positive가 하나도 없는 배치가 있을 수 있으므로(전체 0), 분모 clamp 필요
        num_pos_sum = num_pos.sum().float()
        num_pos_safe = torch.clamp(num_pos_sum, min=1.0)

        total_cls_loss = (pos_loss + neg_loss) / num_pos_safe
        total_loc_loss = loc_loss / num_pos_safe

        loss = total_cls_loss + total_loc_loss

        return loss, total_cls_loss, total_loc_loss
