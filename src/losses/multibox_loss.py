import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.bbox import BBox


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, iou_threshold, neg_pos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio

    def _init_targets(self, batch_size, num_anchors, device):
        conf_t = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=device)
        target_boxes = torch.zeros(
            batch_size, num_anchors, 4, dtype=torch.float32, device=device
        )
        valid_anchor_mask = torch.ones(
            batch_size, num_anchors, dtype=torch.bool, device=device
        )
        return conf_t, target_boxes, valid_anchor_mask

    def _apply_brain_mask(
        self, anchors, brain_masks, valid_anchor_mask, batch_size, num_anchors
    ):
        if brain_masks is not None:
            with torch.no_grad():
                anchor_centers = anchors[:, :2]
                grid = anchor_centers.view(1, num_anchors, 1, 2) * 2 - 1
                grid = grid.expand(batch_size, -1, -1, -1)

                brain_masks = brain_masks[:, 0:1, :, :]

                sampled_mask = F.grid_sample(brain_masks, grid, align_corners=False)
                sampled_mask = sampled_mask.view(batch_size, num_anchors)
                valid_anchor_mask[:] = sampled_mask > 0.01

    def _match_targets(
        self,
        anchors,
        gt_bboxes,
        gt_labels,
        conf_t,
        target_boxes,
        valid_anchor_mask,
        batch_size,
        device,
    ):
        for b in range(batch_size):
            if len(gt_bboxes[b]) == 0:
                continue

            gt_boxes_b = gt_bboxes[b].to(device)
            gt_labels_b = gt_labels[b].to(device)

            ious = BBox.jaccard(gt_boxes_b, anchors)
            values, gt_indices_max = ious.max(dim=0)

            pos_mask_b = values > self.iou_threshold
            best_val, best_idx = ious.max(dim=1)
            pos_mask_b[best_idx] = True

            if pos_mask_b.any():
                conf_t[b, pos_mask_b] = gt_labels_b[gt_indices_max[pos_mask_b]]
                target_boxes[b, pos_mask_b] = gt_boxes_b[gt_indices_max[pos_mask_b]]
                valid_anchor_mask[b, pos_mask_b] = True

    def _compute_loc_loss(
        self, pred_loc, target_boxes, anchors, pos_mask, batch_size, device
    ):
        loc_loss = torch.tensor(0.0, device=device)
        if pos_mask.any():
            expanded_anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)
            pos_target_boxes = target_boxes[pos_mask]
            pos_anchors = expanded_anchors[pos_mask]
            pos_pred_loc = pred_loc[pos_mask].float()

            pos_targets_encoded = BBox.encode(
                pos_target_boxes, pos_anchors, variances=[0.1, 0.2]
            )
            loc_loss = F.smooth_l1_loss(
                pos_pred_loc, pos_targets_encoded, reduction="sum"
            )
        return loc_loss

    def _compute_cls_loss(
        self,
        pred_score,
        conf_t,
        valid_anchor_mask,
        pos_mask,
        batch_size,
        num_anchors,
    ):
        conf_p = pred_score.view(-1, self.num_classes).float()
        conf_t_flat = conf_t.view(-1)

        cls_loss_all = F.cross_entropy(conf_p, conf_t_flat, reduction="none")
        cls_loss_all = cls_loss_all.view(batch_size, num_anchors)
        cls_loss_all = cls_loss_all * valid_anchor_mask.float()

        pos_mask_float = pos_mask.float()
        neg_mask_float = 1.0 - pos_mask_float

        num_pos = pos_mask.sum(dim=1, keepdim=True)
        pos_loss = (cls_loss_all * pos_mask_float).sum()

        loss_neg = cls_loss_all * neg_mask_float
        _, loss_idx = loss_neg.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, min=100)
        neg_mask = idx_rank < num_neg.expand_as(idx_rank)
        neg_loss = (loss_neg * neg_mask.float()).sum()

        num_pos_sum = num_pos.sum().float()
        num_pos_safe = torch.clamp(num_pos_sum, min=1.0)

        return (pos_loss + neg_loss) / num_pos_safe

    def forward(
        self,
        pred_loc,
        pred_score,
        anchors,
        gt_bboxes,
        gt_labels,
        brain_masks=None,
    ):
        batch_size = pred_loc.size(0)
        num_anchors = pred_loc.size(1)
        device = pred_loc.device

        conf_t, target_boxes, valid_anchor_mask = self._init_targets(
            batch_size, num_anchors, device
        )

        self._apply_brain_mask(
            anchors, brain_masks, valid_anchor_mask, batch_size, num_anchors
        )

        self._match_targets(
            anchors,
            gt_bboxes,
            gt_labels,
            conf_t,
            target_boxes,
            valid_anchor_mask,
            batch_size,
            device,
        )

        pos_mask = conf_t > 0

        loc_loss = self._compute_loc_loss(
            pred_loc, target_boxes, anchors, pos_mask, batch_size, device
        )

        total_cls_loss = self._compute_cls_loss(
            pred_score,
            conf_t,
            valid_anchor_mask,
            pos_mask,
            batch_size,
            num_anchors,
        )

        num_pos_sum = pos_mask.sum().float()
        num_pos_safe = torch.clamp(num_pos_sum, min=1.0)
        total_loc_loss = loc_loss / num_pos_safe

        loss = total_cls_loss + total_loc_loss
        return loss, total_cls_loss, total_loc_loss
