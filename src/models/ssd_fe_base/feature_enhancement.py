import torch
import torch.nn as nn
import torch.nn.functional as F


class FELayer(nn.Module):
    def __init__(self, beta=1.5):
        super(FELayer, self).__init__()
        self.beta = beta

    def forward(self, feature_map, gt_image, gt_mask):
        B, C, H_f, W_f = feature_map.shape
        device = feature_map.device

        # 마스크 생성은 gradient 계산에서 완전히 분리
        with torch.no_grad():
            enhanced_mask = torch.zeros((B, 1, H_f, W_f), device=device)

            if gt_mask is None:
                return feature_map

            # gt_image(Normalized -1~1)를 0~1 범위로 역변환
            img_0to1 = (gt_image + 1.0) * 0.5

            # 1채널(밝기)로 변환 (B, 1, H, W)
            img_gray = img_0to1.mean(dim=1, keepdim=True)

            # Feature Map 크기로 리사이즈 (연산 효율성을 위해 미리 수행)
            img_resized = F.interpolate(img_gray, size=(H_f, W_f), mode="nearest")
            mask_resized = F.interpolate(
                gt_mask.float(), size=(H_f, W_f), mode="nearest"
            )

            for b in range(B):
                # 해당 배치에 병변이 없으면 스킵
                if gt_mask[b].sum() == 0:
                    continue

                # 1. 병변 영역의 평균 밝기 계산 (원본 해상도에서 정확하게 계산)
                lesion_pixels = img_gray[b][gt_mask[b] > 0]
                if lesion_pixels.numel() == 0:
                    continue
                region_mean = lesion_pixels.mean().item()

                # 2. B_mean 계산 (β * mean)
                B_mean = self.beta * region_mean
                if B_mean < 1e-6:
                    B_mean = 1e-6

                # 3. 강화 계수 계산: M = 1 - (pixel / B_mean)
                # 어두울수록(pixel이 작을수록) M이 커짐 (최대 1)
                M = (1.0 - (img_resized[b] / B_mean)).clamp(min=0, max=1)

                # 4. 마스크 영역에만 강화 적용
                enhanced_mask[b] = M * mask_resized[b]

        # 5. 최종 특징 강화: X_new = X * (1 + M)
        enhanced_map = feature_map * (1.0 + enhanced_mask)
        return enhanced_map
