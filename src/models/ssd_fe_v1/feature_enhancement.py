import torch
import torch.nn as nn
import torch.nn.functional as F


class FELayer(nn.Module):
    def __init__(self, beta=1.5):
        super(FELayer, self).__init__()
        self.beta = beta

    def forward(self, feature_map, gt_image, gt_bboxes):
        B, C, H_f, W_f = feature_map.shape
        device = feature_map.device

        # 마스크 생성은 gradient 계산에서 완전히 분리
        with torch.no_grad():
            enhanced_mask = torch.zeros((B, 1, H_f, W_f), device=device)

            if gt_bboxes is None:
                return feature_map

            # gt_image(Normalized -1~1)를 0~1 범위로 역변환
            img_0to1 = (gt_image + 1.0) * 0.5

            # 1채널(밝기)로 변환 (B, 1, H, W)
            img_gray = img_0to1.mean(dim=1, keepdim=True)
            _, _, H_i, W_i = img_gray.shape

            # Feature Map 크기로 리사이즈 (연산 효율성을 위해 미리 수행)
            img_resized = F.interpolate(img_gray, size=(H_f, W_f), mode="nearest")

            for b in range(B):
                bboxes = gt_bboxes[b]
                if bboxes is None or len(bboxes) == 0:
                    continue

                # 1. 병변 영역의 평균 밝기 계산 (원본 해상도 기준)
                lesion_pixels = []
                for box in bboxes:
                    cx, cy, w, h = box
                    # (0~1) normalized coordinates -> (0~H_i, 0~W_i)
                    x1, y1 = int((cx - w / 2) * W_i), int((cy - h / 2) * H_i)
                    x2, y2 = int((cx + w / 2) * W_i), int((cy + h / 2) * H_i)
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W_i, x2), min(H_i, y2)

                    if x2 > x1 and y2 > y1:
                        area_pixels = img_gray[b, 0, y1:y2, x1:x2].flatten()
                        lesion_pixels.append(area_pixels)

                if len(lesion_pixels) == 0:
                    continue

                lesion_pixels = torch.cat(lesion_pixels)
                region_mean = lesion_pixels.mean().item()

                # 2. B_mean 계산 (β * mean)
                B_mean = self.beta * region_mean
                if B_mean < 1e-6:
                    B_mean = 1e-6

                # 3. 강화 계수 계산: M = 1 - (pixel / B_mean)
                # 어두울수록(pixel이 작을수록) M이 커짐 (최대 1)
                M = (1.0 - (img_resized[b] / B_mean)).clamp(min=0, max=1)

                # 4. Feature Map 해상도에서 BBox에 해당하는 영역만 강화
                for box in bboxes:
                    cx, cy, w, h = box
                    fx1, fy1 = int((cx - w / 2) * W_f), int((cy - h / 2) * H_f)
                    fx2, fy2 = int((cx + w / 2) * W_f), int((cy + h / 2) * H_f)
                    fx1, fy1, fx2, fy2 = (
                        max(0, fx1),
                        max(0, fy1),
                        min(W_f, fx2),
                        min(H_f, fy2),
                    )

                    if fx2 > fx1 and fy2 > fy1:
                        enhanced_mask[b, 0, fy1:fy2, fx1:fx2] = M[0, fy1:fy2, fx1:fx2]

        # 5. 최종 특징 강화: X_new = X * (1 + M)
        enhanced_map = feature_map * (1.0 + enhanced_mask)
        return enhanced_map
