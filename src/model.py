import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights
import math
import config


class AnchorGenerator:
    def __init__(self, feature_maps, scales, ratios):
        self.default_boxes = self._generate(feature_maps, scales, ratios)

    def _generate(self, feature_maps, scales, ratios):
        anchors = []

        for idx, (h, w) in enumerate(feature_maps):
            s_k, s_k_prime = scales[idx]
            layer_ratios = ratios[idx]

            for y in range(h):
                for x in range(w):
                    cx = (x + 0.5) / w
                    cy = (y + 0.5) / h

                    # 1. Small Square (Ratio 1)
                    anchors.append([cx, cy, s_k, s_k])

                    # 2. Big Square (Ratio 1)
                    s_prime = math.sqrt(s_k * s_k_prime)
                    anchors.append([cx, cy, s_prime, s_prime])

                    # 3. Other Aspect Ratios
                    for r in layer_ratios:
                        if r == 1:
                            continue

                        w_a = s_k * math.sqrt(r)
                        h_a = s_k / math.sqrt(r)
                        anchors.append([cx, cy, w_a, h_a])

        return torch.tensor(anchors, dtype=torch.float32, device=config.DEVICE)


class SSDHead(nn.Module):
    def __init__(self, in_channels, num_ancors, num_classes):
        super(SSDHead, self).__init__()

        self.loc = nn.Conv2d(in_channels, num_ancors * 4, kernel_size=3, padding=1)
        self.conf = nn.Conv2d(
            in_channels, num_ancors * num_classes, kernel_size=3, padding=1
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.loc.weight)
        nn.init.constant_(self.loc.bias, 0)

        nn.init.xavier_uniform_(self.conf.weight)
        nn.init.constant_(self.conf.bias, 0)

    def forward(self, x):
        return self.loc(x), self.conf(x)


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


class SSD_FE(nn.Module):
    def __init__(self, num_classes):
        super(SSD_FE, self).__init__()
        self.num_classes = num_classes
        self.feature_maps = [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]
        self.scales = [
            [0.015, 0.030],
            [0.030, 0.045],
            [0.045, 0.060],
            [0.060, 0.075],
            [0.075, 0.090],
        ]
        self.ratios = [
            [1, 2, 0.5],
            [1, 2, 3, 0.5, 0.333],
            [1, 2, 3, 0.5, 0.333],
            [1, 2, 3, 0.5, 0.333],
            [1, 2, 0.5],
        ]

        self.anchors = AnchorGenerator(self.feature_maps, self.scales, self.ratios)
        self.default_boxes = self.anchors.default_boxes

        self.vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        self.fe = FELayer()
        self.vgg16[30] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Conv6
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.relu6 = nn.ReLU(inplace=True)

        # Conv7
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)

        # Extra Block 1
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.relu8_1 = nn.ReLU(inplace=True)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.relu8_2 = nn.ReLU(inplace=True)

        # Extra Block 2
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.relu9_1 = nn.ReLU(inplace=True)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu9_2 = nn.ReLU(inplace=True)

        # Extra Block 3
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.relu10_1 = nn.ReLU(inplace=True)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu10_2 = nn.ReLU(inplace=True)

        self.heads = nn.ModuleList(
            [
                SSDHead(512, 4, self.num_classes),  # conv4_3
                SSDHead(1024, 6, self.num_classes),  # conv7
                SSDHead(512, 6, self.num_classes),  # conv8_2
                SSDHead(256, 6, self.num_classes),  # conv9_2
                SSDHead(256, 4, self.num_classes),  # conv10_2
            ]
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.constant_(self.conv6.bias, 0)

        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.constant_(self.conv7.bias, 0)

        nn.init.xavier_uniform_(self.conv8_1.weight)
        nn.init.constant_(self.conv8_1.bias, 0)

        nn.init.xavier_uniform_(self.conv8_2.weight)
        nn.init.constant_(self.conv8_2.bias, 0)

        nn.init.xavier_uniform_(self.conv9_1.weight)
        nn.init.constant_(self.conv9_1.bias, 0)

        nn.init.xavier_uniform_(self.conv10_1.weight)
        nn.init.constant_(self.conv10_1.bias, 0)

        nn.init.xavier_uniform_(self.conv10_2.weight)
        nn.init.constant_(self.conv10_2.bias, 0)

    def forward(self, x, gt_image=None, gt_mask=None):
        anchor_features = []

        x = self.vgg16[:23](x)

        # Feature Enhancement 적용
        if gt_mask is not None and gt_image is not None:
            x = self.fe(x, gt_image, gt_mask)
        anchor_features.append(x)

        x = self.vgg16[23:](x)

        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        anchor_features.append(x)

        x = self.relu8_1(self.conv8_1(x))
        x = self.relu8_2(self.conv8_2(x))
        anchor_features.append(x)

        x = self.relu9_1(self.conv9_1(x))
        x = self.relu9_2(self.conv9_2(x))
        anchor_features.append(x)

        x = self.relu10_1(self.conv10_1(x))
        x = self.relu10_2(self.conv10_2(x))
        anchor_features.append(x)

        # Heads
        loc_preds = []
        conf_preds = []

        for head, feature in zip(self.heads, anchor_features):
            loc, conf = head(feature)

            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc.view(loc.size(0), -1, 4))

            conf = conf.permute(0, 2, 3, 1).contiguous()
            conf_preds.append(conf.view(conf.size(0), -1, self.num_classes))

        loc = torch.cat(loc_preds, 1)
        conf = torch.cat(conf_preds, 1)

        # Return pre-generated anchors
        if self.default_boxes.device != x.device:
            self.default_boxes = self.default_boxes.to(x.device)

        return loc, conf, self.default_boxes
