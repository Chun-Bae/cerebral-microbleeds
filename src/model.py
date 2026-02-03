import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights
from torchinfo import summary

import math


class AnchorGenerator:
    def __init__(self, layer_scales, ratios):
        # layer_scales: [[s1], [s2], [s3]] 형태 (레이어별 전담 스케일)
        self.layer_scales = layer_scales
        self.ratios = ratios

    def generate(self, feature_sizes, device):
        anchors = []
        # 각 피처맵 크기와 해당 레이어용 스케일을 매칭
        for f_size, scales in zip(feature_sizes, self.layer_scales):
            for y in range(f_size):
                for x in range(f_size):
                    cx = (x + 0.5) / f_size
                    cy = (y + 0.5) / f_size

                    for scale in scales:
                        for ratio in self.ratios:
                            w = scale * math.sqrt(ratio)
                            h = scale / math.sqrt(ratio)
                            anchors.append([cx, cy, w, h])

        return torch.tensor(anchors, dtype=torch.float32, device=device)


# SSD Backbone 모델 구현(VGG-16 기반)
# SSD-512 모델의 백본(Backbone)으로 사용할 VGG-16 특징 추출 네트워크
class VGG16(nn.Module):
    def __init__(self, in_channels=3):
        super(VGG16, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        # conv3_3 (512 → 128)
        self.stage1 = vgg[2:16]
        # conv4_3 (128 → 64)
        self.stage2 = vgg[16:23]
        # conv5_3 (64 → 32)
        self.stage3 = vgg[23:30]

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f1, f2, f3


# Feature Enhancement Layer 추가
class FELayer(nn.Module):
    def __init__(self):
        super(FELayer, self).__init__()

    def forward(self, feature_map, gt_mask):
        # GT Mask를 Feature Map 크기로 다운샘플링 (Bilinear Interpolation 사용)
        resized_gt_mask = F.interpolate(
            gt_mask,
            size=feature_map.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Ground-Truth 정보 기반으로 특징 강화
        enhanced_map = feature_map * (1 + resized_gt_mask)

        return enhanced_map


# Input Adapter (Stem Block) - Inception Style
# 하나의 이미지에 대해 여러 커널(3x3, 5x5, 7x7)을 동시에 적용하여
# 다양한 크기의 특징(Multi-scale Features)을 포착
class InputAdapter(nn.Module):
    def __init__(self, in_channels=3):
        super(InputAdapter, self).__init__()

        # Branch 1: 작은 특징 (3x3)
        self.branch3 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        # Branch 2: 중간 특징 (5x5)
        self.branch5 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
        # Branch 3: 넓은 문맥 (9x9)
        self.branch7 = nn.Conv2d(in_channels, 16, kernel_size=7, padding=3)
        # Branch 4: 넓은 문맥2 (9x9)
        self.branch9 = nn.Conv2d(in_channels, 16, kernel_size=9, padding=4)

        self.act = nn.SiLU(inplace=True)

        # 초기화
        for m in [self.branch3, self.branch5, self.branch7, self.branch9]:
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b3 = self.act(self.branch3(x))
        b5 = self.act(self.branch5(x))
        b7 = self.act(self.branch7(x))
        b9 = self.act(self.branch9(x))

        # 채널 방향으로 합치기 (Concatenate)
        return torch.cat([b3, b5, b7, b9], dim=1)


# SSD Detection Head
# Bounding Box Regression (Localization)
# Object Classification (CMB 탐지)
class SSDHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(SSDHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # 위치 예측(Bounding Box Regression)
        # (x,y,w,h)
        self.loc = nn.Conv2d(in_channels, 4 * num_anchors, kernel_size=3, padding=1)

        # 클래스 예측(Object Classification)
        # 병변 이진 분류
        self.cls = nn.Conv2d(
            in_channels, num_classes * num_anchors, kernel_size=3, padding=1
        )

        # Head 초기화
        for layer in [self.loc, self.cls]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        locs = self.loc(x)  # Bounding Box 위치 예측
        scores = self.cls(x)  # CMB 분류 점수

        # print(f"SSDHead Output Shapes - locs : {locs.shape}, scores: {scores.shape}")
        return locs, scores

# 최종 SSD-FE 모델
# SSD-512 + FE
class SSD_FE(nn.Module):
    def __init__(self, num_classes):
        super(SSD_FE, self).__init__()

        # CMB 크기(8px~40px)에 맞게 앵커 스케일 조정 (이미지 512 기준)
        # 0.015 -> 7.6px
        # 0.025 -> 12.8px
        # 0.040 -> 20.4px
        # 0.060 -> 30.7px
        # 0.090 -> 46.0px
        self.anchor_scales = [
            [0.015, 0.025],
            [0.035, 0.050],
            [0.070, 0.100],
        ]  # 더 촘촘하게 배치
        self.anchor_ratios = [0.8750, 1.0000, 1.1250]

        # 1. Input Adapter & Backbone
        self.input_adapter = InputAdapter(in_channels=3)
        self.backbone = VGG16(in_channels=64)

        # 2. Feature Enhancement
        self.fe = FELayer()

        # 3. Anchor Generator
        self.anchor_generator = AnchorGenerator(self.anchor_scales, self.anchor_ratios)
        self.num_anchors = len(self.anchor_scales[0]) * len(self.anchor_ratios)

        # 4. Detection Head
        # stage1 - conv3_3(256)
        # stage2 - conv4_3(512)
        # stage3 - conv5_3(512)

        self.head1 = SSDHead(256, self.num_anchors, num_classes)
        self.head2 = SSDHead(512, self.num_anchors, num_classes)
        self.head3 = SSDHead(512, self.num_anchors, num_classes)

    def forward(self, x, gt_mask=None):
        # 0. Input Adapter
        x = self.input_adapter(x)

        # 1. 백본
        f1, f2, f3 = self.backbone(x)

        # 2. Feature Enhancement
        # 학습 시에는 GT Mask 사용, 추론 시에는 Zero Mask (Enhancement Off)
        # Train-Test Mismatch가 있지만, 사용자가 요청한 "Mask 없는 방식"으로 복원
        if gt_mask is None:
            # 배치 사이즈와 장치에 맞는 빈 마스크 생성
            B, _, H, W = x.shape
            gt_mask = torch.zeros((B, 1, H, W), device=x.device)

        f1 = self.fe(f1, gt_mask)
        f2 = self.fe(f2, gt_mask)
        f3 = self.fe(f3, gt_mask)

        # 3. Detection Head
        loc1, cls1 = self.head1(f1)
        loc2, cls2 = self.head2(f2)
        loc3, cls3 = self.head3(f3)

        # 4. 결과 합치기
        def flatten(loc, cls):
            loc = loc.permute(0, 2, 3, 1).contiguous().view(loc.size(0), -1, 4)
            cls = cls.permute(0, 2, 3, 1).contiguous().view(cls.size(0), -1, 2)
            return loc, cls

        l1, c1 = flatten(loc1, cls1)
        l2, c2 = flatten(loc2, cls2)
        l3, c3 = flatten(loc3, cls3)

        all_locs = torch.cat([l1, l2, l3], dim=1)
        all_scores = torch.cat([c1, c2, c3], dim=1)

        # 5. 앵커 생성
        f_sizes = [f1.size(2), f2.size(2), f3.size(2)]
        anchors = self.anchor_generator.generate(f_sizes, x.device)

        return all_locs, all_scores, anchors


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSD_FE(num_classes=2).to(device)

    # 1. 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. 모델 파일 용량 (MB 단위)
    param_size_mb = total_params * 4 / (1024**2)  # float32는 4바이트

    # 3. 더미 입력 및 실행
    x = torch.randn(2, 3, 512, 512).to(device)
    mask = torch.zeros(2, 1, 512, 512).to(device)

    # 메모리 측정 시작 (CUDA 전용)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    locs, scores, anchors = model(x, mask)

    # 4. 출력 결과
    print(f"Device: {device}")
    print("-" * 30)
    print("Model Architecture:")
    print(model)
    print("-" * 30)

    print("Torchinfo Summary:")
    summary(model, input_size=(2, 3, 512, 512), depth=4)
    print("-" * 30)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model File Size: {param_size_mb:.2f} MB")
    print("-" * 30)
    print(f"locs shape: {locs.shape}")
    print(f"scores shape: {scores.shape}")
    print(f"anchors shape: {anchors.shape}")

    print("-" * 30)

    # 5. 예상 VRAM 사용량 (Peak Memory)
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak GPU Memory Usage: {peak_mem:.2f} MB")
