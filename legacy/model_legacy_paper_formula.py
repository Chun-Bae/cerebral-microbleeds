"""
model.py - CMB(뇌미세출혈) 탐지용 SSD-FE 모델 정의

이 모듈은 CMB 탐지를 위한 딥러닝 모델을 정의합니다.
SSD(Single Shot MultiBox Detector) 구조에 FE(Feature Enhancement) 레이어를 추가한 모델입니다.

모델 구조:
1. VGG16 백본: ImageNet 사전학습 가중치 사용
2. FELayer: 병변 영역 특징 강화
3. SSDHead: 분류 및 바운딩 박스 예측
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights
import math


# ==========================================
# VGG16 백본 네트워크
# ==========================================
class VGG16(nn.Module):
    """
    VGG16 특징 추출 네트워크 (백본)

    ImageNet으로 사전학습된 VGG16의 특징 추출 레이어를 사용합니다.
    conv4_3 레이어까지 사용 (30개 레이어)하여 512채널 feature map을 출력합니다.

    입력: (B, 3, 256, 256)
    출력: (B, 512, 16, 16) - 16배 다운샘플링
    """

    def __init__(self):
        super(VGG16, self).__init__()

        # ImageNet 사전학습 가중치 로드
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # conv4_3까지의 레이어만 사용 (인덱스 0~29)
        # 구성: [conv1, conv2, maxpool, conv3, conv4, maxpool, ...]
        self.features = vgg.features[:30]

    def forward(self, x):
        """
        순전파

        Args:
            x: 입력 이미지 텐서 (B, 3, H, W)

        Returns:
            특징 맵 (B, 512, H/16, W/16)
        """
        return self.features(x)


# ==========================================
# Feature Enhancement Layer (FE Layer)
# ==========================================
class FELayer(nn.Module):
    """
    Feature Enhancement Layer (특징 강화 레이어)

    병변 영역의 특징을 강화하여 탐지 성능을 향상시킵니다.
    Ground Truth 마스크와 원본 이미지 정보를 활용하여
    어두운 병변 영역의 특징을 증폭합니다.

    수식:
        M(i,j) = (1 - P(i,j)) / B_mean
        X_enhanced = X + M * X

    여기서:
        - P(i,j): 픽셀 강도값 (0~1 정규화)
        - B_mean: 마스크 영역 내 평균 강도 * 1.5
        - M: Enhancement 마스크
    """

    def __init__(self):
        super(FELayer, self).__init__()

    def forward(self, feature_map, gt_mask, images=None):
        """
        순전파

        Args:
            feature_map: 백본 출력 특징 맵 (B, 512, H, W)
            gt_mask: Ground Truth 마스크 (B, H_orig, W_orig)
            images: 원본 입력 이미지 (B, 3, H_orig, W_orig), 정규화된 상태 [-1, 1]

        Returns:
            enhanced_map: 강화된 특징 맵 (B, 512, H, W)
        """
        device = feature_map.device
        B, C, H, W = feature_map.shape

        # 원본 이미지가 없는 경우 (Fallback)
        # 단순히 마스크를 곱하는 방식으로 대체
        if images is None:
            resized_gt_mask = F.interpolate(
                gt_mask.unsqueeze(1).float(),
                size=feature_map.shape[2:],
                mode="bilinear",
                align_corners=False
            )
            return feature_map * (1 + resized_gt_mask)

        # ----------------------------------------
        # [벡터화된 FE Layer 구현]
        # GPU 텐서 연산으로 효율적 처리
        # ----------------------------------------

        # 1. 입력 이미지 역정규화: [-1, 1] -> [0, 1]
        # 원본 이미지의 첫 번째 채널만 사용 (그레이스케일)
        P = images[:, 0:1, :, :] * 0.5 + 0.5
        P = torch.clamp(P, 0.0, 1.0)  # 범위 제한

        # 2. 마스크 영역 내 평균 강도(B_mean) 계산
        mask_expanded = gt_mask.unsqueeze(1).float()  # (B, 1, H, W)

        # 마스크 내 픽셀값 합계
        sum_P = torch.sum(P * mask_expanded, dim=(2, 3))  # (B, 1)
        # 마스크 내 픽셀 수
        count_P = torch.sum(mask_expanded, dim=(2, 3))    # (B, 1)

        # B_mean = 평균값 * 1.5 (스케일 팩터)
        mean_val = sum_P / (count_P + 1e-8)  # 0으로 나누기 방지
        b_mean = mean_val * 1.5              # (B, 1)

        # 3. Enhancement 마스크 M(i,j) 계산
        # M = (1 - P) / B_mean
        # -> 어두운 픽셀(낮은 P)일수록 M 값이 커짐
        b_mean = b_mean.unsqueeze(2).unsqueeze(
            3)  # (B, 1, 1, 1) for broadcasting
        M = (1.0 - P) / (b_mean + 1e-8)

        # 안정성을 위한 클리핑 (너무 큰 값 방지)
        M = torch.clamp(M, 0.0, 10.0)

        # 마스크 영역만 적용 (배경은 0)
        M = M * mask_expanded

        # 4. 특징 맵 크기로 리사이즈
        resized_M = F.interpolate(
            M,
            size=feature_map.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        # 5. Feature Enhancement 적용
        # X_enhanced = X + M * X = X * (1 + M)
        enhanced_map = feature_map + feature_map * resized_M

        return enhanced_map


# ==========================================
# SSD Detection Head
# ==========================================
class SSDHead(nn.Module):
    """
    SSD Detection Head (탐지 헤드)

    특징 맵으로부터 다음을 예측합니다:
    1. 바운딩 박스 좌표 (Localization)
    2. 객체 클래스 점수 (Classification)

    입력: (B, 512, H, W) - 특징 맵
    출력:
        - locs: (B, 4, H, W) - 바운딩 박스 좌표 [x_min, y_min, x_max, y_max]
        - scores: (B, num_classes, H, W) - 클래스별 점수
    """

    def __init__(self, in_channels, num_classes):
        """
        Args:
            in_channels: 입력 채널 수 (VGG16 백본의 경우 512)
            num_classes: 분류할 클래스 수 (배경 + 병변 = 2)
        """
        super(SSDHead, self).__init__()

        # 바운딩 박스 위치 예측 레이어
        # 출력: 4채널 (x_min, y_min, x_max, y_max)
        self.loc = nn.Conv2d(
            in_channels, 4,
            kernel_size=3, padding=1
        )

        # 객체 분류 예측 레이어
        # 출력: num_classes 채널 (배경/병변 점수)
        self.cls = nn.Conv2d(
            in_channels, num_classes,
            kernel_size=3, padding=1
        )

    def forward(self, x):
        """
        순전파

        Args:
            x: 특징 맵 (B, 512, H, W)

        Returns:
            locs: 바운딩 박스 좌표 예측 (B, 4, H, W)
            scores: 클래스 점수 예측 (B, num_classes, H, W)
        """
        locs = self.loc(x)     # 바운딩 박스 위치 예측
        scores = self.cls(x)   # 클래스 분류 점수
        return locs, scores


# ==========================================
# SSD-FE 통합 모델
# ==========================================
class SSD_FE(nn.Module):
    """
    SSD-FE (Single Shot Detector with Feature Enhancement)

    CMB(뇌미세출혈) 탐지를 위한 통합 모델입니다.
    VGG16 백본 + FE Layer + SSD Head로 구성됩니다.

    구조:
        입력 이미지 (B, 3, 256, 256)
            ↓
        VGG16 백본 → 특징 맵 (B, 512, 16, 16)
            ↓
        FE Layer → 강화된 특징 맵 (학습 시에만 GT 마스크 사용)
            ↓
        SSD Head → 예측 결과
            - locs: 바운딩 박스 (B, 4, 16, 16)
            - scores: 클래스 점수 (B, 2, 16, 16)
    """

    def __init__(self, num_classes):
        """
        Args:
            num_classes: 분류할 클래스 수 (배경 + CMB = 2)
        """
        super(SSD_FE, self).__init__()

        self.backbone = VGG16()      # VGG16 백본 (사전학습 가중치)
        self.fe = FELayer()          # Feature Enhancement Layer
        self.head = SSDHead(
            in_channels=512,
            num_classes=num_classes
        )  # Detection Head

    def forward(self, x, gt_mask=None):
        """
        순전파

        Args:
            x: 입력 이미지 (B, 3, H, W), 정규화됨 [-1, 1]
            gt_mask: Ground Truth 마스크 (B, H, W)
                     - 학습 시: 제공됨 (FE Layer에서 사용)
                     - 추론 시: None (FE Layer 비활성화)

        Returns:
            locs: 바운딩 박스 좌표 예측 (B, 4, H', W')
            scores: 클래스 점수 예측 (B, num_classes, H', W')
        """
        # 1. 백본 특징 추출
        features = self.backbone(x)

        # 2. Feature Enhancement (학습 시에만 적용)
        if gt_mask is not None:
            features = self.fe(features, gt_mask, images=x)

        # 3. Detection Head로 예측
        locs, scores = self.head(features)

        return locs, scores
