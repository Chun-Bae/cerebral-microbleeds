"""
model.py - CMB 탐지 모델 정의 (SSD-FE)

SSD(Single Shot MultiBox Detector) 기반 CMB 탐지 모델입니다.
VGG16 백본에 Feature Enhancement Layer를 추가하여 병변 영역을 강화합니다.

구성:
1. VGG16: 사전학습된 VGG16 백본 (특징 추출)
2. FELayer: Feature Enhancement Layer (병변 영역 특징 강화)
3. SSDHead: 탐지 헤드 (위치 예측 + 분류)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights
import math


# ==========================================
# Anchor Generator (다중 객체 탐지용)
# ==========================================
class AnchorGenerator:
    """
    Feature map 위치마다 k개의 anchor box 생성

    CMB 탐지에 맞춘 설정:
    - scales: [0.1, 0.2, 0.4] (작은 병변 크기)
    - ratios: [1.0] (대체로 정사각형)
    """

    def __init__(self, scales=[0.1, 0.2, 0.4], ratios=[1.0]):
        self.scales = scales
        self.ratios = ratios
        self.num_anchors = len(scales) * len(ratios)  # 3개

    def generate(self, feature_size, device):
        """
        Anchor boxes 생성

        Args:
            feature_size: feature map 크기 (예: 16 또는 32)
            device: torch device (cuda/cpu)

        Returns:
            anchors: (H*W*k, 4) - [cx, cy, w, h] 정규화 좌표 (0~1)
        """
        anchors = []

        for y in range(feature_size):
            for x in range(feature_size):
                # 셀 중심 좌표 (정규화)
                cx = (x + 0.5) / feature_size
                cy = (y + 0.5) / feature_size

                for scale in self.scales:
                    for ratio in self.ratios:
                        # width, height 계산
                        w = scale * math.sqrt(ratio)
                        h = scale / math.sqrt(ratio)
                        anchors.append([cx, cy, w, h])

        return torch.tensor(anchors, dtype=torch.float32, device=device)


# ==========================================
# VGG16 백본 (특징 추출 네트워크)
# ==========================================
class VGG16(nn.Module):
    """
    VGG16 특징 추출 네트워크

    ImageNet 사전학습 가중치를 사용하여 이미지에서 특징을 추출합니다.
    conv4_3까지의 레이어만 사용하여 512채널 특징 맵을 출력합니다.

    입력: (B, 3, 256, 256)
    출력: (B, 512, 16, 16)
    """

    def __init__(self):
        super(VGG16, self).__init__()
        # 사전학습된 VGG16 로드
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # conv4_3까지만 사용 (인덱스 0~29)
        self.features = vgg.features[:30]

    def forward(self, x):
        """
        순전파

        Args:
            x: 입력 이미지 (B, 3, H, W)

        Returns:
            특징 맵 (B, 512, H/16, W/16)
        """
        return self.features(x)


# ==========================================
# Feature Enhancement Layer
# ==========================================
class FELayer(nn.Module):
    """
    Feature Enhancement Layer (특징 강화 레이어)

    Ground-Truth 마스크를 사용하여 병변 영역의 특징을 강화합니다.
    강화 공식: X_enhanced = X * (1 + resized_mask)

    - 병변 영역 (mask=1): 특징값 2배로 강화
    - 배경 영역 (mask=0): 원본 유지
    """

    def __init__(self):
        super(FELayer, self).__init__()

    def forward(self, feature_map, gt_mask):
        """
        순전파 - 특징 강화

        Args:
            feature_map: 백본 출력 특징 맵 (B, C, H, W)
            gt_mask: Ground Truth 마스크 (B, H_orig, W_orig)
                     값 범위: 0~255 (병변=255, 배경=0)

        Returns:
            강화된 특징 맵 (B, C, H, W)
        """
        # GT Mask를 float32로 변환하고 0~1 정규화
        gt_mask = gt_mask.float() / 255.0

        # GT Mask를 Feature Map 크기로 다운샘플링
        resized_gt_mask = F.interpolate(
            gt_mask.unsqueeze(1),
            size=feature_map.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        # Feature Enhancement 적용
        # 병변 영역: X * 2, 배경 영역: X * 1
        enhanced_map = feature_map * (1 + resized_gt_mask)

        return enhanced_map


# ==========================================
# SSD Detection Head (Anchor 기반)
# ==========================================
class SSDHead(nn.Module):
    """
    SSD Detection Head (Anchor 기반 탐지 헤드)

    특징 맵으로부터 각 anchor에 대해 다음을 예측합니다:
    1. 바운딩 박스 오프셋 (dx, dy, dw, dh)
    2. 객체 존재 확률 (Sigmoid, 1-class)

    입력: (B, 512, H, W) - 특징 맵
    출력:
        - locs: (B, num_anchors*4, H, W) - 각 anchor의 bbox 오프셋
        - scores: (B, num_anchors, H, W) - 각 anchor의 CMB 확률 (Sigmoid 적용 전)
    """

    def __init__(self, in_channels, num_anchors):
        """
        Args:
            in_channels: 입력 채널 수 (VGG16 백본의 경우 512)
            num_anchors: 위치당 anchor 개수 (예: 3)
        """
        super(SSDHead, self).__init__()

        self.num_anchors = num_anchors

        # 위치 예측 레이어 (Bounding Box Regression)
        # 출력: 4*num_anchors 채널 (각 anchor당 dx, dy, dw, dh)
        self.loc = nn.Conv2d(in_channels, 4 * num_anchors,
                             kernel_size=3, padding=1)

        # 클래스 예측 레이어 (1-class Sigmoid)
        # 출력: num_anchors 채널 (각 anchor당 CMB 확률)
        self.cls = nn.Conv2d(in_channels, num_anchors,
                             kernel_size=3, padding=1)

    def forward(self, x):
        """
        순전파

        Args:
            x: 특징 맵 (B, C, H, W)

        Returns:
            locs: bbox 오프셋 예측 (B, num_anchors*4, H, W)
            scores: CMB 확률 예측 (B, num_anchors, H, W) - Sigmoid 적용 전
        """
        locs = self.loc(x)
        scores = self.cls(x)
        return locs, scores


# ==========================================
# SSD-FE 통합 모델 (Anchor 기반)
# ==========================================
class SSD_FE(nn.Module):
    """
    SSD-FE (SSD with Feature Enhancement) - Anchor 기반 다중 객체 탐지

    CMB 탐지를 위한 통합 모델입니다.

    구성:
    1. VGG16 백본: 특징 추출
    2. FELayer: 병변 영역 특징 강화 (학습 시에만 활성화)
    3. AnchorGenerator: anchor boxes 생성
    4. SSDHead: anchor별 위치 + 분류 예측

    입력:
        - x: 입력 이미지 (B, 3, H, W)
        - gt_mask: Ground Truth 마스크 (학습 시, B, H, W)

    출력:
        - locs: bbox 오프셋 (B, num_anchors*4, H', W')
        - scores: CMB 확률 (B, num_anchors, H', W')
        - anchors: anchor boxes (H'*W'*num_anchors, 4)
    """

    def __init__(self, num_anchors=3, anchor_scales=[0.1, 0.2, 0.4], anchor_ratios=[1.0]):
        """
        Args:
            num_anchors: 위치당 anchor 개수 (scales * ratios)
            anchor_scales: anchor 크기 리스트
            anchor_ratios: anchor 비율 리스트
        """
        super(SSD_FE, self).__init__()

        # 1. VGG16 백본 (사전학습)
        self.backbone = VGG16()

        # 2. Feature Enhancement Layer
        self.fe = FELayer()

        # 3. Anchor Generator
        self.anchor_generator = AnchorGenerator(
            scales=anchor_scales, ratios=anchor_ratios)
        self.num_anchors = self.anchor_generator.num_anchors

        # 4. Detection Head (anchor 기반)
        self.head = SSDHead(in_channels=512, num_anchors=self.num_anchors)

        # Anchor 캐시 (같은 feature size면 재사용)
        self._cached_anchors = None
        self._cached_feature_size = None

    def forward(self, x, gt_mask=None):
        """
        순전파

        Args:
            x: 입력 이미지 (B, 3, H, W), 정규화됨 [-1, 1]
            gt_mask: Ground Truth 마스크 (B, H, W)
                     - 학습 시: 제공됨 (FE Layer에서 사용)
                     - 추론 시: None (FE Layer 비활성화)

        Returns:
            locs: bbox 오프셋 예측 (B, num_anchors*4, H', W')
            scores: CMB 확률 예측 (B, num_anchors, H', W') - Sigmoid 적용 전
            anchors: anchor boxes (H'*W'*num_anchors, 4) - [cx, cy, w, h]
        """
        # 1. 백본 특징 추출
        features = self.backbone(x)

        # 2. Feature Enhancement (학습 시에만 적용)
        if gt_mask is not None:
            features = self.fe(features, gt_mask)

        # 3. Detection Head로 예측
        locs, scores = self.head(features)

        # 4. Anchor 생성 (feature size에 맞춰)
        _, _, H, W = features.shape
        feature_size = H  # 정방형 가정

        # 캐시된 anchor 사용 (같은 크기면)
        if self._cached_feature_size != feature_size or self._cached_anchors is None:
            self._cached_anchors = self.anchor_generator.generate(
                feature_size, x.device)
            self._cached_feature_size = feature_size

        anchors = self._cached_anchors

        return locs, scores, anchors
