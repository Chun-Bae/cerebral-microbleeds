import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

from .anchors import AnchorGenerator
from .feature_enhancement import FELayer
from .head import SSDHead


class SSD_FE_V2(nn.Module):
    def __init__(self, num_classes):
        super(SSD_FE_V2, self).__init__()
        self.num_classes = num_classes
        # Feature map sizes assuming 512x512 input
        self.feature_maps = [(128, 128), (128, 128), (128, 128)]
        self.scales = [
            [0.015, 0.025],
            [0.035, 0.050],
            [0.070, 0.100],
        ]
        self.ratios = [
            [0.8750, 1.0000, 1.1250],
            [0.8750, 1.0000, 1.1250],
            [0.8750, 1.0000, 1.1250],
        ]

        self.anchors = AnchorGenerator(self.feature_maps, self.scales, self.ratios)
        self.default_boxes = self.anchors.default_boxes

        self.vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        self.fe = FELayer()

        self.heads = nn.ModuleList(
            [
                SSDHead(256, 4, self.num_classes),  # stage1 - conv3_1
                SSDHead(256, 4, self.num_classes),  # stage2 - conv3_2
                SSDHead(256, 4, self.num_classes),  # stage3 - conv3_3
            ]
        )

        self._init_weights()

    def _init_weights(self):
        # Heads are initialized in their own __init__
        pass

    def forward(self, x, gt_image=None, gt_mask=None):
        anchor_features = []

        # Stage 1: conv3_1
        # 0 ~ 11 (ReLU3_1)
        x = self.vgg16[:12](x)
        anchor_features.append(x)

        # Stage 2: conv3_2
        # 12 ~ 13 (ReLU3_2)
        x = self.vgg16[12:14](x)

        # Apply Feature Enhancement
        if gt_mask is not None and gt_image is not None:
            x = self.fe(x, gt_image, gt_mask)

        anchor_features.append(x)

        # Stage 3: conv3_3
        # 14 ~ 15 (ReLU3_3)
        x = self.vgg16[14:16](x)
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
