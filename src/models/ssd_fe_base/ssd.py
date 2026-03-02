import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

from .anchors import AnchorGenerator
from .feature_enhancement import FELayer
from .head import SSDHead


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

    def forward(self, x, gt_image=None, gt_bboxes=None):
        anchor_features = []

        x = self.vgg16[:23](x)

        # Feature Enhancement 적용
        if gt_bboxes is not None and gt_image is not None:
            x = self.fe(x, gt_image, gt_bboxes)
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
