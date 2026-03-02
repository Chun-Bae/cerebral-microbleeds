import torch
import torch.nn as nn


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
