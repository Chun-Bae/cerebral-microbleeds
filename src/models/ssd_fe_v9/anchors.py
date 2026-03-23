import torch
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
