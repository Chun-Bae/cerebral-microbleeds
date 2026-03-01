import torch
import cv2
import numpy as np


class BBox:
    """
    Bounding Box 관련 유틸리티 함수 모음
    """

    @staticmethod
    def intersect(box_a, box_b):
        """
        두 박스 세트의 교집합 영역 계산

        Args:
            box_a (1, 4): 정답(GT) 박스 [[2, 2, 6, 6]]
            box_b (2, 4): 앵커 박스들 [[4, 4, 8, 8], [10, 10, 12, 12]]
        """

        # 1. 오른쪽 위 벽 찾기 (xmax, ymax 중 작은 값)
        # box_a[:, 2:].unsqueeze(1) -> [[[6, 6]]] (1, 1, 2)
        # box_b[:, 2:].unsqueeze(0) -> [[[8, 8], [12, 12]]] (1, 2, 2)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1), box_b[:, 2:].unsqueeze(0))

        # 2. 왼쪽 아래 벽 찾기 (xmin, ymin 중 큰 값)
        # box_a[:, :2].unsqueeze(1) -> [[[2, 2]]] (1, 1, 2)
        # box_b[:, :2].unsqueeze(0) -> [[[4, 4], [10, 10]]] (1, 2, 2)
        min_xy = torch.max(box_a[:, :2].unsqueeze(1), box_b[:, :2].unsqueeze(0))

        # 3. 교집합 가로/세로 길이 계산 (끝점 - 시작점)
        inter = torch.clamp((max_xy - min_xy), min=0)

        # 4. 최종 면적 리턴 (가로 * 세로)
        return inter[:, :, 0] * inter[:, :, 1]

    @staticmethod
    def jaccard(box_a, box_b):
        """
        IoU (Intersection over Union) 계산

        Args:
            box_a (1, 4): 정답 [cx, cy, w, h] -> [[4, 4, 4, 4]]
            box_b (2, 4): 앵커 [[6, 6, 4, 4], [11, 11, 2, 2]]
        """

        # 1. 중심/크기(cx, cy, w, h) 포맷을 좌표(xmin, ymin, xmax, ymax)로 변환
        def to_coords(boxes):
            return torch.cat(
                [boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2],
                dim=1,
            )

        A = to_coords(box_a)
        B = to_coords(box_b)

        # 2. 교집합 면적 계산
        inter = BBox.intersect(A, B)

        # 3. 각 박스의 개별 전체 면적 계산 (가로 * 세로)
        area_a = ((A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])).unsqueeze(1)  # (1, 1)
        area_b = ((B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])).unsqueeze(0)  # (1, 2)

        # 4. 합집합(Union) 면적 계산 (A면적 + B면적 - 교집합면적)
        union = area_a + area_b - inter

        # 5. 최종 IoU 계산 (교집합 / 합집합) (0으로 나누는 것 방지)
        return inter / (union + 1e-6)

    @staticmethod
    def encode(matched, anchors, variances=[0.1, 0.2]):
        """
        Ground Truth 좌표를 앵커 기준의 Offset(locs)으로 인코딩

        Args:
            matched (N, 4): 앵커와 매칭된 실제 정답 박스 [cx, cy, w, h]
            anchors (N, 4): 기준이 되는 앵커 박스 [cx, cy, w, h]
        """

        # 1. 중심점 이동량 계산 (cx, cy)
        # (정답 중심 - 앵커 중심) / 앵커 너비
        g_cxcy = (matched[:, :2] - anchors[:, :2]) / (
            variances[0] * anchors[:, 2:] + 1e-7
        )

        # 2. 너비/높이 변화량 계산 (w, h)
        # 정답 크기를 앵커 크기로 나눈 뒤 로그(log)를 취함. (비율이 0이 되지 않도록 clamp)
        ratio = torch.clamp(matched[:, 2:] / (anchors[:, 2:] + 1e-7), min=1e-6)
        g_wh = torch.log(ratio) / variances[1]

        # 3. 하나로 합치기
        return torch.cat([g_cxcy, g_wh], 1)

    @staticmethod
    def decode(loc, anchors, variances=[0.1, 0.2]):
        """
        인코딩된 locs를 다시 이미지 좌표 [cx, cy, w, h]로 복원

        Args:
            loc (N, 4): 모델이 예측한 수정치
            anchors (N, 4): 기준 앵커 좌표
        """

        # 1. 중심점 복원 (cx, cy)
        offset_limit = 2.0
        cxcy_offset = torch.clamp(loc[:, :2], min=-offset_limit, max=offset_limit)
        res_cxcy = anchors[:, :2] + cxcy_offset * variances[0] * anchors[:, 2:]

        # 2. 크기 변화량 제한
        size_limit = 2.0
        wh_offset = torch.clamp(loc[:, 2:], min=-size_limit, max=size_limit)
        res_wh = anchors[:, 2:] * torch.exp(wh_offset * variances[1])

        return torch.cat((res_cxcy, res_wh), 1)


def masks_to_bboxes(masks):
    """
    증강된 마스크(B, 1, H, W)에서 다시 BBox를 추출
    return: List [Tensor(N, 4), Tensor(M, 4), ...]
    """
    batch_size, _, H, W = masks.shape
    new_bboxes_list = []

    # CPU numpy로 변환 (cv2 사용을 위해)
    masks_np = masks.detach().cpu().numpy()

    for i in range(batch_size):
        mask_np = masks_np[i, 0]  # (H, W)

        # 1. 단순 이진화 (0 or 255)
        binary = (mask_np > 0).astype(np.uint8) * 255

        # 2. 윤곽선 찾기
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bboxes = []
        for cnt in contours:
            # 3. bbox 계산
            x, y, w, h = cv2.boundingRect(cnt)

            if w <= 0 or h <= 0:
                continue

            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H

            bboxes.append([cx, cy, nw, nh])

        if len(bboxes) > 0:
            new_bboxes_list.append(torch.tensor(bboxes, dtype=torch.float32))
        else:
            new_bboxes_list.append(torch.zeros((0, 4), dtype=torch.float32))

    return new_bboxes_list
