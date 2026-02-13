import torch
from src.box_ops import BoxOps

# 1. 상황 설정
# GT 박스 2개 (정답)
gt_boxes = torch.tensor(
    [
        [100, 100, 50, 50],  # 1번 정답 (중심 100,100 / 크기 50)
        [200, 200, 50, 50],  # 2번 정답 (중심 200,200 / 크기 50)
    ],
    dtype=torch.float32,
)

# Pred 박스 3개 (모델 예측)
pred_boxes = torch.tensor(
    [
        [105, 105, 52, 48],  # 1번 예측 (1번 정답이랑 비슷함)
        [190, 190, 60, 60],  # 2번 예측 (2번 정답이랑 겹침)
        [500, 500, 20, 20],  # 3번 예측 (완전 엉뚱한 곳)
    ],
    dtype=torch.float32,
)

print(f"GT 박스 개수: {gt_boxes.shape[0]}개")  # 2
print(f"Pred 박스 개수: {pred_boxes.shape[0]}개")  # 3

# ---------------------------------------------------------
# 2. BoxOps 사용 (한 방에 계산하기)
# ---------------------------------------------------------
# 내부적으로 Broadcasting을 사용해서 (2, 3) 행렬을 만듭니다.
# 행(Row) = GT, 열(Col) = Pred
ious = BoxOps.jaccard(gt_boxes, pred_boxes)

print("\n=== [결과] IoU 행렬 (2 x 3) ===")
print(ious)
# 예상 결과:
# tensor([[0.85xx, 0.0000, 0.0000],  <- GT 1번과 Pred 1,2,3와의 IoU
#         [0.0000, 0.65xx, 0.0000]]) <- GT 2번과 Pred 1,2,3와의 IoU

# ---------------------------------------------------------
# 3. 결과 해석하기 (직관적으로!)
# ---------------------------------------------------------

# Q1. 각 예측 박스가 어떤 GT랑 가장 잘 맞았나? (Dim=0, 세로 방향 압축)
# "Pred 1번은 GT 중 누구랑 친해? Pred 2번은?"
max_iou_per_pred, best_gt_idx = ious.max(dim=0)

print("\n=== [해석] 각 예측 박스의 운명 ===")
for i in range(len(pred_boxes)):
    score = max_iou_per_pred[i].item()
    target_gt = best_gt_idx[i].item()

    if score > 0.5:
        print(f"Pred {i + 1}번: GT {target_gt + 1}번을 잘 찾음! (IoU: {score:.4f})")
    elif score > 0:
        print(f"Pred {i + 1}번: GT {target_gt + 1}번에 걸치긴 함 (IoU: {score:.4f})")
    else:
        print(f"Pred {i + 1}번: 허공을 가름 (IoU: 0.0)")
