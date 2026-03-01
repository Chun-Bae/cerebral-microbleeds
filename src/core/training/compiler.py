import torch
import config
from src.models import SSD_FE
from src.loss import MultiBoxLoss


def compile_model():
    """
    학습에 필요한 모델 객체, 손실 함수(Criterion), 옵티마이저를 설정하고 반환합니다.
    """
    # 1. 모델 할당
    model = SSD_FE(num_classes=2).to(config.DEVICE)

    # 2. 손실 함수 (Loss)
    criterion = MultiBoxLoss(
        num_classes=2,
        iou_threshold=config.TRAIN_IOU_THRESH,
        neg_pos_ratio=config.NEG_POS_RATIO,
    )

    # 3. 옵티마이저 (Optimizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    return model, criterion, optimizer
