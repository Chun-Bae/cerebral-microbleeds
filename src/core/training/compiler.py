import torch
import config
from src.models import SSD_FE, SSD_FE_V1, SSD_FE_V2, SSD_FE_V3, SSD_FE_V4, SSD_FE_V5, SSD_FE_V6, SSD_FE_V7
from src.losses import MultiBoxLoss


def compile_model(model_name="SSD_FE"):
    """
    학습에 필요한 모델 객체, 손실 함수(Criterion), 옵티마이저를 설정하고 반환합니다.
    """
    # 1. 모델 할당
    if model_name == "SSD_FE_V1":
        model = SSD_FE_V1(num_classes=2).to(config.DEVICE)
    elif model_name == "SSD_FE_V2":
        model = SSD_FE_V2(num_classes=2).to(config.DEVICE)
    elif model_name == "SSD_FE_V3":
        model = SSD_FE_V3(num_classes=2).to(config.DEVICE)
    elif model_name == "SSD_FE_V4":
        model = SSD_FE_V4(num_classes=2).to(config.DEVICE)
    elif model_name == "SSD_FE_V5":
        model = SSD_FE_V5(num_classes=2).to(config.DEVICE)
    elif model_name == "SSD_FE_V6":
        model = SSD_FE_V6(num_classes=2).to(config.DEVICE)
    elif model_name == "SSD_FE_V7":
        model = SSD_FE_V7(num_classes=2).to(config.DEVICE)
    else:
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
