import time
import torch
import config
from src.datasets import get_transforms
from src.utils.logger import log
from src.core.training.trainer import train_one_epoch
from src.core.training.validator import validate
from src.core.training.checkpoint import save_checkpoint, load_checkpoint


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    fold_idx=0,
    pretrained_weights=None,
):
    """
    학습 메인 루프
    1 Epoch 단위로 train_one_epoch과 validate를 번갈아 실행하고 모델을 저장합니다.
    """

    # 시작 로그 출력
    log.info(
        f"\n🚀 시작: FOLD {fold_idx + 1 if config.USE_K_FOLD else '(Fixed Split)'}"
    )

    # 1. Transform 생성
    train_transform, _ = get_transforms(device)

    # AMP GradScaler 생성
    scaler = torch.amp.GradScaler(device_type=device)

    # Loss 히스토리 초기화 및 가중치 복원 (있을 경우)
    start_epoch, start_cur_iteration = 1, 0

    if pretrained_weights:
        start_epoch, start_cur_iteration, history = load_checkpoint(
            pretrained_weights, model, optimizer, scaler
        )
        if history is not None:
            loss_history = history
        else:
            loss_history = {
                "train_loss": [],
                "val_loss": [],
                "train_cls_loss": [],
                "train_loc_loss": [],
                "val_cls_loss": [],
                "val_loc_loss": [],
            }
    else:
        loss_history = {
            "train_loss": [],
            "val_loss": [],
            "train_cls_loss": [],
            "train_loc_loss": [],
            "val_cls_loss": [],
            "val_loc_loss": [],
        }

    max_iterations = config.MAX_ITERATIONS

    cur_iteration = (
        start_cur_iteration
        if start_cur_iteration > 0
        else (start_epoch - 1) * len(train_loader)
    )
    epoch = start_epoch

    train_start_time = time.time()
    train_start_step = cur_iteration

    while cur_iteration < max_iterations:
        # 2. 학습 실행
        train_loss, train_cls, train_loc, cur_iteration = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train_transform,
            epoch,
            max_iterations,
            scaler,
            cur_iteration,
            train_start_time,
            train_start_step,
        )

        # 3. 검증
        validation_interval = config.VALIDATION_INTERVAL
        if epoch % validation_interval == 0 or cur_iteration >= max_iterations:
            val_loss, val_cls, val_loc = validate(model, val_loader, criterion, device)

            # 4. 로그 출력
            log.info(
                f"Epoch {epoch} | Iter {cur_iteration}/{max_iterations} - "
                f"Train: {train_loss:.4f} (cls:{train_cls:.4f}, loc:{train_loc:.4f}) | "
                f"Val: {val_loss:.4f} (cls:{val_cls:.4f}, loc:{val_loc:.4f}) | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
        else:
            val_loss, val_cls, val_loc = 0.0, 0.0, 0.0

        # 3-1. Loss 히스토리에 추가
        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_loss)
        loss_history["train_cls_loss"].append(train_cls)
        loss_history["train_loc_loss"].append(train_loc)
        loss_history["val_cls_loss"].append(val_cls)
        loss_history["val_loc_loss"].append(val_loc)

        # 5. Checkpoint 저장
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            cur_iteration=cur_iteration,
            train_loss=train_loss,
            val_loss=val_loss,
            loss_history=loss_history,
            fold_idx=fold_idx,
            max_iterations=max_iterations,
            save_dir=config.WEIGHTS_DIR,
        )

        epoch += 1

    log.success("학습 완료!")
