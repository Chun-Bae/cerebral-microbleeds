import time
import torch
import config
import os
from src.datasets import get_transforms
from src.utils.logger import log
from src.core.training.engine.trainer import train_one_epoch
from src.core.training.engine.validator import validate
from src.core.training.checkpoint import save_checkpoint, load_checkpoint


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    fold_idx=0,
    use_k_fold=False,
    model_name="SSD_FE",
    run_name="default",
):
    """
    학습 메인 루프
    1 Epoch 단위로 train_one_epoch과 validate를 번갈아 실행하고 모델을 저장합니다.
    """

    # 시작 로그 출력
    log.info(
        f"\n🚀 시작: {'FOLD ' + str(fold_idx + 1) if use_k_fold else '(Fixed Split)'}"
    )

    # 1. Transform 생성
    train_transform, _ = get_transforms(device)

    # AMP GradScaler 생성
    scaler = torch.amp.GradScaler("cuda")

    # Loss 히스토리 초기화 및 가중치 복원 (있을 경우)
    start_epoch, start_cur_iteration = 1, 0

    # 4. 저장될 / 불러올 가중치 파일 경로 확인
    suffix = f"fold_{fold_idx}" if use_k_fold else "fixed_split"
    expected_weights_path = os.path.join(
        config.WEIGHTS_DIR, f"{model_name}_{run_name}_{suffix}.pth"
    )

    # 명령어로 별도의 weights가 주어졌다면 그것을 최우선으로, 아니라면 기존에 저장된 동일 이름의 파일 경로를 사용
    if os.path.exists(expected_weights_path):
        pretrained_weights = expected_weights_path
        log.info(
            f"동일한 이름의 기존 체크포인트를 발견하여 이어서 학습합니다: {pretrained_weights}"
        )
    else:
        pretrained_weights = None

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

        log.info(
            f"Epoch {epoch} | Iter {cur_iteration}/{max_iterations} - "
            f"Train: {train_loss:.4f} (cls:{train_cls:.4f}, loc:{train_loc:.4f}) | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # 3. 검증
        validation_interval = config.VALIDATION_INTERVAL
        if epoch % validation_interval == 0 or cur_iteration >= max_iterations:
            val_loss, val_cls, val_loc = validate(model, val_loader, criterion, device)

            log.info(f"  └─ Val: {val_loss:.4f} (cls:{val_cls:.4f}, loc:{val_loc:.4f})")
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
            model_name=model_name,
            run_name=run_name,
            use_k_fold=use_k_fold,
        )

        epoch += 1

    log.success("학습 완료!")
