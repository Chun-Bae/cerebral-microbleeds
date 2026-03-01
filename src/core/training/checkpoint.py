import os
import torch
from src.utils.logger import log


def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    cur_iteration,
    train_loss,
    val_loss,
    loss_history,
    fold_idx,
    max_iterations,
    save_dir="weights",
):
    """
    학습 중 모델과 옵티마이저 등 현재 상태를 저장하는 체크포인트 함수
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "loss_history": loss_history,
        "fold_idx": fold_idx,
        "cur_iteration": cur_iteration,
        "config": {
            "max_iterations": max_iterations,
            "learning_rate": optimizer.param_groups[0]["lr"]
            if optimizer.param_groups
            else 0,
            # batch_size is not strictly needed in the checkpoint info unless for logging
        },
    }

    # 최신 덮어쓰기 저장
    save_path = os.path.join(save_dir, f"latest_ssd_fold_{fold_idx}.pth")
    torch.save(checkpoint, save_path)


def load_checkpoint(weights_path, model, optimizer=None, scaler=None):
    """
    저장된 체크포인트를 불러와 모델, 옵티마이저, 스케일러 상태를 복원합니다.
    (Resume 학습 또는 평가 시 사용)
    """
    if not os.path.exists(weights_path):
        log.warning(f"체크포인트 파일을 찾을 수 없습니다: {weights_path}")
        return 1, 0, None  # 기본 epoch 1, iteration 0, loss_history None 반환

    log.info(f"가중치를 로드합니다: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint.get("epoch", 0) + 1  # 마지막 학습 에포크 다음부터 재개
    cur_iteration = checkpoint.get("cur_iteration", 0)
    loss_history = checkpoint.get("loss_history", None)

    log.success(
        f"가중치 성공적으로 로드됨! (이어서 시작할 Epoch: {epoch}, Iter: {cur_iteration})"
    )
    return epoch, cur_iteration, loss_history
