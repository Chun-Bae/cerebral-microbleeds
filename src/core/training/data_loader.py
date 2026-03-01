import os
from torch.utils.data import DataLoader
import config
from src.datasets import CMBsDataset, collate_fn, BBOX_JSON_PATH


def build_dataloaders(use_k_fold=False, fold_idx=0):
    """
    설정에 맞는 Train / Validation 데이터 로더를 생성하여 반환합니다.
    - use_k_fold: True이면 k-fold 경로 사용, False이면 fixed_split 경로 사용
    - fold_idx: k-fold 일 경우의 폴드 번호
    """
    if use_k_fold:
        train_lmdb = os.path.join(config.LMDB_DIR, f"fold_{fold_idx}", "train.lmdb")
        val_lmdb = os.path.join(config.LMDB_DIR, f"fold_{fold_idx}", "test.lmdb")
    else:
        train_lmdb = os.path.join(config.LMDB_DIR, "fixed_split", "train.lmdb")
        val_lmdb = os.path.join(config.LMDB_DIR, "fixed_split", "valid.lmdb")

    train_dataset = CMBsDataset(train_lmdb, BBOX_JSON_PATH, split_type="train")
    val_dataset = CMBsDataset(val_lmdb, BBOX_JSON_PATH, split_type="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
