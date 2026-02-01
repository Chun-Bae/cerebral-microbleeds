import multiprocessing
import datetime
import os
import sys
import torch
from torch.utils.data import DataLoader
from utils import Logger
from convert_nii_folder_to_images import convert_nii_folder_to_images
from dataset import CMBsDatasetLMDB, collate_fn, BBOX_JSON_PATH
from model import SSD_FE
from train import MultiBoxLoss, train
from make_stratified_folds import main as make_folds
from create_lmdb import main as create_lmdb_main
from extract_bboxes import main as extract_bboxes_main

# [설정] Albumentations 업데이트 경고 끄기 (Import 이전에 설정해야 함)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

BATCH_SIZE = 30
NUM_EPOCHS = 1000
NUM_WORKERS = 8
LEARNING_RATE = 1e-5
SPLIT_RATIO = 0.2
IOU_THRESH = 0.35
ALPHA = 0.75
ALPHA_LOC = 5.0
K = 5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SWI_INPUT_DIR = "data/samsung_data/swi"
ROI_INPUT_DIR = "data/samsung_data/roi"
SWI_OUTPUT_DIR = "data/output_images/swi"
ROI_OUTPUT_DIR = "data/output_images/roi"
LMDB_DIR = "data/lmdb"
SPLITS_DIR = "data/splits"


def get_results_dir():
    """
    결과물 dir 이름 가져오기
    """
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d(%Hh-%Mm-%Ss)")
    result_dir = os.path.join("results", f"train_{run_timestamp}")
    return result_dir


def set_results_dir(result_dir):
    os.makedirs(result_dir, exist_ok=True)


def set_logger(result_dir):
    # 타임 스탬프 출력 및 로그 파일 생성
    sys.stdout = Logger(os.path.join(result_dir, "log.txt"))
    print(f"Results directory created at: {result_dir}")


def prepare_data():
    # 1. NIfTI → PNG 변환
    print("[1/4] NIfTI → PNG 변환...")
    convert_nii_folder_to_images(SWI_INPUT_DIR, SWI_OUTPUT_DIR)
    convert_nii_folder_to_images(ROI_INPUT_DIR, ROI_OUTPUT_DIR)

    # 2. Stratified K-Fold + Hold-out 분할
    print("[2/4] Stratified K-Fold 분할...")
    make_folds()

    # 3. Bounding Box 추출
    print("[3/4] Bounding Box 추출...")
    extract_bboxes_main()

    # 4. LMDB 생성
    print("[4/4] LMDB 데이터베이스 생성...")
    create_lmdb_main()

    print("✅ 데이터 준비 완료!")


def get_fold_loaders(fold_idx):
    """특정 fold의 Train/Val DataLoader 생성"""
    train_lmdb = os.path.join(LMDB_DIR, f"fold_{fold_idx}", "train.lmdb")
    val_lmdb = os.path.join(LMDB_DIR, f"fold_{fold_idx}", "test.lmdb")

    train_dataset = CMBsDatasetLMDB(train_lmdb, BBOX_JSON_PATH)
    val_dataset = CMBsDatasetLMDB(val_lmdb, BBOX_JSON_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
    )

    print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")

    return train_loader, val_loader


def select_pretrained_weights():
    """
    weights 폴더의 가중치 파일 목록을 보여주고 선택하게 함
    Returns: 선택된 가중치 파일 경로 또는 None (새로 학습)
    """
    weights_dir = "weights"

    # weights 폴더가 없거나 비어있으면 스킵
    if not os.path.exists(weights_dir):
        print("📂 weights 폴더가 없습니다. 새로 학습을 시작합니다.")
        return None

    # .pth 파일 목록 가져오기
    weight_files = sorted([f for f in os.listdir(weights_dir) if f.endswith(".pth")])

    if not weight_files:
        print("📂 저장된 가중치가 없습니다. 새로 학습을 시작합니다.")
        return None

    # 파일 목록 출력
    print("\n" + "=" * 50)
    print("  📦 저장된 가중치 목록")
    print("=" * 50)
    print(f"  [0] 🆕 새로 학습 시작 (가중치 로드 안 함)")

    for idx, filename in enumerate(weight_files, start=1):
        filepath = os.path.join(weights_dir, filename)
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        file_time = datetime.datetime.fromtimestamp(
            os.path.getmtime(filepath)
        ).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{idx}] {filename} ({file_size:.1f}MB, {file_time})")

    print("=" * 50)

    # 사용자 입력
    while True:
        try:
            choice = input(
                "👉 로드할 가중치 번호를 선택하세요 (0~{}): ".format(len(weight_files))
            )
            choice = int(choice.strip())

            if choice == 0:
                print("🆕 새로 학습을 시작합니다.")
                return None
            elif 1 <= choice <= len(weight_files):
                selected_file = os.path.join(weights_dir, weight_files[choice - 1])
                print(f"✅ 선택된 가중치: {selected_file}")
                return selected_file
            else:
                print(f"⚠️ 0~{len(weight_files)} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("⚠️ 숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n🆕 새로 학습을 시작합니다.")
            return None


def train_fold(fold_idx, result_dir, pretrained_weights=None):
    """단일 Fold 학습"""
    print(f"\n{'=' * 50}")
    print(f"  FOLD {fold_idx + 1} / {K}")
    print(f"{'=' * 50}")

    # 1. Fold별 결과 디렉토리
    fold_result_dir = os.path.join(result_dir, f"fold_{fold_idx}")
    os.makedirs(fold_result_dir, exist_ok=True)

    # 2. DataLoader 생성
    train_loader, val_loader = get_fold_loaders(fold_idx)

    # 3. 모델 초기화
    model = SSD_FE(num_classes=2).to(DEVICE)

    # 4. 손실 함수 & 옵티마이저
    criterion = MultiBoxLoss(
        num_classes=2, iou_threshold=IOU_THRESH, alpha=ALPHA, alpha_loc=ALPHA_LOC, gamma=2.0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 시작
    start_epoch = 1
    loss_history = None

    # 5-1. 사전 학습 가중치 로드 (checkpoint 형태)
    if pretrained_weights and os.path.exists(pretrained_weights):
        print(f"  📥 가중치 로드 중: {pretrained_weights}")
        checkpoint = torch.load(pretrained_weights, map_location=DEVICE)

        # checkpoint 형태인지 확인 (epoch 키가 있으면 새로운 형식)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

            # optimizer 상태 복원
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # epoch 복원 (다음 epoch부터 시작)
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
                print(f"  📍 저장된 epoch: {checkpoint['epoch']}")

            if "val_loss" in checkpoint:
                print(f"  📊 저장된 Val Loss: {checkpoint['val_loss']:.4f}")

            if "loss_history" in checkpoint:
                loss_history = checkpoint["loss_history"]
                print(f"  📈 저장된 히스토리: {len(loss_history['train_loss'])} epochs")
        else:
            # 예전 형식 (state_dict만 저장된 경우)
            model.load_state_dict(checkpoint)

        print(f"  ✅ 가중치 로드 완료! (epoch {start_epoch}부터 시작)")

    # 6. 학습 실행
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        fold_idx=fold_idx,
        start_epoch=start_epoch,
        loss_history=loss_history,
    )

    return fold_result_dir


def get_holdout_loader():
    """Hold-out Test Set DataLoader 생성"""
    holdout_lmdb = os.path.join(LMDB_DIR, "holdout", "test.lmdb")

    # LMDB가 없으면 생성 필요
    if not os.path.exists(holdout_lmdb):
        print(f"⚠️ Hold-out LMDB 없음: {holdout_lmdb}")
        print("  create_lmdb.py에서 holdout LMDB를 먼저 생성하세요.")
        return None

    holdout_dataset = CMBsDatasetLMDB(holdout_lmdb, BBOX_JSON_PATH)

    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"  Hold-out: {len(holdout_dataset)} samples ({len(holdout_loader)} batches)")
    return holdout_loader


def evaluate_holdout(fold_results, result_dir):
    """
    Hold-out Test Set에서 K개 모델 평가
    각 Fold의 best 모델로 추론 후 결과 앙상블
    """
    print(f"\n{'=' * 50}")
    print("  HOLD-OUT TEST SET 평가")
    print(f"{'=' * 50}")

    # 1. Hold-out DataLoader
    holdout_loader = get_holdout_loader()
    if holdout_loader is None:
        return

    # 2. 각 Fold 모델 로드 및 평가
    from train import validate

    criterion = MultiBoxLoss(
        num_classes=2, iou_threshold=IOU_THRESH, alpha=ALPHA, alpha_loc=ALPHA_LOC, gamma=2.0
    )

    fold_losses = []

    for fold_idx, fold_dir in enumerate(fold_results):
        model_path = os.path.join(fold_dir, "latest_ssd.pth")

        if not os.path.exists(model_path):
            print(f"  ⚠️ Fold {fold_idx} 모델 없음: {model_path}")
            continue

        # 모델 로드
        model = SSD_FE(num_classes=2).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        # 평가
        loss, cls_loss, loc_loss = validate(model, holdout_loader, criterion, DEVICE)
        fold_losses.append((fold_idx, loss, cls_loss, loc_loss))

        print(
            f"  Fold {fold_idx}: Loss={loss:.4f} (cls={cls_loss:.4f}, loc={loc_loss:.4f})"
        )

    # 3. 평균 성능 출력
    if fold_losses:
        avg_loss = sum(fl[1] for fl in fold_losses) / len(fold_losses)
        avg_cls = sum(fl[2] for fl in fold_losses) / len(fold_losses)
        avg_loc = sum(fl[3] for fl in fold_losses) / len(fold_losses)

        print(f"\n  📊 Hold-out 평균 성능:")
        print(f"     Total Loss: {avg_loss:.4f}")
        print(f"     Cls Loss:   {avg_cls:.4f}")
        print(f"     Loc Loss:   {avg_loc:.4f}")

        # 결과 저장
        with open(os.path.join(result_dir, "holdout_results.txt"), "w") as f:
            f.write("Fold,Loss,ClsLoss,LocLoss\n")
            for fold_idx, loss, cls_loss, loc_loss in fold_losses:
                f.write(f"{fold_idx},{loss:.4f},{cls_loss:.4f},{loc_loss:.4f}\n")
            f.write(f"Average,{avg_loss:.4f},{avg_cls:.4f},{avg_loc:.4f}\n")

        print(f"  결과 저장: {result_dir}/holdout_results.txt")


def main():
    multiprocessing.freeze_support()

    result_dir = get_results_dir()
    set_results_dir(result_dir)
    set_logger(result_dir)

    print(f"Device: {DEVICE}")
    print(
        f"K-Fold: {K}, Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}"
    )

    # 2. 데이터 준비 (NIfTI → PNG)
    print("\n[Step 1] 데이터 준비...")
    prepare_data()

    # 2.5 가중치 선택 (이어서 학습할지 결정)
    print("\n[Step 1.5] 사전 학습 가중치 선택...")
    pretrained_weights = select_pretrained_weights()

    # 3. K-Fold 학습
    print("\n[Step 2] K-Fold 학습 시작...")
    fold_results = []

    for fold_idx in range(K):
        fold_dir = train_fold(fold_idx, result_dir, pretrained_weights)
        fold_results.append(fold_dir)

    # 4. 학습 완료 요약
    print("\n" + "=" * 50)
    print("  K-FOLD 학습 완료!")
    print("=" * 50)
    print(f"결과 저장 위치: {result_dir}")
    for i, fd in enumerate(fold_results):
        print(f"  Fold {i}: {fd}")

    evaluate_holdout(fold_results, result_dir)


if __name__ == "__main__":
    main()
