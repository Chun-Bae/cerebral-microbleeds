import multiprocessing
import datetime
import os
import sys
import torch
from torch.utils.data import DataLoader
from src.logger import Logger
from scripts.preprocess_data import convert_nii_folder_to_images, run_hdbet_processing
from src.dataset import CMBsDatasetLMDB, collate_fn, BBOX_JSON_PATH
from src.model import SSD_FE
from train import train, validate
from src.loss import MultiBoxLoss
from scripts.make_folds import main as make_folds
from scripts.create_lmdb import main as create_lmdb_main
from scripts.extract_bboxes import main as extract_bboxes_main

import config

# [설정] Albumentations 업데이트 경고 끄기 (Import 이전에 설정해야 함)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def prepare_data():
    # 0. HD-BET (뇌 추출)
    print("🧠 [0/5] HD-BET Brain Extraction (SWI)...")
    if not os.path.exists(config.SWI_BET_OUTPUT_DIR) or not os.listdir(
        config.SWI_BET_OUTPUT_DIR
    ):
        run_hdbet_processing(config.SWI_INPUT_DIR, config.SWI_BET_OUTPUT_DIR)
    else:
        print(f"⏭️ 이미 HD-BET 결과 존재: {config.SWI_BET_OUTPUT_DIR}")

    # 1. NIfTI → PNG 변환
    print("[1/5] NIfTI → PNG 변환...")
    apply_n4 = getattr(config, "APPLY_N4_BIAS_CORRECTION", False)

    # SWI: HD-BET 결과물 사용 + N4 적용
    convert_nii_folder_to_images(
        config.SWI_BET_OUTPUT_DIR, config.SWI_OUTPUT_DIR, apply_n4=apply_n4
    )
    # ROI: 원본 사용 + N4 미적용
    convert_nii_folder_to_images(
        config.ROI_INPUT_DIR, config.ROI_OUTPUT_DIR, apply_n4=False
    )

    # 2. Stratified K-Fold + Hold-out 분할
    print("[2/5] Stratified K-Fold 분할...")
    make_folds()

    # 3. Bounding Box 추출
    print("[3/5] Bounding Box 추출...")
    extract_bboxes_main()

    # 4. LMDB 생성
    print("[4/4] LMDB 데이터베이스 생성...")
    create_lmdb_main()

    print("✅ 데이터 준비 완료!")


def get_fold_loaders(fold_idx):
    """특정 fold의 Train/Val DataLoader 생성"""
    if config.USE_K_FOLD:
        train_lmdb = os.path.join(config.LMDB_DIR, f"fold_{fold_idx}", "train.lmdb")
        val_lmdb = os.path.join(config.LMDB_DIR, f"fold_{fold_idx}", "test.lmdb")
    else:
        # Fixed split (Always fixed_split folder)
        train_lmdb = os.path.join(config.LMDB_DIR, "fixed_split", "train.lmdb")
        val_lmdb = os.path.join(config.LMDB_DIR, "fixed_split", "valid.lmdb")

    train_dataset = CMBsDatasetLMDB(train_lmdb, BBOX_JSON_PATH, is_train=True)
    val_dataset = CMBsDatasetLMDB(val_lmdb, BBOX_JSON_PATH, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
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
    weights_dir = config.WEIGHTS_DIR

    # weights 폴더가 없거나 비어있으면 스킵
    if not os.path.exists(config.WEIGHTS_DIR):
        print("📂 weights 폴더가 없습니다. 새로 학습을 시작합니다.")
        return None

    # .pth 파일 목록 가져오기
    weight_files = sorted(
        [f for f in os.listdir(config.WEIGHTS_DIR) if f.endswith(".pth")]
    )

    if not weight_files:
        print("📂 저장된 가중치가 없습니다. 새로 학습을 시작합니다.")
        return None

    # 파일 목록 출력
    print("\n" + "=" * 50)
    print("  📦 저장된 가중치 목록")
    print("=" * 50)
    print(f"  [0] 🆕 새로 학습 시작 (가중치 로드 안 함)")

    for idx, filename in enumerate(weight_files, start=1):
        filepath = os.path.join(config.WEIGHTS_DIR, filename)
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
                selected_file = os.path.join(
                    config.WEIGHTS_DIR, weight_files[choice - 1]
                )
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
    if config.USE_K_FOLD:
        print(f"  FOLD {fold_idx + 1} / {config.K_FOLDS}")
        fold_dirname = f"fold_{fold_idx}"
    else:
        print(f"  FIXED SPLIT TRAINING")
        fold_dirname = "fixed_split"
    print(f"{'=' * 50}")

    # 1. Fold별 결과 디렉토리
    fold_result_dir = os.path.join(result_dir, fold_dirname)
    os.makedirs(fold_result_dir, exist_ok=True)

    # 2. DataLoader 생성
    train_loader, val_loader = get_fold_loaders(fold_idx)

    import math

    estimated_epochs = math.ceil(config.MAX_ITERATIONS / len(train_loader))
    print(f"  📌 총 {config.MAX_ITERATIONS} Iterations 까지 반복합니다.")
    print(
        f"  📌 1 Epoch = {len(train_loader)} Iterations (예상 학습 종료: 약 {estimated_epochs} Epochs)"
    )

    # 3. 모델 초기화
    model = SSD_FE(num_classes=2).to(config.DEVICE)

    # 4. 손실 함수 & 옵티마이저
    criterion = MultiBoxLoss(
        num_classes=2,
        iou_threshold=config.TRAIN_IOU_THRESH,
        neg_pos_ratio=config.NEG_POS_RATIO,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 5. 시작
    start_epoch = 1
    loss_history = None
    start_global_step = None

    # 5-1. 사전 학습 가중치 로드 (checkpoint 형태)
    if pretrained_weights and os.path.exists(pretrained_weights):
        print(f"  📥 가중치 로드 중: {pretrained_weights}")
        checkpoint = torch.load(pretrained_weights, map_location=config.DEVICE)

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

            # global_step 복원
            if "global_step" in checkpoint:
                start_global_step = checkpoint["global_step"]
                print(f"  📍 저장된 이터레이션: {start_global_step}")

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
        device=config.DEVICE,
        fold_idx=fold_idx,
        start_epoch=start_epoch,
        loss_history=loss_history,
        start_global_step=start_global_step,
    )

    return fold_result_dir


def get_holdout_loader():
    """Hold-out Test Set DataLoader 생성"""
    holdout_lmdb = os.path.join(config.LMDB_DIR, "fixed_split", "test.lmdb")

    # LMDB가 없으면 생성 필요
    if not os.path.exists(holdout_lmdb):
        print(f"⚠️ Test LMDB 없음: {holdout_lmdb}")
        print("  create_lmdb.py에서 LMDB를 먼저 생성하세요.")
        return None

    holdout_dataset = CMBsDatasetLMDB(holdout_lmdb, BBOX_JSON_PATH)

    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
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

    criterion = MultiBoxLoss(
        num_classes=2,
        iou_threshold=config.IOU_THRESH,
        neg_pos_ratio=config.NEG_POS_RATIO,
    )

    fold_losses = []

    for fold_idx, fold_dir in enumerate(fold_results):
        model_path = os.path.join(fold_dir, "latest_ssd.pth")

        if not os.path.exists(model_path):
            print(f"  ⚠️ Fold {fold_idx} 모델 없음: {model_path}")
            continue

        # 모델 로드
        model = SSD_FE(num_classes=2).to(config.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))

        # 평가
        loss, cls_loss, loc_loss = validate(
            model, holdout_loader, criterion, config.DEVICE
        )
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


import argparse


def main():
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="CMB Detection Training")
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to pretrained weights"
    )
    parser.add_argument(
        "--fold", type=int, default=None, help="Specific fold to train (0-indexed)"
    )
    parser.add_argument(
        "--fixed_split",
        action="store_true",
        help="Use fixed split (Train/Val/Test) instead of K-Fold",
    )
    args = parser.parse_args()

    if args.fixed_split:
        config.USE_K_FOLD = False
        print("🔧 Configuration override: Fixed Split Mode ON")

    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d(%Hh-%Mm-%Ss)")
    result_dir = os.path.join(config.RESULTS_DIR, f"train_{run_timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(result_dir, "log.txt"))

    print(f"Results directory created at: {result_dir}")

    print(f"Device: {config.DEVICE}")
    print(
        f"K-Fold: {config.K_FOLDS}, Iterations: {config.MAX_ITERATIONS}, Batch: {config.BATCH_SIZE}, LR: {config.LEARNING_RATE}"
    )

    # 2. 데이터 준비 (NIfTI → PNG)
    print("\n[Step 1] 데이터 준비...")
    prepare_data()

    # 2.5 가중치 선택
    if args.weights:
        print(f"\n[Step 2] 지정된 가중치 사용: {args.weights}")
        pretrained_weights = args.weights
    else:
        print("\n[Step 2] 사전 학습 가중치 선택...")
        pretrained_weights = select_pretrained_weights()

    # 3. K-Fold 학습
    # 3. K-Fold 학습
    print(
        "\n[Step 3] K-Fold 학습 시작..."
        if config.USE_K_FOLD
        else "\n[Step 3] 학습 시작 (Fixed Split)..."
    )
    fold_results = []

    # 정규식 사용을 위해 import (함수 내부 import 혹은 상단 이동 권장, 여기서는 편의상 상단에 추가되었다고 가정하거나 지역적으로 사용)
    import re

    if args.fold is not None:
        folds_to_run = [args.fold]
    else:
        folds_to_run = range(config.K_FOLDS) if config.USE_K_FOLD else range(1)

    for fold_idx in folds_to_run:
        current_fold_weights = pretrained_weights

        # 사용자가 가중치를 선택했고, 파일명에 'fold_X' 패턴이 있다면 해당 fold에 맞는 파일로 교체 시도
        if pretrained_weights:
            filename = os.path.basename(pretrained_weights)
            dirname = os.path.dirname(pretrained_weights)

            # "fold_숫자" 패턴 감지 (예: latest_ssd_fold_0.pth)
            pattern = r"(fold_)(\d+)"
            match = re.search(pattern, filename)

            if match:
                # 현재 fold_idx로 숫자를 치환
                new_filename = re.sub(pattern, f"\\g<1>{fold_idx}", filename)
                candidate_path = os.path.join(dirname, new_filename)

                if os.path.exists(candidate_path):
                    current_fold_weights = candidate_path
                    print(f"  🔄 Fold {fold_idx}용 가중치 자동 선택: {new_filename}")
                else:
                    # 해당 fold의 가중치 파일이 없으면, 다른 fold의 가중치를 쓰지 않고 초기화 상태로 학습
                    print(
                        f"  ⚠️ Fold {fold_idx}용 가중치 파일({new_filename})이 없어 처음부터 학습합니다."
                    )
                    current_fold_weights = None
            else:
                # fold 특정 패턴이 없는 일반 가중치(예: vgg_backbone.pth)는 그대로 사용
                pass

        fold_dir = train_fold(fold_idx, result_dir, current_fold_weights)
        fold_results.append(fold_dir)

    # 4. 학습 완료 요약
    print("\n" + "=" * 50)
    print("  K-FOLD 학습 완료!")
    print("=" * 50)
    print(f"결과 저장 위치: {result_dir}")
    for i, fd in enumerate(fold_results):
        print(f"  Fold {i}: {fd}")

    if not config.USE_K_FOLD:
        evaluate_holdout(fold_results, result_dir)
    else:
        print("ℹ️ K-Fold 완료. (별도의 Hold-out 평가는 수행하지 않음)")


if __name__ == "__main__":
    main()
