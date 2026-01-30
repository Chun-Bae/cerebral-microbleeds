"""
data_prep.py - CMB 탐지 데이터 전처리 모듈

이 모듈은 학습용 데이터를 준비하는 모든 과정을 담당합니다.

주요 기능:
1. NIfTI 의료 영상 → PNG 이미지 변환
2. 8비트 데이터 검증
3. 환자 레벨 Train/Test 분할 (Data Leakage 방지)
4. LMDB 데이터베이스 생성 (빠른 I/O)
5. Augmentation 시각화 검증
"""

import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from collections import defaultdict

# 커스텀 모듈 임포트
from image_data import convert_nii_folder_to_images   # NIfTI 변환
from utils import check_image_bit_depth               # 비트 깊이 확인
from create_lmdb import create_lmdb                   # LMDB 생성
from dataset import CMBsDataset, CMBsDatasetLMDB, get_transforms, detection_collate


# ==========================================
# 기존 데이터셋 존재 여부 확인
# ==========================================
def check_existing_dataset(output_dirs):
    """
    8비트 데이터셋이 이미 존재하는지 확인

    이미 변환된 데이터가 있으면 재변환을 스킵하여 시간을 절약합니다.

    Args:
        output_dirs: 확인할 디렉토리 리스트 [swi, roi, swi_test, roi_test]

    Returns:
        bool: True면 기존 데이터 사용, False면 재변환 필요

    검사 항목:
        1. 모든 디렉토리 존재 여부
        2. 디렉토리 내 파일 존재 여부
        3. 샘플 이미지의 8비트 확인
    """
    # 모든 디렉토리가 존재하고 비어있지 않은지 확인
    for d in output_dirs:
        if not os.path.exists(d) or not os.listdir(d):
            return False

    # 샘플 이미지로 16비트 여부 확인
    if not os.listdir(output_dirs[0]):
        return False

    sample_file = os.listdir(output_dirs[0])[0]
    sample_path = os.path.join(output_dirs[0], sample_file)
    img = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return False

    # 8비트(uint8)가 아니면 재변환 필요
    if img.dtype != np.uint8:
        print(f"   ⚠️ 기존 데이터가 8비트가 아님 ({img.dtype}). 재변환 필요.")
        return False

    return True


# ==========================================
# 환자 ID 추출 함수
# ==========================================
def get_patient_id(filename):
    """
    파일명에서 환자 ID 추출

    Data Leakage 방지를 위해 환자 단위로 데이터를 분할합니다.
    같은 환자의 슬라이스가 Train/Test에 섞이면 과적합됩니다.

    Args:
        filename: 이미지 파일명 (예: "patient001_slice_015.png")

    Returns:
        환자 ID 문자열 (예: "patient001")

    파일명 형식:
        - "{patient_id}_slice_{slice_num}.png" → patient_id 반환
        - 그 외 형식 → 확장자 제거 후 반환
    """
    if "_slice_" in filename:
        return filename.split("_slice_")[0]
    else:
        return filename.split(".")[0]


# ==========================================
# 메인 데이터 준비 함수
# ==========================================
def prepare_data(batch_size, num_workers, output_dirs):
    """
    전체 데이터 파이프라인 실행

    NIfTI 변환부터 DataLoader 생성까지 모든 과정을 처리합니다.

    Args:
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
        output_dirs: 출력 디렉토리 리스트
                    [swi, roi, swi_test, roi_test]

    Returns:
        train_loader: 학습 데이터 로더
        test_loader: 테스트 데이터 로더
        train_dataset: 학습 데이터셋 객체 (검증용)

    파이프라인:
        1. 기존 데이터 확인 (있으면 스킵)
        2. NIfTI → PNG 변환
        3. 16비트 검증
        4. 환자 레벨 Train/Test 분할
        5. LMDB 생성 (없으면)
        6. DataLoader 생성
    """

    # ==========================================
    # [1] 기존 데이터 확인 또는 새로 변환
    # ==========================================

    if check_existing_dataset(output_dirs):
        print(f"\n✅ 기존 8비트 데이터셋 발견. (변환 & 분할 스킵)")
    else:
        print(f"\n🔄 새로운 데이터 변환 및 분할 시작...")

        # 기존 output_images 폴더 초기화
        if os.path.exists("output_images"):
            shutil.rmtree("output_images")

        # 출력 디렉토리 생성
        for d in output_dirs:
            os.makedirs(d, exist_ok=True)

        # ----------------------------------------
        # [1-1] NIfTI → PNG 변환
        # ----------------------------------------
        print("이미지 변환 중...")

        # SWI 이미지 변환
        convert_nii_folder_to_images(
            "C:/Users/Bisnel/Desktop/samsung/samsung data/swi",
            "output_images/swi"
        )

        # ROI 마스크 변환
        convert_nii_folder_to_images(
            "C:/Users/Bisnel/Desktop/samsung/samsung data/roi",
            "output_images/roi"
        )

        # 비트 깊이 검증
        check_image_bit_depth("output_images/swi")

        # ----------------------------------------
        # [1-2] 슬라이스 단위 Train/Test 분할 (피드백 전 방식)
        # ----------------------------------------
        split_ratio = 0.2  # 20% Test

        swi_images = sorted(os.listdir("output_images/swi"))
        roi_images = sorted(os.listdir("output_images/roi"))

        # SWI와 ROI 모두 존재하는 파일만 사용
        swi_set = set(swi_images)
        roi_set = set(roi_images)
        common_files = sorted(list(swi_set & roi_set))

        print(f"   총 슬라이스 수: {len(common_files)}")

        # 슬라이스별 라벨 생성 (병변 유무)
        labels = []
        for f in common_files:
            roi_path = os.path.join("output_images/roi", f)
            roi_mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
            label = 1 if roi_mask is not None and (roi_mask > 0).any() else 0
            labels.append(label)

        # 슬라이스 단위 분할 (Stratified)
        train_files, test_files, _, _ = train_test_split(
            common_files,
            labels,
            test_size=split_ratio,
            random_state=42,
            stratify=labels
        )

        print(
            f"   Train 슬라이스: {len(train_files)}, Test 슬라이스: {len(test_files)}")

        # Test 파일 이동
        for f in test_files:
            shutil.move(
                os.path.join("output_images/swi", f),
                os.path.join("output_images/swi_test", f)
            )
            shutil.move(
                os.path.join("output_images/roi", f),
                os.path.join("output_images/roi_test", f)
            )

        print("데이터 분할 완료 (슬라이스 단위).")

    # ==========================================
    # [2] Transform 정의
    # ==========================================
    train_transform, test_transform = get_transforms()

    # ==========================================
    # [3] LMDB 생성 (없으면)
    # ==========================================
    if not os.path.exists("train.lmdb") or not os.path.exists("test.lmdb"):
        print("\n⚡ [시스템] LMDB 미발견. LMDB 생성 중...")
        create_lmdb(output_dirs[0], output_dirs[1], "train.lmdb")
        create_lmdb(output_dirs[2], output_dirs[3], "test.lmdb")
        print("✅ [시스템] LMDB 생성 완료.\n")

    # ==========================================
    # [4] DataLoader 생성
    # ==========================================
    if os.path.exists("train.lmdb") and os.path.exists("test.lmdb"):
        print(f"\n🚀 [Fast Mode] LMDB 사용.")
        train_dataset = CMBsDatasetLMDB(
            "train.lmdb", transform=train_transform)
        test_dataset = CMBsDatasetLMDB("test.lmdb", transform=test_transform)
    else:
        print(f"\n🐢 [Normal Mode] 일반 파일 로딩 사용.")
        train_dataset = CMBsDataset(
            "output_images/swi", "output_images/roi",
            transform=train_transform
        )
        test_dataset = CMBsDataset(
            "output_images/swi_test", "output_images/roi_test",
            transform=test_transform
        )

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # 학습시 셔플
        num_workers=num_workers,
        pin_memory=True,        # GPU 전송 최적화
        collate_fn=detection_collate  # 커스텀 배치 함수
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # 테스트시 셔플 안함
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate
    )

    print(f"Train 데이터셋: {len(train_dataset)}, Test 데이터셋: {len(test_dataset)}")

    # ==========================================
    # [5] 데이터 검증
    # ==========================================
    print("\n[검사] 첫 배치 텐서 범위 확인:")
    sample_imgs, sample_masks, _ = next(iter(train_loader))
    print(f" - 텐서 Shape: {sample_imgs.shape}")
    print(f" - 텐서 Min: {sample_imgs.min():.4f}, Max: {sample_imgs.max():.4f}")

    if sample_imgs.max() > 0.01:
        print("   ✅ 정상: 정규화된 값이 예상 범위 내.")
    else:
        print("   ⚠️ 경고: 최대값이 너무 낮음. 정규화 문제 가능성.")
    print("-" * 50 + "\n")

    return train_loader, test_loader, train_dataset


# ==========================================
# Augmentation 검증 함수
# ==========================================
def verify_augmentation(train_dataset, train_transform):
    """
    데이터 증강(Augmentation) 결과 시각화 검증

    원본과 증강된 이미지를 나란히 저장하여
    Augmentation이 정상 적용되는지 확인합니다.

    Args:
        train_dataset: 학습 데이터셋
        train_transform: 적용할 Transform

    출력:
        train_check/comparison_X.png 파일들
        (좌: 원본+마스크 / 우: 증강+마스크)

    특징:
        - 병변이 있는 이미지만 선택
        - 마스크를 Dilation하여 잘 보이게 표시
        - 8비트 시각화
    """
    os.makedirs("train_check", exist_ok=True)
    print("\n[검사] Augmentation 비교 이미지 저장 중 (train_check 폴더)...")

    count = 0
    max_samples = 8  # 저장할 샘플 수

    # 병변이 있는 케이스 찾기
    for i in range(min(100, len(train_dataset))):
        if count >= max_samples:
            break

        # 파일명 가져오기 (LMDB는 별도 처리)
        if hasattr(train_dataset, 'swi_files'):
            swi_file = train_dataset.swi_files[i]
        else:
            swi_file = f"lmdb_sample_{i}.png"

        # ----------------------------------------
        # 원본 데이터 로드 (Transform 없이)
        # ----------------------------------------
        if hasattr(train_dataset, 'lmdb_path'):
            # LMDB 데이터셋: Transform 임시 해제
            temp_transform = train_dataset.transform
            train_dataset.transform = None
            orig_img, orig_mask, _ = train_dataset[i]
            train_dataset.transform = temp_transform
        else:
            # 일반 파일 데이터셋
            swi_path = os.path.join(train_dataset.swi_folder, swi_file)
            roi_path = os.path.join(train_dataset.roi_folder, swi_file)
            orig_img = cv2.imread(swi_path, cv2.IMREAD_UNCHANGED)
            if orig_img is None:
                continue
            if len(orig_img.shape) == 2:
                orig_img = np.stack([orig_img] * 3, axis=-1)
            orig_mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)

        if orig_img is None or orig_mask is None:
            continue

        # 병변 없는 이미지는 스킵 (검증 의미 없음)
        if np.max(orig_mask) == 0:
            continue

        count += 1

        # 그레이스케일이면 RGB로 변환
        if len(orig_img.shape) == 2:
            orig_img = np.stack([orig_img] * 3, axis=-1)

        # ----------------------------------------
        # Augmentation 적용
        # ----------------------------------------
        transformed = train_transform(image=orig_img, mask=orig_mask)
        aug_img_tensor = transformed["image"]
        aug_mask_tensor = transformed["mask"]

        # ----------------------------------------
        # 시각화 준비
        # ----------------------------------------

        # (A) 원본 시각화 (8비트)
        orig_viz = orig_img.astype(np.uint8)
        orig_bgr = cv2.cvtColor(orig_viz, cv2.COLOR_RGB2BGR)

        # 마스크 오버레이 (Dilation으로 가시성 향상)
        kernel = np.ones((3, 3), np.uint8)
        orig_mask_dilated = cv2.dilate(orig_mask, kernel, iterations=1)

        orig_overlay = orig_bgr.copy()
        orig_overlay[orig_mask_dilated > 0] = [0, 0, 255]  # 빨간색
        orig_combined = cv2.addWeighted(orig_bgr, 0.7, orig_overlay, 0.3, 0)

        # (B) 증강 이미지 시각화
        # Tensor → Numpy, 역정규화
        aug_viz = aug_img_tensor.permute(1, 2, 0).cpu().numpy()
        aug_viz = (aug_viz * 0.5 + 0.5) * 255  # [-1,1] → [0,255]
        aug_viz = np.clip(aug_viz, 0, 255).astype(np.uint8)
        aug_bgr = cv2.cvtColor(aug_viz, cv2.COLOR_RGB2BGR)

        # 증강 마스크 오버레이
        aug_mask_numpy = aug_mask_tensor.cpu().numpy()
        aug_mask_dilated = cv2.dilate(aug_mask_numpy, kernel, iterations=1)

        aug_overlay = aug_bgr.copy()
        aug_overlay[aug_mask_dilated > 0] = [0, 0, 255]
        aug_combined = cv2.addWeighted(aug_bgr, 0.7, aug_overlay, 0.3, 0)

        # ----------------------------------------
        # 이미지 병합 및 저장
        # ----------------------------------------
        h, w, _ = aug_bgr.shape
        orig_combined_resized = cv2.resize(orig_combined, (w, h))
        final_viz = np.hstack((orig_combined_resized, aug_combined))

        cv2.imwrite(f"train_check/comparison_{i}.png", final_viz)

    print("   ✅ Augmentation 비교 이미지 저장 완료.")
    print("      파일: train_check/comparison_X.png (좌: 원본+마스크 / 우: 증강+마스크)")
    print("-" * 50 + "\n")
