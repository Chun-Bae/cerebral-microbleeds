import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
from scipy.ndimage import label
from sklearn.model_selection import train_test_split, StratifiedKFold

import config

K = config.K_FOLDS
SEED = config.SEED
SAVE_DIR = config.SPLITS_DIR  # K-Fold용
FIXED_DIR = config.FIXED_SPLIT_DIR  # Fixed Split용
ROI_DIR = config.ROI_INPUT_DIR

EXTREME_ALWAYS_TRAIN = True


def stratum(count: int) -> str:
    if count == 0:
        return "none"
    if count <= 2:
        return "very low"
    if count <= 5:
        return "low"
    if count <= 10:
        return "medium"
    if count <= 20:
        return "high"
    return "extreme"


def list_roi_ids(roi_dir: str):
    files = sorted([f for f in os.listdir(roi_dir) if f.endswith(".nii")])
    return [os.path.splitext(f)[0] for f in files]


def count_lesions(roi_path: str) -> int:
    roi = nib.load(roi_path).get_fdata()
    mask = roi > 0
    labeled, num_lesions = label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    return int(num_lesions)


def stratified_kfold_indices(ids, y, k, seed):
    # 난수 생성기
    rng = np.random.default_rng(seed)

    # strata별 인덱스 묶기
    buckets = {}
    for i, lab in enumerate(y):
        buckets.setdefault(lab, []).append(i)

    # folds = [[], [], ..., []]
    folds = [[] for _ in range(k)]

    for lab, idxs in buckets.items():
        idxs = list(idxs)
        rng.shuffle(idxs)

        for j, idx in enumerate(idxs):
            folds[j % k].append(idx)

    # 각 폴드별로 한 번 더 섞어줌
    for f in folds:
        rng.shuffle(f)

    return folds


def load_lesion_cache(path):
    counts = {}
    strata = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            pid, cnt, s = line.strip().split("\t")
            counts[pid] = int(cnt)
            strata[pid] = s
    return counts, strata


def save_lesion_cache(path, counts, strata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("patient_id\tlesion_count\tstratum\n")
        for pid in sorted(counts.keys()):
            f.write(f"{pid}\t{counts[pid]}\t{strata[pid]}\n")


def save_list(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(str(x) + "\n")


def save_kfold_info(folds, normal_ids, save_dir, fixed_train):
    """
    K-Fold 정보 저장
    각 폴드가 Test가 되고 나머지가 Train이 됨.
    """
    for fold_idx in range(len(folds)):
        # 현재 fold가 Test Set
        test_indices = set(folds[fold_idx])

        test_ids = [normal_ids[i] for i in test_indices]
        train_ids = [
            normal_ids[i] for i in range(len(normal_ids)) if i not in test_indices
        ]

        # Extreme Case 추가
        train_ids.extend(fixed_train)

        fold_dir = os.path.join(save_dir, f"fold_{fold_idx}")
        save_list(os.path.join(fold_dir, "train.txt"), sorted(train_ids))
        save_list(os.path.join(fold_dir, "test.txt"), sorted(test_ids))

    print(f"✔ Saved {len(folds)} folds to {save_dir}")


def main():
    target_dir = SAVE_DIR if config.USE_K_FOLD else FIXED_DIR

    # 이미 분할된 폴더가 있으면 생략 (중복 생성 방지)
    # K-Fold: fold_0 존재 여부 확인
    # Fixed: train.txt 존재 여부 확인
    if config.USE_K_FOLD:
        check_path = os.path.join(target_dir, "fold_0")
    else:
        check_path = os.path.join(target_dir, "train.txt")

    if os.path.exists(check_path):
        print(f"⏭️ 이미 분할 정보 존재, 생략: {target_dir}")
        return

    os.makedirs(target_dir, exist_ok=True)

    # 1. 환자 ID 및 병변 수 로드
    ids = list_roi_ids(ROI_DIR)
    lesion_cache_path = os.path.join(
        config.SPLITS_DIR, "lesion_counts.tsv"
    )  # 항상 splits 폴더에 저장/로드

    if os.path.exists(lesion_cache_path):
        print("✔ 병변 개수를 캐시된 파일로부터 가져왔습니다.")
        counts, strata = load_lesion_cache(lesion_cache_path)
    else:
        counts = {}
        strata = {}
        for pid in tqdm(ids, desc="⏳ 환자 ROI로부터 병변 개수 계산 중"):
            roi_path = os.path.join(ROI_DIR, f"{pid}.nii")
            c = count_lesions(roi_path)
            counts[pid] = c
            strata[pid] = stratum(c)
        save_lesion_cache(lesion_cache_path, counts, strata)
        print("✔ 환자별 병변 개수 계산 완료")

    # 2. Extreme Case 분리
    extreme_ids = [pid for pid in ids if strata[pid] == "extreme"]

    if EXTREME_ALWAYS_TRAIN:
        if len(extreme_ids) != 1:
            # 데이터 특성상 1명이어야 함 (안전장치)
            # 만약 데이터가 늘어나면 로직 수정 필요
            pass
        fixed_train = extreme_ids[:]
        normal_ids = [pid for pid in ids if pid not in fixed_train]
    else:
        fixed_train = []
        normal_ids = ids[:]

    # 3. 분할 전략 실행
    if config.USE_K_FOLD:
        print(
            f"🔄 Strategy: Stratified {config.K_FOLDS}-Fold Cross Validation (No Separate Holdout)"
        )

        # 전체 데이터에 대해 Stratified K-Fold
        y = [strata[pid] for pid in normal_ids]
        folds = stratified_kfold_indices(normal_ids, y, config.K_FOLDS, SEED)

        save_kfold_info(folds, normal_ids, SAVE_DIR, fixed_train)

    else:
        print("🔄 Strategy: Fixed Split (Train/Valid/Holdout)")

        # 1. Total -> Train_Total(70%) / Holdout_Test(30%)
        y_total = [strata[pid] for pid in normal_ids]
        train_total_ids, holdout_ids = train_test_split(
            normal_ids,
            test_size=0.3,  # 30% Holdout
            random_state=SEED,
            stratify=y_total,
        )

        # 2. Train_Total -> Train(75%) / Valid(25%) => (Overall ~ 52.5% Train / 17.5% Valid)
        y_train_total = [strata[pid] for pid in train_total_ids]
        train_ids, valid_ids = train_test_split(
            train_total_ids,
            test_size=0.25,  # 25% Validation of Train_Total
            random_state=SEED,
            stratify=y_train_total,
        )

        # Extreme Case 추가
        train_ids.extend(fixed_train)

        # 저장
        save_list(os.path.join(FIXED_DIR, "train.txt"), sorted(train_ids))
        save_list(os.path.join(FIXED_DIR, "valid.txt"), sorted(valid_ids))
        save_list(os.path.join(FIXED_DIR, "test.txt"), sorted(holdout_ids))

        print(f"✔ Fixed Split Generated at {FIXED_DIR}:")
        print(f"  Train:   {len(train_ids)} (Includes Fixed Train)")
        print(f"  Valid:   {len(valid_ids)}")
        print(f"  Test:    {len(holdout_ids)}")


if __name__ == "__main__":
    main()
