import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
from scipy.ndimage import label
from sklearn.model_selection import train_test_split

import config

K = config.K_FOLDS
SEED = config.SEED
SAVE_DIR = config.SPLITS_DIR
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
    print(rng)
    # strata별 인덱스 묶기
    # buckets = {'very low': [0, 5, ...], ...  'none': [1, 2, 3, ...], ...}
    buckets = {}
    # idx, label
    for i, lab in enumerate(y):
        buckets.setdefault(lab, []).append(i)

    # folds = [[], [], [], [], []]
    folds = [[] for _ in range(k)]

    for lab, idxs in buckets.items():
        idxs = idxs.copy()
        # 라벨에 담긴 번호를 무작위로 섞어줌
        rng.shuffle(idxs)

        # folds 내부에서는 strata를 구분할 필요 없기 때문에,
        # 각 strata별로 k개 분할해서 넣으면 됌
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


def save_folds_info(folds, normal_ids, svae_dir, fixed_train):
    for fold_idx in range(K):
        # fold k번째만 일단 들어감
        # k=5여서 기준 폴드 하나는 test고 나머지는 train임 (20:80)
        # [test, train, train, train, train]
        # [train, test, train, train, train]
        # [train, train, test, train, train]
        # [train, train, train, test, train]
        # [train, train, train, train, test]
        # test_idx = {1, 131, 4, 260, 5, 6, 263, 265, ...}
        test_idx = set(folds[fold_idx])

        # test_ids = ['VK002', 'VK226', 'VK005', ...]
        test_ids = [normal_ids[i] for i in test_idx]
        # train_ids = test_idsr가 아닌 ['VK002', 'VK226', 'VK005', ...]
        train_ids = [pid for i, pid in enumerate(normal_ids) if i not in test_idx]
        train_ids = train_ids + fixed_train

        fold_dir = os.path.join(SAVE_DIR, f"fold_{fold_idx}")
        save_list(os.path.join(fold_dir, "train.txt"), sorted(train_ids))
        save_list(os.path.join(fold_dir, "test.txt"), sorted(test_ids))


def main():
    # 이미 분할된 폴더가 있으면 생략
    fold_0_dir = os.path.join(SAVE_DIR, "fold_0")
    if os.path.exists(fold_0_dir):
        print(f"⏭️ 이미 분할 폴더 존재, 생략: {SAVE_DIR}")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ids = ['VK001', 'VK002', 'VK003', 'VK004', ... , 'VK652']
    ids = list_roi_ids(ROI_DIR)

    lesion_cache_path = os.path.join(SAVE_DIR, "lesion_counts.tsv")

    if os.path.exists(lesion_cache_path):
        print("✔ 병변 개수를 캐시된 파일로부터 가져왔습니다.")
        counts, strata = load_lesion_cache(lesion_cache_path)
    else:
        # counts = {'VK001': 1, ..., 'VK652': 6}
        counts = {}
        # strata = {'VK001': 'very low', ..., 'VK652': 'medium'}
        strata = {}
        for pid in tqdm(ids, desc="⏳ 환자 ROI로부터 병변 개수 계산 중"):
            roi_path = os.path.join(ROI_DIR, f"{pid}.nii")
            c = count_lesions(roi_path)
            counts[pid] = c
            strata[pid] = stratum(c)

        save_lesion_cache(lesion_cache_path, counts, strata)
        print("✔ 환자별 병변 개수 계산 완료")

    # extreme 환자 1명이라고 고정(실제 데이터에 1명만 존재)
    # extreme_ids = ['VK049']
    extreme_ids = [pid for pid in ids if strata[pid] == "extreme"]

    # 1명 고정으로 해놨기 때문에, 혹시 아니라면 예외 발생
    if EXTREME_ALWAYS_TRAIN:
        if len(extreme_ids) != 1:
            raise RuntimeError(
                f"EXTREME_ALWAYS_TRAIN=True인데 extreme 환자가 1명이 아닙니다: {extreme_ids}"
            )

        # fixed_train = ['VK049']
        fixed_train = extreme_ids[:]
        # fixed_train을 제외한, normal_ids = ['VK001', 'VK002', 'VK003', 'VK004', ... , 'VK652']
        normal_ids = [pid for pid in ids if pid not in fixed_train]
    else:
        # 2명이상이면 split 가능하므로 fixed_train은 없음
        fixed_train = []
        # normal_ids = ['VK001', 'VK002', 'VK003', 'VK004', ... , 'VK652']
        normal_ids = ids[:]

    # Stratified Split
    if config.USE_K_FOLD:
        print(f"🔄 Strategy: Stratified {config.K_FOLDS}-Fold Cross Validation")
        # 1. Hold-out Test 분리 (Dev:Test = 8:2)
        dev_ids, holdout_test_ids = train_test_split(
            normal_ids,
            test_size=0.2,
            random_state=SEED,
            stratify=[strata[pid] for pid in normal_ids],
        )
        save_list(os.path.join(SAVE_DIR, "holdout_test.txt"), sorted(holdout_test_ids))
        print(f"✔ Hold-out Test: {len(holdout_test_ids)}명 (20%)")

        # 2. K-Fold 분할
        normal_ids = dev_ids
        y = [strata[pid] for pid in normal_ids]
        folds = stratified_kfold_indices(normal_ids, y, config.K_FOLDS, SEED)

        # Fold 정보 저장
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_folds_info(folds, normal_ids, SAVE_DIR, fixed_train)

    else:
        print("🔄 Strategy: Fixed Split (Train/Test 7:3 -> Train/Valid 3:1)")
        # 1. Total -> Train_Total(70%) / Real_Test(30%)
        # Stratify 적용
        y_total = [strata[pid] for pid in normal_ids]
        train_total_ids, real_test_ids = train_test_split(
            normal_ids,
            test_size=0.3,
            random_state=SEED,
            stratify=y_total,
        )

        # Hold-out (=Real Test) 저장
        # main.py에서 evaluate_holdout 호출 시 이 목록 사용
        save_list(os.path.join(SAVE_DIR, "holdout_test.txt"), sorted(real_test_ids))
        print(f"✔ Hold-out Test (Real Test): {len(real_test_ids)}명 (30%)")

        # 2. Train_Total -> Train(75%) / Valid(25%)  (= 3:1)
        y_train_total = [strata[pid] for pid in train_total_ids]
        train_final_ids, valid_ids = train_test_split(
            train_total_ids,
            test_size=0.25,
            random_state=SEED,
            stratify=y_train_total,
        )

        # Extreme Case (fixed_train)은 무조건 Train에 포함
        train_final_ids.extend(fixed_train)

        # Fold 0 폴더에 저장
        fold_0_dir = os.path.join(SAVE_DIR, "fold_0")
        os.makedirs(fold_0_dir, exist_ok=True)

        save_list(os.path.join(fold_0_dir, "train.txt"), sorted(train_final_ids))
        save_list(os.path.join(fold_0_dir, "test.txt"), sorted(valid_ids))

        print(f"✔ Fold 0 Generated:")
        print(f"  Train: {len(train_final_ids)}명 (Includes Fixed Train)")
        print(f"  Valid: {len(valid_ids)}명")


if __name__ == "__main__":
    main()
