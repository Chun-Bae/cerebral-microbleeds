import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
from scipy.ndimage import label
from sklearn.model_selection import train_test_split
import config
from src.utils.logger import log

EXTREME_ALWAYS_TRAIN = True

split_dir = config.SPLITS_DIR
fixed_dir = config.FIXED_SPLIT_DIR
roi_dir = config.ROI_INPUT_DIR
kfolds = config.K_FOLDS
seed = config.SEED


def _stratum(count: int) -> str:
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


def _list_roi_ids(roi_dir: str):
    files = sorted([f for f in os.listdir(roi_dir) if f.endswith(".nii")])
    return [os.path.splitext(f)[0] for f in files]


def _count_lesions(roi_path: str) -> int:
    roi = nib.load(roi_path).get_fdata()
    mask = roi > 0
    labeled, num_lesions = label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    return int(num_lesions)


def _stratified_kfold_indices(ids, y, k, seed):
    rng = np.random.default_rng(seed)
    buckets = {}
    for i, lab in enumerate(y):
        buckets.setdefault(lab, []).append(i)

    folds = [[] for _ in range(k)]
    for lab, idxs in buckets.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            folds[j % k].append(idx)

    for f in folds:
        rng.shuffle(f)
    return folds


def _load_lesion_cache(path):
    counts = {}
    strata = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            pid, cnt, s = line.strip().split("\t")
            counts[pid] = int(cnt)
            strata[pid] = s
    return counts, strata


def _save_lesion_cache(path, counts, strata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("patient_id\tlesion_count\tstratum\n")
        for pid in sorted(counts.keys()):
            f.write(f"{pid}\t{counts[pid]}\t{strata[pid]}\n")


def _save_list(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(str(x) + "\n")


def _get_or_create_lesion_counts():
    cache_path = os.path.join(split_dir, "lesion_counts.tsv")

    ids = _list_roi_ids(roi_dir)
    os.makedirs(split_dir, exist_ok=True)

    if os.path.exists(cache_path):
        log.info("병변 개수를 캐시된 파일로부터 가져옵니다.")
        counts, strata = _load_lesion_cache(cache_path)
    else:
        counts = {}
        strata = {}
        for pid in tqdm(ids, desc="⏳ 환자 ROI로부터 병변 개수 계산 중"):
            roi_path = os.path.join(roi_dir, f"{pid}.nii")
            c = _count_lesions(roi_path)
            counts[pid] = c
            strata[pid] = _stratum(c)
        _save_lesion_cache(cache_path, counts, strata)
        log.success("환자별 병변 개수 계산 완료")

    return ids, counts, strata


def _get_fixed_and_normal_ids(ids, strata):
    extreme_ids = [pid for pid in ids if strata[pid] == "extreme"]
    if EXTREME_ALWAYS_TRAIN:
        fixed_train = extreme_ids[:]
        normal_ids = [pid for pid in ids if pid not in fixed_train]
    else:
        fixed_train = []
        normal_ids = ids[:]
    return fixed_train, normal_ids


def run_generate_splits_kfold():
    """Stratified K-Fold 전략으로 데이터를 분할합니다."""
    save_dir = split_dir
    check_path = os.path.join(save_dir, "fold_0")

    if os.path.exists(check_path):
        log.info(f"이미 K-Fold 분할 모델이 존재합니다, 생략: {save_dir}")
        return

    log.info(f"K-Fold 분할 시작 (K={kfolds}, SEED={seed})")
    ids, _, strata = _get_or_create_lesion_counts()
    fixed_train, normal_ids = _get_fixed_and_normal_ids(ids, strata)

    y = [strata[pid] for pid in normal_ids]
    folds = _stratified_kfold_indices(normal_ids, y, kfolds, seed)

    for fold_idx in range(len(folds)):
        test_indices = set(folds[fold_idx])
        test_ids = [normal_ids[i] for i in test_indices]
        train_ids = [
            normal_ids[i] for i in range(len(normal_ids)) if i not in test_indices
        ]

        # Extreme Case 무조건 Train에 배정
        train_ids.extend(fixed_train)

        fold_dir = os.path.join(save_dir, f"fold_{fold_idx}")
        _save_list(os.path.join(fold_dir, "train.txt"), sorted(train_ids))
        _save_list(os.path.join(fold_dir, "test.txt"), sorted(test_ids))

    log.success(f"{len(folds)}개의 K-Fold 데이터를 성공적으로 저장했습니다: {save_dir}")


def run_generate_splits_fixed():
    """Fixed_Split (Train/Valid/Test) 전략으로 데이터를 분할합니다."""
    check_path = os.path.join(fixed_dir, "train.txt")

    if os.path.exists(check_path):
        log.info(f"이미 Fixed-Split 분할 모델이 존재합니다, 생략: {fixed_dir}")
        return

    log.info(f"Fixed-Split 분할 시작 (SEED={seed})")
    os.makedirs(fixed_dir, exist_ok=True)

    ids, _, strata = _get_or_create_lesion_counts()
    fixed_train, normal_ids = _get_fixed_and_normal_ids(ids, strata)

    # 1. Train_Total 70%, Holdout 30%
    y_total = [strata[pid] for pid in normal_ids]
    train_total_ids, holdout_ids = train_test_split(
        normal_ids, test_size=0.3, random_state=seed, stratify=y_total
    )

    # 2. Train_Total 내에서 Train 75%, Valid 25% 생성
    y_train_total = [strata[pid] for pid in train_total_ids]
    train_ids, valid_ids = train_test_split(
        train_total_ids,
        test_size=0.25,
        random_state=seed,
        stratify=y_train_total,
    )

    # Extreme Case 무조건 Train에 배정
    train_ids.extend(fixed_train)

    _save_list(os.path.join(fixed_dir, "train.txt"), sorted(train_ids))
    _save_list(os.path.join(fixed_dir, "valid.txt"), sorted(valid_ids))
    _save_list(os.path.join(fixed_dir, "test.txt"), sorted(holdout_ids))

    log.success(f"Fixed Split 저장 완료: {fixed_dir}")
    log.info(
        f"   Train: {len(train_ids)}, Valid: {len(valid_ids)}, Test: {len(holdout_ids)}"
    )
