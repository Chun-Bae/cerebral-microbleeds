from dask.array.wrap import w
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label

K = 5
SEED = 42
SAVE_DIR = r".\data\splits"
ROI_DIR = r".\data\samsung_data\roi"

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
    labeled, num_lesions = label(mask, structure=np.ones((3,3,3), dtype=np.uint8))
    return int(num_lesions)

def stratified_kfold_indices(ids, y, k, seed):
    # 난수 생성기
    rng = np.random.default_rng(seed)
    
    # strata별 인덱스 묶기
    buckets = {}
    for i, lab in enumerate(y):
        buckets.setdefault(lab, []).append(i)


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

def main():
    # ids = ['VK001', 'VK002', 'VK003', 'VK004', ... , 'VK652']
    ids = list_roi_ids(ROI_DIR)

    lesion_cache_path = os.path.join(SAVE_DIR, "lesion_counts.tsv")

    if os.path.exists(lesion_cache_path):
        print("✔ 병변 개수를 캐시된 파일로부터 가져왔습니다.")
        counts, strata = load_lesion_cache(lesion_cache_path)
    else:
        print("⏳ 환자 ROI로부터 병변 개수 계산중...")
        # counts = {'VK001': 1, ..., 'VK652': 6}
        counts = {}
        # strata = {'VK001': 'very low', ..., 'VK652': 'medium'}
        strata = {}
        for pid in ids:
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
            raise RuntimeError(f"EXTREME_ALWAYS_TRAIN=True인데 extreme 환자가 1명이 아닙니다: {extreme_ids}")
        
        # fixed_train = ['VK049']
        fixed_train = extreme_ids[:] 
        # fixed_train을 제외한, normal_ids = ['VK001', 'VK002', 'VK003', 'VK004', ... , 'VK652']
        normal_ids = [pid for pid in ids if pid not in fixed_train]
    else:
        # 2명이상이면 split 가능하므로 fixed_train은 없음
        fixed_train = []
        # normal_ids = ['VK001', 'VK002', 'VK003', 'VK004', ... , 'VK652']
        normal_ids = ids[:]

    y = [strata[pid] for pid in normal_ids]

if __name__ == "__main__":
    main()


# import os
# import numpy as np
# import nibabel as nib
# from scipy.ndimage import label
# from collections import Counter

# # =========================
# # 설정
# # =========================
# ROI_DIR = r".\data\samsung_data\roi"
# SAVE_DIR = r".\splits"

# K = 5
# SEED = 42

# CONNECTIVITY = 26     # 6 or 26
# MIN_VOXELS = 1        # 너무 작은 조각 제거하고 싶으면 3,5,10 등으로

# # extreme이 1명이라고 가정하고: 이 환자는 항상 train에만 넣기
# EXTREME_ALWAYS_TRAIN = True

# # 유석이 strata (extreme 포함)
# def stratum(count: int) -> str:
#     if count == 0:
#         return "none"
#     if count <= 2:
#         return "very_low"
#     if count <= 5:
#         return "low"
#     if count <= 10:
#         return "medium"
#     if count <= 20:
#         return "high"
#     return "extreme"

# # =========================
# # 유틸
# # =========================
# def get_structure(connectivity: int):
#     if connectivity == 26:
#         return np.ones((3, 3, 3), dtype=np.uint8)
#     if connectivity == 6:
#         s = np.zeros((3, 3, 3), dtype=np.uint8)
#         s[1,1,1] = 1
#         s[0,1,1] = s[2,1,1] = 1
#         s[1,0,1] = s[1,2,1] = 1
#         s[1,1,0] = s[1,1,2] = 1
#         return s
#     raise ValueError("CONNECTIVITY must be 6 or 26")

# STRUCTURE = get_structure(CONNECTIVITY)

# def list_roi_ids(roi_dir: str):
#     files = sorted([f for f in os.listdir(roi_dir) if f.endswith(".nii")])
#     return [os.path.splitext(f)[0] for f in files]

# def count_lesions(roi_path: str) -> int:
#     roi = nib.load(roi_path).get_fdata()
#     mask = roi > 0

#     labeled, num = label(mask, structure=STRUCTURE)

#     if MIN_VOXELS <= 1:
#         return int(num)

#     sizes = np.bincount(labeled.ravel())
#     sizes[0] = 0
#     return int(np.sum(sizes >= MIN_VOXELS))

# def stratified_kfold_indices(ids, y, k, seed):
#     """
#     간단 stratified k-fold (sklearn 없이)
#     y: strata 라벨 리스트
#     return: folds[i] = val indices list
#     """
#     rng = np.random.default_rng(seed)

#     buckets = {}
#     for i, lab in enumerate(y):
#         buckets.setdefault(lab, []).append(i)

#     folds = [[] for _ in range(k)]
#     for lab, idxs in buckets.items():
#         idxs = idxs.copy()
#         rng.shuffle(idxs)
#         for j, idx in enumerate(idxs):
#             folds[j % k].append(idx)

#     for f in folds:
#         rng.shuffle(f)

#     return folds

# def save_list(path, items):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         for x in items:
#             f.write(str(x) + "\n")

# # =========================
# # 메인
# # =========================
# def main():
#     ids = list_roi_ids(ROI_DIR)
#     if not ids:
#         raise RuntimeError(f"No .nii files found in {ROI_DIR}")

#     # 환자별 lesion count + strata
#     counts = {}
#     strata = {}
#     for pid in ids:
#         roi_path = os.path.join(ROI_DIR, f"{pid}.nii")
#         c = count_lesions(roi_path)
#         counts[pid] = c
#         strata[pid] = stratum(c)

#     # extreme 환자 찾기 (1명이라고 가정)
#     extreme_ids = [pid for pid in ids if strata[pid] == "extreme"]

#     if EXTREME_ALWAYS_TRAIN:
#         if len(extreme_ids) != 1:
#             raise RuntimeError(
#                 f"EXTREME_ALWAYS_TRAIN=True인데 extreme 환자가 1명이 아닙니다: {extreme_ids}"
#             )

#         fixed_train = extreme_ids[:]          # extreme 1명 고정 train
#         normal_ids = [pid for pid in ids if pid not in fixed_train]
#     else:
#         fixed_train = []
#         normal_ids = ids[:]

#     # stratified k-fold는 normal_ids에 대해서만 수행
#     y = [strata[pid] for pid in normal_ids]
#     folds = stratified_kfold_indices(normal_ids, y, K, SEED)

#     os.makedirs(SAVE_DIR, exist_ok=True)

#     # 전체 분포 출력
#     print("====================================")
#     print(f"Total patients: {len(ids)}")
#     print(f"K: {K}, seed: {SEED}")
#     print(f"connectivity: {CONNECTIVITY}, min_voxels: {MIN_VOXELS}")
#     print("Strata counts (ALL):")
#     print(Counter([strata[pid] for pid in ids]))
#     print("------------------------------------")
#     if fixed_train:
#         print(f"Extreme(always train): {fixed_train}")
#         print("Strata counts (excluding extreme fixed):")
#         print(Counter([strata[pid] for pid in normal_ids]))
#     print("====================================")

#     # fold 저장
#     for fold_idx in range(K):
#         val_idx = set(folds[fold_idx])

#         val_ids = [normal_ids[i] for i in val_idx]
#         train_ids = [pid for i, pid in enumerate(normal_ids) if i not in val_idx]

#         # extreme(1명) 항상 train에 추가
#         train_ids = train_ids + fixed_train

#         fold_dir = os.path.join(SAVE_DIR, f"fold_{fold_idx}")
#         save_list(os.path.join(fold_dir, "train.txt"), sorted(train_ids))
#         save_list(os.path.join(fold_dir, "val.txt"), sorted(val_ids))

#         # fold별 strata 분포 출력
#         val_strata = [strata[pid] for pid in val_ids]
#         train_strata = [strata[pid] for pid in train_ids]
#         print(
#             f"[fold {fold_idx}] train={len(train_ids)} val={len(val_ids)} "
#             f"train_strata={Counter(train_strata)} val_strata={Counter(val_strata)}"
#         )

#     # 환자별 count 테이블 저장
#     table_path = os.path.join(SAVE_DIR, "lesion_counts.tsv")
#     with open(table_path, "w", encoding="utf-8") as f:
#         f.write("patient_id\tlesion_count\tstratum\n")
#         for pid in ids:
#             f.write(f"{pid}\t{counts[pid]}\t{strata[pid]}\n")

#     print(f"\nSaved splits to: {SAVE_DIR}")
#     print(f"Saved table to: {table_path}")

# if __name__ == "__main__":
#     main()
