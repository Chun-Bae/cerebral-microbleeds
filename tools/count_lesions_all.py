import os
import nibabel as nib
import numpy as np
from scipy.ndimage import label

# =========================
# 설정
# =========================
ROI_DIR = r".\data\samsung_data\roi"
CONNECTIVITY = 26        # 6 or 26
MIN_VOXELS = 1           # 최소 병변 크기 (필터 안 쓰면 1)

# =========================
# connectivity 구조 생성
# =========================
def get_structure(connectivity):
    if connectivity == 26:
        return np.ones((3, 3, 3), dtype=np.uint8)
    elif connectivity == 6:
        s = np.zeros((3, 3, 3), dtype=np.uint8)
        s[1,1,1] = 1
        s[0,1,1] = s[2,1,1] = 1
        s[1,0,1] = s[1,2,1] = 1
        s[1,1,0] = s[1,1,2] = 1
        return s
    else:
        raise ValueError("CONNECTIVITY must be 6 or 26")

STRUCTURE = get_structure(CONNECTIVITY)

# =========================
# 병변 개수 계산
# =========================
def count_lesions_in_roi(roi_path):
    roi = nib.load(roi_path).get_fdata()
    mask = roi > 0

    labeled, num = label(mask, structure=STRUCTURE)

    if MIN_VOXELS <= 1:
        return num

    # 크기 필터
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    valid = np.sum(sizes >= MIN_VOXELS)
    return int(valid)

# =========================
# 메인
# =========================
def main():
    if not os.path.exists(ROI_DIR):
        raise RuntimeError(f"ROI_DIR not found: {ROI_DIR}")

    roi_files = sorted([
        f for f in os.listdir(ROI_DIR)
        if f.endswith(".nii")
    ])

    results = []

    for fname in roi_files:
        path = os.path.join(ROI_DIR, fname)
        try:
            n = count_lesions_in_roi(path)
            results.append((fname.replace(".nii", ""), n))
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    # 병변 개수 기준 내림차순 정렬
    results.sort(key=lambda x: x[1], reverse=True)

    # 출력
    print("====================================")
    print(f"Lesion count per patient")
    print(f"Connectivity: {CONNECTIVITY}, min_voxels: {MIN_VOXELS}")
    print("====================================")

    for pid, cnt in results:
        print(f"{pid:10s} : {cnt}")

    print("====================================")
    print(f"Total patients: {len(results)}")

if __name__ == "__main__":
    main()
