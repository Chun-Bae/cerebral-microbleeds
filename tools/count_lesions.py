import sys
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import label


def count_lesions(roi_path):
    if not os.path.exists(roi_path):
        print(f"Error: File not found at {roi_path}")
        return

    roi = nib.load(roi_path).get_fdata()

    # binary mask
    mask = roi > 0

    # 26-connectivity
    structure = np.ones((3, 3, 3), dtype=np.uint8)

    labeled, num_lesions = label(mask, structure=structure)

    print(f"--- Lesion Count ---")
    print(f"File: {os.path.basename(roi_path)}")
    print(f"3D 병변 개수: {num_lesions}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_lesions.py <path_to_roi_nii>")
    else:
        count_lesions(sys.argv[1])

# scipy.ndimage.label:
#  이진 마스크(0/1)에서 서로 붙어있는 픽셀/복셀 덩어리들을 찾아서
# 각 덩어리에 서로 다른 번호(ID)를 붙여주는 함수

# mask =
# 0 1 1 0 0
# 0 1 0 0 1
# 0 0 0 0 1

# labeled =
# 0 1 1 0 0
# 0 1 0 0 2
# 0 0 0 0 2
# num = 2
