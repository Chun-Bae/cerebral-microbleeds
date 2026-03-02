"""
환자별 슬라이스당 병변 개수 분석 도구
가장 많은 병변을 가진 슬라이스 번호와 개수를 출력
"""

import os
import sys
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import label


def count_lesions_per_slice(nii_path):
    """
    NIfTI ROI 파일에서 각 슬라이스별 병변 개수 계산

    Args:
        nii_path: ROI NIfTI 파일 경로

    Returns:
        dict: {slice_idx: lesion_count}
    """
    if not os.path.exists(nii_path):
        print(f"❌ 파일 없음: {nii_path}")
        return None

    # NIfTI 로드
    nii = nib.load(nii_path)
    data = nii.get_fdata()

    # 각 슬라이스별 병변 개수
    slice_counts = {}
    num_slices = data.shape[2]

    for i in range(num_slices):
        slice_2d = data[:, :, i]

        # 이진화
        binary = (slice_2d > 0).astype(np.uint8)

        # Connected component 라벨링
        labeled, num_lesions = label(binary)

        slice_counts[i] = num_lesions

    return slice_counts


def find_max_lesion_slice(nii_path):
    """
    가장 많은 병변을 가진 슬라이스 찾기

    Returns:
        (slice_idx, lesion_count, total_lesions)
    """
    slice_counts = count_lesions_per_slice(nii_path)

    if slice_counts is None:
        return None, None, None

    # 최대 병변 슬라이스
    max_slice = max(slice_counts, key=slice_counts.get)
    max_count = slice_counts[max_slice]

    # 전체 병변 수 (중복 없이 3D 라벨링)
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    binary = (data > 0).astype(np.uint8)
    _, total_3d = label(binary)

    return max_slice, max_count, total_3d


def print_all_slices(nii_path, show_zero=False):
    """
    모든 슬라이스의 병변 개수 출력
    """
    slice_counts = count_lesions_per_slice(nii_path)

    if slice_counts is None:
        return

    patient_id = os.path.basename(nii_path).replace(".nii", "")
    print(f"\n=== {patient_id} 슬라이스별 병변 개수 ===")

    for i, count in sorted(slice_counts.items()):
        if count > 0 or show_zero:
            print(f"  Slice {i:2d}: {count}개")

    # 요약
    max_slice, max_count, total_3d = find_max_lesion_slice(nii_path)
    slices_with_lesions = sum(1 for c in slice_counts.values() if c > 0)

    print(f"\n📊 요약:")
    print(f"  전체 슬라이스: {len(slice_counts)}개")
    print(f"  병변 있는 슬라이스: {slices_with_lesions}개")
    print(f"  🏆 최대 병변 슬라이스: slice_{max_slice} ({max_count}개)")
    print(f"  3D 전체 병변 수: {total_3d}개")


def main():
    parser = argparse.ArgumentParser(description="슬라이스별 병변 개수 분석")
    parser.add_argument(
        "nii_path", help="ROI NIfTI 파일 경로 (예: data/samsung_data/roi/VK049.nii)"
    )
    parser.add_argument("--all", action="store_true", help="모든 슬라이스 출력")
    parser.add_argument("--zero", action="store_true", help="병변 없는 슬라이스도 출력")

    args = parser.parse_args()

    if args.all:
        print_all_slices(args.nii_path, show_zero=args.zero)
    else:
        max_slice, max_count, total_3d = find_max_lesion_slice(args.nii_path)

        if max_slice is not None:
            patient_id = os.path.basename(args.nii_path).replace(".nii", "")
            print(f"\n=== {patient_id} ===")
            print(f"🏆 최대 병변 슬라이스: slice_{max_slice}")
            print(f"   해당 슬라이스 병변 수: {max_count}개")
            print(f"   3D 전체 병변 수: {total_3d}개")


if __name__ == "__main__":
    main()
