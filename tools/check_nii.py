import sys
import os
import nibabel as nib
import numpy as np


def check_nii(nii_path):
    if not os.path.exists(nii_path):
        print(f"Error: File not found at {nii_path}")
        return

    nii_img = nib.load(nii_path)
    header = nii_img.header
    shape = nii_img.shape
    pixdim = header['pixdim']

    print(f"{'='*60}")
    print(f" NIfTI Info: {os.path.basename(nii_path)}")
    print(f"{'='*60}")

    # 1. Matrix Size (해상도)
    print(f"\n[Matrix Size (해상도)]")
    print(f"  {shape[0]} x {shape[1]}")
    if len(shape) > 2:
        print(f"  Slices: {shape[2]}")

    # 2. Voxel Size (복셀 크기, mm)
    voxel = pixdim[1:4]
    print(f"\n[Voxel Size (mm)]")
    print(f"  X: {voxel[0]:.4f}")
    print(f"  Y: {voxel[1]:.4f}")
    print(f"  Z: {voxel[2]:.4f}")

    # 3. Slice Thickness (슬라이스 두께)
    print(f"\n[Slice Thickness (슬라이스 두께)]")
    print(f"  {voxel[2]:.4f} mm")

    # 4. FOV (촬영 시야각, mm)
    if len(shape) >= 2:
        fov_x = voxel[0] * shape[0]
        fov_y = voxel[1] * shape[1]
        print(f"\n[FOV (촬영 시야각)]")
        print(f"  X: {fov_x:.2f} mm")
        print(f"  Y: {fov_y:.2f} mm")

    # 5. Orientation (방향 정보)
    qform_code = int(header['qform_code'])
    sform_code = int(header['sform_code'])
    qform_labels = {0: 'unknown', 1: 'scanner', 2: 'aligned', 3: 'talairach', 4: 'mni'}
    print(f"\n[Orientation]")
    print(f"  qform_code: {qform_labels.get(qform_code, qform_code)}")
    print(f"  sform_code: {qform_labels.get(sform_code, sform_code)}")
    affine = nii_img.affine
    print(f"  Affine matrix:")
    for row in affine:
        print(f"    [{', '.join(f'{v:10.4f}' for v in row)}]")

    # 6. Data Type
    print(f"\n[Data Type]")
    print(f"  dtype: {header.get_data_dtype()}")
    print(f"  bitpix: {header['bitpix']}")

    # 7. Data Range (실제 데이터 값 범위)
    data = np.asarray(nii_img.dataobj)
    print(f"\n[Data Range]")
    print(f"  min: {data.min():.4f}")
    print(f"  max: {data.max():.4f}")
    print(f"  mean: {data.mean():.4f}")
    print(f"  std: {data.std():.4f}")

    # 8. TE/TR 관련 안내
    print(f"\n[TE / TR]")
    descrip = str(header['descrip'])
    if descrip and descrip not in ("b''", "np.bytes_(b'')"):
        print(f"  descrip 필드: {descrip}")
    else:
        print(f"  NIfTI 표준 헤더에는 TE/TR 정보가 포함되지 않습니다.")
        print(f"  원본 DICOM 파일에서만 확인 가능합니다.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_nii.py <path_to_nii_file>")
        print("  예: python tools/check_nii.py data/samsung_data/swi/VK003.nii")
    else:
        for path in sys.argv[1:]:
            check_nii(path)
            print()
