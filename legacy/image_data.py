import nibabel as nib
import numpy as np
import cv2
import os

# NIfTI(.nii) 파일이 포함된 폴더를 받아서, 모든 .nii 파일을 8-bit PNG 이미지로 변환


def convert_nii_folder_to_images(folder_path, output_folder, slice_axis=2):
    os.makedirs(output_folder, exist_ok=True)  # 저장 폴더가 없으면 생성

    # 폴더 내 모든 .nii 파일 가져오기
    nii_files = [f for f in os.listdir(folder_path) if f.endswith(".nii")]

    if not nii_files:
        print(f"⚠️ 폴더에 .nii 파일이 없습니다: {folder_path}")
        return

    for nii_file in nii_files:
        nii_path = os.path.join(folder_path, nii_file)

        # NIfTI 파일 로드
        nii_img = nib.load(nii_path)
        nii_data = nii_img.get_fdata()

        num_slices = nii_data.shape[slice_axis]  # 전체 슬라이스 개수
        file_name = os.path.splitext(nii_file)[0]  # 파일명(확장자 제거)

        # 각 슬라이스를 PNG 이미지로 변환
        for i in range(num_slices):
            if slice_axis == 2:  # Axial (기본값)
                slice_2d = nii_data[:, :, i]
            elif slice_axis == 1:  # Sagittal (측면)
                slice_2d = nii_data[:, i, :]
            elif slice_axis == 0:  # Coronal (전후)
                slice_2d = nii_data[i, :, :]

            # [Removed] Brain Region Extraction (Bounding Box Masking) code removed as per user request.
            # Using raw slice data for normalization instead.

            # 8-bit 정규화 (0~255 범위로 변환)
            min_val, max_val = np.min(slice_2d), np.max(slice_2d)
            if max_val - min_val > 0:
                slice_2d = (slice_2d - min_val) / (max_val - min_val) * 255
            else:
                slice_2d = slice_2d * 0  # 변화가 없으면 0으로

            slice_2d = slice_2d.astype(np.uint8)

            # 파일명과 슬라이스 번호를 조합하여 PNG 저장 (8-bit)
            output_path = os.path.join(
                output_folder, f"{file_name}_slice_{i}.png")
            cv2.imwrite(output_path, slice_2d)

        print(f"변환 완료: {nii_path} → {output_folder}")
