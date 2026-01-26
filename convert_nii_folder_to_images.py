"""
This module converts NIfTI (.nii) files in a folder to a series of 2D PNG images.
"""
import os
import nibabel as nib
import numpy as np
import cv2

def convert_nii_folder_to_images(input_dir, output_dir, slice_axis=2):
    """
    NIfTI(nii) 파일을 2D PNG 파일로 슬라이스 변환
    """

    # nii_files = ['VK001.nii', 'VK002.nii', 'VK003.nii', ... ,'VK0652.nii']
    nii_files = [f for f in os.listdir(input_dir) if f.endswith(".nii")]
    
    if not nii_files:
        print(f"⚠️ 폴더에 .nii 파일이 없습니다: {input_dir}")
        return

    for nii_file in nii_files:
        # nii_path = "data/samsung_data/swi/VK001.nii"
        nii_path = os.path.join(input_dir, nii_file)

        # nii 객체, 데이터 크기, 값, 메타데이터 등이 들어감
        nii_img = nib.load(nii_path)

        # 3D pixels 값 (512, 512, 72)
        nii_data = nii_img.get_fdata()

        # num_slices = 72
        num_slices = nii_data.shape[slice_axis]

        # 확장자 제거 (split extenstion)
        # patient_name = VK001
        patient_name = os.path.splitext(nii_file)[0]

        # 각 슬라이스를 PNG 이미지로 변환
        # slice_axis = 2만 사용 중
        for i in range(num_slices):
            if slice_axis == 2:         # Axial (기본값)
                slice_2d = nii_data[:, :, i]
            elif slice_axis == 1:       # Sagittal (측면)
                slice_2d = nii_data[:, i, :]
            elif slice_axis == 0:       # Coronal (전후)
                slice_2d = nii_data[i, :, :]
            else:
                raise ValueError(f"Invalid slice_axis: {slice_axis}. Must be 0, 1, or 2.")

            # 16-bit 정규화 (0~65535)
            # 정규화 공식: x_new = \frac{x - x_min}{x_max - x_min} * (0~원하는 범위) 
            x_min, x_max = np.min(slice_2d), np.max(slice_2d)
            
            if x_max - x_min > 0:
                slice_2d = (slice_2d - x_min) / (x_max - x_min) * 65535
            # roi 이미지 같은 경우, x_max, x_min이 둘다 0이면 그냥 0 이므로 연산 최적화
            else:
                slice_2d[:] = 0
            
            # uint16으로 변환해야 cv2에서 올바르게 16-bit PNG로 저장됨
            slice_2d = slice_2d.astype(np.uint16)

            output_path = os.path.join(output_dir, f"{patient_name}_slice_{i}.png")

            # 이미지 파일 저장
            # pylint: disable=no-member
            cv2.imwrite(output_path, slice_2d)
            
        print(f"변환 완료: {nii_path} → {output_dir}")

