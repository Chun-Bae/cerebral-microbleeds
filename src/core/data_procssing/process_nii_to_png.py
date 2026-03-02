import os
import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm
import config
from src.utils.logger import log


def run_nii_to_png():
    """
    NIfTI(.nii, .nii.gz) 파일들을 2D PNG 파일로 썰어서(Slice) 저장합니다.
    (1) SWI: N4 보정이 완료된 뇌 영역 NIfTI -> PNG
    (2) ROI: 원본 마스크 NIfTI -> PNG
    """

    # 처리할 (입력 폴더, 출력 폴더, 설명) 목록
    # 설정 파일(config)에서 중앙 통제된 경로들을 바로 가져옵니다.
    conversion_tasks = [
        (
            config.SWI_N4_OUTPUT_DIR,
            config.SWI_OUTPUT_DIR,
            "SWI (Skull Stripped & N4 Corrected)",
        ),
        (config.ROI_INPUT_DIR, config.ROI_OUTPUT_DIR, "ROI (Ground Truth Masks)"),
    ]

    log.info("NIfTI to PNG 2D 자르기(Slicing) 시작...")

    for input_dir, output_dir, desc in conversion_tasks:
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"[대상: {desc}] 입력: {input_dir} 출력: {output_dir}")

        # 이미 변환된 파일 갯수 확인
        existing_pngs = [f for f in os.listdir(output_dir) if f.endswith(".png")]
        if len(existing_pngs) > 0:
            log.info(
                f"이미 {len(existing_pngs)}개의 PNG 파일이 존재합니다. 해당 작업을 생략합니다."
            )
            continue

        # NIfTI 파일 검색
        nii_files = sorted(
            [f for f in os.listdir(input_dir) if f.endswith((".nii", ".nii.gz"))]
        )

        if not nii_files:
            log.warning(f"폴더에 NIfTI 파일이 없습니다: {input_dir}")
            continue

        # 변환 돌파 (For Loop)
        for nii_file in tqdm(nii_files, desc=f"Converting {desc}", unit="vol"):
            nii_path = os.path.join(input_dir, nii_file)

            try:
                # 1. NIfTI 로드
                nii_img = nib.load(nii_path)
                nii_data = nii_img.get_fdata()
            except Exception as e:
                log.error(f"파일 로드 실패 ({nii_file}): {e}")
                continue

            # 2. 파일 이름에서 확장자 및 찌꺼기 문자열 정리
            patient_name = (
                nii_file.replace(".nii.gz", "").replace(".nii", "").replace("_bet", "")
            )

            # 3. Z축(Depth) 방향으로 썰어서 2D PNG로 저장
            # 의료 영상 관례상 대체로 Z축(slice_axis=2)이 깊이 방향입니다.
            num_slices = nii_data.shape[2]

            for i in range(num_slices):
                slice_2d = nii_data[:, :, i]

                # Max-Min 스케일링 (정규화) 후 16-bit uint 변환
                x_min, x_max = np.min(slice_2d), np.max(slice_2d)
                if x_max - x_min > 0:
                    slice_2d = (slice_2d - x_min) / (x_max - x_min) * 65535.0
                else:
                    slice_2d[:] = 0

                slice_2d = slice_2d.astype(np.uint16)

                # 최종 PNG 저장
                output_path = os.path.join(output_dir, f"{patient_name}_slice_{i}.png")
                cv2.imwrite(output_path, slice_2d)

    log.success("모든 NIfTI -> PNG 변환(Slicing) 작업이 완료되었습니다!")
