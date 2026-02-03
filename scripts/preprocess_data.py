"""
This module converts NIfTI (.nii) files in a folder to a series of 2D PNG images.
"""

import os
import nibabel as nib
import numpy as np
import cv2
import collections
import torch
import glob
from multiprocessing import Pool
from HD_BET.checkpoint_download import maybe_download_parameters
from HD_BET.hd_bet_prediction import hdbet_predict, apply_bet
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from HD_BET.paths import folder_with_parameter_files

SWI_INPUT_DIR = "data/samsung_data/swi"
ROI_INPUT_DIR = "data/samsung_data/roi"

SWI_BET_OUTPUT_DIR = "data/output_images/swi_bet"

SWI_OUTPUT_DIR = "data/output_images/swi"
ROI_OUTPUT_DIR = "data/output_images/roi"


def run_hdbet_processing(input_dir, output_dir):
    """
    폴더 내의 모든 NIfTI 파일(.nii, .nii.gz)에 대해 HD-BET(뇌 추출)을 수행합니다.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"🧠 HD-BET 뇌 추출 시작... (GPU 사용 여부: {torch.cuda.is_available()})")
    print(f"   입력: {input_dir}")
    print(f"   출력: {output_dir}")

    # HD-BET 파라미터 다운로드 (필요시)
    maybe_download_parameters()

    # 입력 파일 찾기 (.nii 또는 .nii.gz)
    # nifti_files 등 라이브러리 함수가 .nii를 못찾는 문제 해결
    input_files = sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith((".nii", ".nii.gz"))
        ]
    )

    if not input_files:
        print(f"⚠️ 경고: {input_dir} 에서 NIfTI 파일을 하나도 찾지 못했습니다.")
        return

    files_to_process = []
    output_files = []
    brain_mask_files = []

    for f in input_files:
        fname = os.path.basename(f)
        # 출력은 항상 .nii.gz로 저장 (HD-BET 표준)
        base_name = fname
        if base_name.endswith(".nii.gz"):
            base_name = base_name[:-7]
        elif base_name.endswith(".nii"):
            base_name = base_name[:-4]

        out_fname = base_name + ".nii.gz"
        out_path = os.path.join(output_dir, out_fname)
        mask_fname = base_name + "_bet.nii.gz"
        mask_path = os.path.join(output_dir, mask_fname)

        if os.path.exists(out_path):
            continue

        files_to_process.append(f)
        output_files.append(out_path)
        brain_mask_files.append(mask_path)

    if not files_to_process:
        print("✅ 모든 파일이 이미 처리되어 있습니다. 건너뜁니다.")
        return

    print(f"   처리할 파일 수: {len(files_to_process)} / {len(input_files)}")

    # 장치 설정
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Predictor 초기화
    predictor = get_hdbet_predictor(use_tta=False, device=device, verbose=True)

    # 1. Brain Mask 예측
    print("   [1/2] Brain Mask 생성 중...")
    # predict_from_files expects list of lists for inputs
    predictor.predict_from_files(
        [[i] for i in files_to_process],
        brain_mask_files,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=8,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )

    # 불필요한 json 파일 제거
    try:
        dir_name = os.path.dirname(brain_mask_files[0])
        for json_file in [
            "dataset.json",
            "plans.json",
            "predict_from_raw_data_args.json",
        ]:
            fpath = os.path.join(dir_name, json_file)
            if os.path.exists(fpath):
                os.remove(fpath)
    except Exception:
        pass

    # 2. Brain Extraction (Mask 적용)
    print("   [2/2] Mask 적용하여 뇌 영역 추출 중...")

    # 멀티프로세싱으로 처리
    pool_size = 4
    with Pool(pool_size) as p:
        p.starmap(apply_bet, zip(files_to_process, brain_mask_files, output_files))

    # 마스크 파일 삭제
    for mask_file in brain_mask_files:
        if os.path.exists(mask_file):
            os.remove(mask_file)

    print("✅ HD-BET 처리가 완료되었습니다!")


def convert_nii_folder_to_images_by_bet(input_dir, output_dir, slice_axis=2):
    """
    NIfTI(.nii 또는 .nii.gz) 파일을 2D PNG 파일로 슬라이스 변환
    """
    os.makedirs(output_dir, exist_ok=True)

    # 이미 변환된 파일이 있으면 생략
    existing_files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    if len(existing_files) > 0:
        print(f"⏭️ 이미 {len(existing_files)}개 PNG 파일 존재, 생략: {output_dir}")
        return

    # .nii 뿐만 아니라 .nii.gz (HD-BET 결과물)도 찾도록 수정
    nii_files = [f for f in os.listdir(input_dir) if f.endswith((".nii", ".nii.gz"))]

    if not nii_files:
        print(f"⚠️ 폴더에 NIfTI 파일이 없습니다: {input_dir}")
        return

    for nii_file in nii_files:
        nii_path = os.path.join(input_dir, nii_file)

        # nii 객체 로드
        nii_img = nib.load(nii_path)
        nii_data = nii_img.get_fdata()

        # HD-BET 결과물은 이름 뒤에 _bet 등이 붙을 수 있으므로 원본 환자명만 추출
        # 예: VK001.nii.gz -> VK001
        patient_name = nii_file.split(".")[0]
        if "_bet" in patient_name:
            patient_name = patient_name.replace("_bet", "")

        num_slices = nii_data.shape[slice_axis]

        for i in range(num_slices):
            if slice_axis == 2:
                slice_2d = nii_data[:, :, i]
            elif slice_axis == 1:
                slice_2d = nii_data[:, i, :]
            else:
                slice_2d = nii_data[i, :, :]

            # 16-bit 정규화 (0~65535)
            x_min, x_max = np.min(slice_2d), np.max(slice_2d)
            if x_max - x_min > 0:
                slice_2d = (slice_2d - x_min) / (x_max - x_min) * 65535
            else:
                slice_2d[:] = 0

            slice_2d = slice_2d.astype(np.uint16)

            # (이미 HD-BET으로 배경이 0이 되었으므로, 별도의 마스크 함수 필요 없음)

            output_path = os.path.join(output_dir, f"{patient_name}_slice_{i}.png")
            cv2.imwrite(output_path, slice_2d)

        print(f"변환 완료: {patient_name} -> {output_dir}")


def convert_nii_folder_to_images(input_dir, output_dir, slice_axis=2):
    """
    NIfTI(nii) 파일을 2D PNG 파일로 슬라이스 변환
    """
    # 출력 디렉토리 자동 생성
    os.makedirs(output_dir, exist_ok=True)

    # 이미 변환된 파일이 있으면 생략
    existing_files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    if len(existing_files) > 0:
        print(f"⏭️ 이미 {len(existing_files)}개 PNG 파일 존재, 생략: {output_dir}")
        return

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
            if slice_axis == 2:  # Axial (기본값)
                slice_2d = nii_data[:, :, i]
            elif slice_axis == 1:  # Sagittal (측면)
                slice_2d = nii_data[:, i, :]
            elif slice_axis == 0:  # Coronal (전후)
                slice_2d = nii_data[i, :, :]
            else:
                raise ValueError(
                    f"Invalid slice_axis: {slice_axis}. Must be 0, 1, or 2."
                )

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

            # mask = get_brain_mask(slice_2d)
            # slice_2d = slice_2d * mask

            output_path = os.path.join(output_dir, f"{patient_name}_slice_{i}.png")

            # 이미지 파일 저장
            # pylint: disable=no-member
            cv2.imwrite(output_path, slice_2d)

        # 16-bit 인지 확인하려면 다음 명령어 실행
        # file data/output_images/swi/VK001_slice_0.png
        print(f"변환 완료: {nii_path} → {output_dir}")


if __name__ == "__main__":
    print("🚀 NIfTI to PNG 변환 시작...")

    print("\n[1/3] HD-BET Brain Extraction (SWI -> SWI_BET)")
    # 먼저 HD-BET를 돌려서 .nii.gz 파일들을 생성해야 함
    run_hdbet_processing(SWI_INPUT_DIR, SWI_BET_OUTPUT_DIR)

    print("\n[2/3] SWI (Skull Stripped) -> PNG 변환")
    convert_nii_folder_to_images_by_bet(SWI_BET_OUTPUT_DIR, SWI_OUTPUT_DIR)

    print("\n[2/2] ROI (Original) -> PNG 변환")
    convert_nii_folder_to_images_by_bet(ROI_INPUT_DIR, ROI_OUTPUT_DIR)

    print("\n🎉 모든 변환 작업이 완료되었습니다!")
