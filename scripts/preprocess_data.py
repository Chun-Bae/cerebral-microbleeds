"""
This module converts NIfTI (.nii) files in a folder to a series of 2D PNG images.
"""

import os
import sys
import nibabel as nib
import numpy as np
import cv2
import collections
import torch
import glob
from tqdm import tqdm
from multiprocessing import Pool

# HD-BET 패키지 경로 추가 (현재 스크립트 위치 기준으로 ../HD-BET)
current_dir = os.path.dirname(os.path.abspath(__file__))
hd_bet_path = os.path.join(current_dir, "../HD-BET")
if hd_bet_path not in sys.path:
    sys.path.append(hd_bet_path)

try:
    from HD_BET.checkpoint_download import maybe_download_parameters

    # get_hdbet_predictor 추가 import
    from HD_BET.hd_bet_prediction import hdbet_predict, apply_bet, get_hdbet_predictor
    from HD_BET.paths import folder_with_parameter_files
except ImportError:
    pass

# Try importing ANTsPy
try:
    import ants

    HAS_ANTS = True
except ImportError:
    HAS_ANTS = False

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
    try:
        with Pool(pool_size) as p:
            p.starmap(apply_bet, zip(files_to_process, brain_mask_files, output_files))
    except Exception as e:
        print(f"❌ Mask 적용 중 에러 발생: {e}")
        # 일부 실패해도 진행되도록

    # 마스크 파일 삭제
    for mask_file in brain_mask_files:
        if os.path.exists(mask_file):
            os.remove(mask_file)

    print("✅ HD-BET 처리가 완료되었습니다!")


def apply_n4_correction(nii_data):
    """ANTs N4 Bias Field Correction 적용"""
    if not HAS_ANTS:
        return nii_data
    try:
        img = ants.from_numpy(nii_data)
        mask = ants.get_mask(img)
        img_n4 = ants.n4_bias_field_correction(img, mask=mask, shrink_factor=2)
        return img_n4.numpy()
    except Exception as e:
        print(f"  N4 Error: {e}")
        return nii_data


def convert_nii_folder_to_images(input_dir, output_dir, slice_axis=2, apply_n4=False):
    """
    NIfTI(nii) 파일을 2D PNG 파일로 슬라이스 변환
    apply_n4: True일 경우 N4 보정 수행
    """
    os.makedirs(output_dir, exist_ok=True)

    # 이미 변환된 파일 확인
    existing_files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    if len(existing_files) > 0:
        print(f"⏭️ 이미 {len(existing_files)}개 PNG 파일 존재, 생략: {output_dir}")
        return

    # .nii 및 .nii.gz 모두 지원
    nii_files = [f for f in os.listdir(input_dir) if f.endswith((".nii", ".nii.gz"))]

    if not nii_files:
        print(f"⚠️ 폴더에 NIfTI 파일이 없습니다: {input_dir}")
        return

    if apply_n4 and HAS_ANTS:
        print(f"✨ N4 Bias Field Correction 활성화됨 (총 {len(nii_files)}개 파일)")

    iterator = tqdm(nii_files, desc="Converting NIfTI", unit="vol")
    for nii_file in iterator:
        nii_path = os.path.join(input_dir, nii_file)

        try:
            nii_img = nib.load(nii_path)
            nii_data = nii_img.get_fdata()
        except Exception:
            continue

        patient_name = nii_file.split(".")[0]
        if "_bet" in patient_name:
            patient_name = patient_name.replace("_bet", "")
        # 확장자 처리가 split(".")로 하면 .nii.gz의 경우 이름에 .nii가 남을 수 있음
        if patient_name.endswith(".nii"):
            patient_name = patient_name[:-4]

        # N4 Bias Correction
        if apply_n4 and HAS_ANTS:
            nii_data = apply_n4_correction(nii_data)

        num_slices = nii_data.shape[slice_axis]

        for i in range(num_slices):
            if slice_axis == 2:
                slice_2d = nii_data[:, :, i]
            elif slice_axis == 1:
                slice_2d = nii_data[:, i, :]
            else:
                slice_2d = nii_data[i, :, :]

            x_min, x_max = np.min(slice_2d), np.max(slice_2d)
            if x_max - x_min > 0:
                slice_2d = (slice_2d - x_min) / (x_max - x_min) * 65535
            else:
                slice_2d[:] = 0

            slice_2d = slice_2d.astype(np.uint16)

            output_path = os.path.join(output_dir, f"{patient_name}_slice_{i}.png")
            cv2.imwrite(output_path, slice_2d)

    print(f"✅ 폴더 변환 완료: {input_dir} -> {output_dir}")


def convert_nii_folder_to_images_by_bet(
    input_dir, output_dir, slice_axis=2, apply_n4=False
):
    """
    Wrapper for compatibility
    """
    convert_nii_folder_to_images(input_dir, output_dir, slice_axis, apply_n4)


if __name__ == "__main__":
    print("🚀 NIfTI to PNG 변환 시작...")

    print("\n[1/3] HD-BET Brain Extraction (SWI -> SWI_BET)")
    # run_hdbet_processing(SWI_INPUT_DIR, SWI_BET_OUTPUT_DIR)

    print("\n[2/3] SWI (Skull Stripped) -> PNG 변환")
    # convert_nii_folder_to_images(SWI_BET_OUTPUT_DIR, SWI_OUTPUT_DIR, apply_n4=True)

    print("\n[2/2] ROI (Original) -> PNG 변환")
    # convert_nii_folder_to_images(ROI_INPUT_DIR, ROI_OUTPUT_DIR, apply_n4=False)

    print("\n🎉 모든 변환 작업이 완료되었습니다!")
