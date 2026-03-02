import os
import sys
import torch
from multiprocessing import Pool
import config
from src.utils.logger import log

# === Global ===
device = config.DEVICE
input_dir = config.SWI_INPUT_DIR
output_dir = config.SWI_BET_OUTPUT_DIR

# === HD-BET Path ===
current_dir = os.path.dirname(os.path.abspath(__file__))
hd_bet_path = os.path.abspath(
    os.path.join(current_dir, "../../../../third_party/HD-BET")
)

if hd_bet_path not in sys.path:
    sys.path.append(hd_bet_path)

try:
    from third_party.HD_BET.checkpoint_download import maybe_download_parameters
    from third_party.HD_BET.hd_bet_prediction import hdbet_predict, apply_bet, get_hdbet_predictor
    from third_party.HD_BET.paths import folder_with_parameter_files
except ImportError:
    pass


def run_skull_stripping():
    """
    폴더 내의 모든 NIfTI 파일(.nii, .nii.gz)에 대해 HD-BET(뇌 추출)을 수행합니다.
    """
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"HD-BET 뇌 추출 시작... (GPU 사용 여부: {torch.cuda.is_available()})")
    log.info(f"입력: {input_dir}")
    log.info(f"출력: {output_dir}")

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
        log.warning(f"{input_dir} 에서 NIfTI 파일을 하나도 찾지 못했습니다.")
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
        log.info("모든 파일이 이미 처리되어 있습니다. 건너뜁니다.")
        return

    log.info(f"처리할 파일 수: {len(files_to_process)} / {len(input_files)}")

    # Predictor 초기화
    predictor = get_hdbet_predictor(use_tta=False, device=device, verbose=True)

    # 1. Brain Mask 예측
    log.info("[1/2] Brain Mask 생성 중...")
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
    log.info("[2/2] Mask 적용하여 뇌 영역 추출 중...")

    # 멀티프로세싱으로 처리
    pool_size = 4
    try:
        with Pool(pool_size) as p:
            p.starmap(apply_bet, zip(files_to_process, brain_mask_files, output_files))
    except Exception as e:
        log.error(f"Mask 적용 중 에러 발생: {e}")
        # 일부 실패해도 진행되도록

    # 마스크 파일 삭제
    for mask_file in brain_mask_files:
        if os.path.exists(mask_file):
            os.remove(mask_file)

    log.success("HD-BET 처리가 완료되었습니다!")
