import os
from tqdm import tqdm
import nibabel as nib
import config
from src.utils.logger import log

# === Global ===
input_dir = config.SWI_BET_OUTPUT_DIR
output_dir = config.SWI_N4_OUTPUT_DIR

# Try importing ANTsPy
try:
    import ants

    HAS_ANTS = True
except ImportError:
    HAS_ANTS = False


def run_n4_correction():
    """
    폴더 내의 모든 NIfTI 파일(.nii, .nii.gz)에 대해 ANTs를 사용한 N4 보정을 수행하고 새로운 NIfTI로 저장합니다.
    """

    os.makedirs(output_dir, exist_ok=True)

    log.info(
        f"N4 의존성 상태 파악됨: {'사용 가능' if HAS_ANTS else '사용 불가 (ANTs가 설치되지 않음)'}"
    )
    if not HAS_ANTS:
        log.warning(
            "ANTs가 없어서 N4 보정을 건너뛰고 입력 데이터를 바로 출력 디렉토리로 복사합니다."
        )

    log.info("N4 Bias Field Correction 시작...")
    log.info(f"입력: {input_dir}")
    log.info(f"출력: {output_dir}")

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

    for f in input_files:
        fname = os.path.basename(f)
        out_path = os.path.join(output_dir, fname)

        if os.path.exists(out_path):
            continue

        files_to_process.append((f, out_path))

    if not files_to_process:
        log.info("모든 파일이 이미 N4 보정되어 있습니다. 건너뜁니다.")
        return

    log.info(f"처리할 파일 수: {len(files_to_process)} / {len(input_files)}")

    for in_path, out_path in tqdm(
        files_to_process, desc="Applying N4 Correction", unit="vol"
    ):
        try:
            # 1. NIfTI 로드
            nii_img = nib.load(in_path)
            nii_data = nii_img.get_fdata()

            # 2. N4 보정 (ANTs가 설치된 경우에만)
            if HAS_ANTS:
                try:
                    img = ants.from_numpy(nii_data)
                    mask = ants.get_mask(img)
                    img_n4 = ants.n4_bias_field_correction(
                        img, mask=mask, shrink_factor=2
                    )
                    nii_data = img_n4.numpy()
                except Exception as e:
                    log.warning(
                        f"{os.path.basename(in_path)} N4 Error (원본 유지): {e}"
                    )

            # 3. 보정된(또는 원본) 데이터를 새로운 NIfTI로 저장
            new_nii_img = nib.Nifti1Image(nii_data, nii_img.affine, nii_img.header)
            nib.save(new_nii_img, out_path)

        except Exception as e:
            log.error(f"{os.path.basename(in_path)} 파일 보정 중 에러 발생: {e}")
            continue

    log.success("N4 Bias Field Correction 처리가 완료되었습니다!")
