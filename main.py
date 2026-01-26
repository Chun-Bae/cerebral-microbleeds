import multiprocessing
import datetime
import os
import sys
import torch
from utils import Logger
from convert_nii_folder_to_images import convert_nii_folder_to_images

# [설정] Albumentations 업데이트 경고 끄기 (Import 이전에 설정해야 함)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
EVAL_INTERVAL = 10
SPLIT_RATIO = 0.2
K = 5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dirs = [
    "data/output_images/swi",
    "data/output_images/roi",
    "data/output_images/swi_test",
    "data/output_images/roi_test",
]

def main():
    multiprocessing.freeze_support()
    
    result_dir = get_results_dir()
    set_results_dir(result_dir)
    set_logger(result_dir)

    for d in output_dirs:
        os.makedirs(d, exist_ok=True)

    convert_nii_folder_to_images("data/samsung_data/swi", "data/output_images/swi")
    convert_nii_folder_to_images("data/samsung_data/roi", "data/output_images/roi")

def get_results_dir():
    """
    결과물 dir 이름 가져오기
    """
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d(%Hh-%Mm-%Ss)")
    result_dir = os.path.join("results", f"run_{run_timestamp}")
    return result_dir

def set_results_dir(result_dir):
    os.makedirs(result_dir, exist_ok=True)

def set_logger(result_dir):
    # 타임 스탬프 출력 및 로그 파일 생성
    sys.stdout = Logger(os.path.join(result_dir,"log.txt"))
    print(f"Results directory created at: {result_dir}")

if __name__ == "__main__":
    main()