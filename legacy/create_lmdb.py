import lmdb
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm


def create_lmdb(swi_folder, roi_folder, lmdb_path, check_map_size=True):
    """
    이미지와 마스크를 읽어서 LMDB 파일로 저장합니다.
    (PNG 바이너리 형태 그대로 저장 -> 읽을 때 디코딩)
    """
    if os.path.exists(lmdb_path):
        print(f"⚠️ 이미 존재함: {lmdb_path}")
        return

    # 1. 파일 목록 매칭
    swi_files = sorted(os.listdir(swi_folder))
    roi_files = sorted(os.listdir(roi_folder))

    # 교집합만 추출
    valid_files = [f for f in swi_files if f in roi_files]
    print(f"[{lmdb_path}] 변환 시작: 총 {len(valid_files)}개 파일")

    # 2. 예상 용량 계산 (대략적으로)
    # 이미지 1개당 200KB 가정 * 개수 * 여유분
    map_size = len(valid_files) * 500 * 1024 * 3

    # 3. LMDB 생성
    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, filename in enumerate(tqdm(valid_files)):
            swi_path = os.path.join(swi_folder, filename)
            roi_path = os.path.join(roi_folder, filename)

            # 바이너리로 읽기 (가장 빠름 + 용량 절약)
            with open(swi_path, 'rb') as f:
                swi_bytes = f.read()
            with open(roi_path, 'rb') as f:
                roi_bytes = f.read()

            # 키 생성 (index 기반)
            # {index}_image, {index}_mask, {index}_name
            # index는 5자리 숫자로 (00001)
            str_idx = f"{idx:05d}"

            txn.put(f"{str_idx}_image".encode(), swi_bytes)
            txn.put(f"{str_idx}_mask".encode(), roi_bytes)
            txn.put(f"{str_idx}_name".encode(), filename.encode())

        # 메타데이터 저장: 총 개수
        txn.put("length".encode(), str(len(valid_files)).encode())

    env.close()
    print(f"✅ 생성 완료: {lmdb_path}")


if __name__ == '__main__':
    # 폴더 경로 설정 (main.py와 동일하게)
    output_dirs = {
        "train_swi": "output_images/swi",
        "train_roi": "output_images/roi",
        "test_swi": "output_images/swi_test",
        "test_roi": "output_images/roi_test"
    }

    # Train LMDB
    create_lmdb(output_dirs["train_swi"],
                output_dirs["train_roi"], "train.lmdb")

    # Test LMDB
    create_lmdb(output_dirs["test_swi"], output_dirs["test_roi"], "test.lmdb")
