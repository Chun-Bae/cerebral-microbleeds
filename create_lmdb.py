import lmdb
import os
from tqdm import tqdm

K = 5
SWI_DIR = "data/output_images/swi"
ROI_DIR = "data/output_images/roi"
SPLITS_DIR = "data/splits"
LMDB_DIR = "data/lmdb"


def create_lmdb(patient_ids, swi_dir, roi_dir, lmdb_path):
    """
    환자 ID 리스트를 해당 슬라이스를 받아서 LMDB로 저장
    """

    if os.path.exists(lmdb_path):
        print(f"이미 존재하므로 넘어갑니다. {lmdb_path}")
        return

    # TODO: 어떻게 생긴지 확인
    all_swi_files = sorted(os.listdir(swi_dir))
    valid_files = []

    for f in all_swi_files:
        # 파일명에서 환자 ID 추출
        # VK001_slice_0.png -> VK001
        patient_id = f.split("_slice_")[0]
        if patient_id in patient_ids:
            valid_files.append(f)

    print(f"[{lmdb_path}] 변환 시작: 총 {len(valid_files)}개 파일")

    # TODO: 이 주석에 대해서 좀더 해석
    # 예상 용량 계산 (이미지당 500KB * 개수 * 여유분)
    map_size = len(valid_files) * 500 * 1024 * 3
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, filename in enumerate(tqdm(valid_files)):
            swi_path = os.path.join(swi_dir, filename)
            roi_path = os.path.join(roi_dir, filename)

            # 바이너리로 열기
            with open(swi_path, "rb") as f:
                swi_bytes = f.read()
            with open(roi_path, "rb") as f:
                roi_bytes = f.read()

            # str_idx = "00001"
            str_idx = f"{idx:05d}"
            # 00000_image: VK001_slice_0.png의 바이너리
            txn.put(f"{str_idx}_image".encode(), swi_bytes)
            # 00000_mask: VK001_slice_0.png의 ROI 바이너리
            txn.put(f"{str_idx}_mask".encode(), roi_bytes)
            # 00000_name: "VK001_slice_0.png"
            txn.put(f"{str_idx}_name".encode(), filename.encode())

        # 메타 데이터
        txn.put("length".encode(), str(len(valid_files)).encode())

    env.close()
    print(f"✅ 생성 완료: {lmdb_path}")


def load_patient_ids(txt_path):
    """
    train.txt or test.txt에서 ID 로드
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def main():
    for fold_idx in range(K):
        fold_dir = os.path.join(SPLITS_DIR, f"fold_{fold_idx}")

        train_ids = load_patient_ids(os.path.join(fold_dir, "train.txt"))
        test_ids = load_patient_ids(os.path.join(fold_dir, "test.txt"))
        print(f"\n=== Fold {fold_idx} ===")
        print(f"Train 환자 수: {len(train_ids)}, Test 환자 수: {len(test_ids)}")
        # Train LMDB
        create_lmdb(
            train_ids,
            SWI_DIR,
            ROI_DIR,
            os.path.join(LMDB_DIR, f"fold_{fold_idx}", "train.lmdb"),
        )
        # Test LMDB
        create_lmdb(
            test_ids,
            SWI_DIR,
            ROI_DIR,
            os.path.join(LMDB_DIR, f"fold_{fold_idx}", "test.lmdb"),
        )

    holdout_path = os.path.join(SPLITS_DIR, "holdout_test.txt")
    if os.path.exists(holdout_path):
        holdout_ids = load_patient_ids(holdout_path)
        print(f"\n=== Hold-out Test ===")
        print(f"Hold-out 환자 수: {len(holdout_ids)}")
        create_lmdb(
            holdout_ids,
            SWI_DIR,
            ROI_DIR,
            os.path.join(LMDB_DIR, "holdout_test.lmdb")
        )

if __name__ == "__main__":
    main()

# data/lmdb/fold_0/train.lmdb
# ├── "00000_image" → [PNG 바이너리 bytes]
# ├── "00000_mask"  → [PNG 바이너리 bytes]  
# ├── "00000_name"  → b"VK001_slice_0.png"
# ├── "00001_image" → [PNG 바이너리 bytes]
# ├── "00001_mask"  → [PNG 바이너리 bytes]
# ├── "00001_name"  → b"VK001_slice_1.png"
# ├── ...
# ├── "15551_image" → [PNG 바이너리 bytes]
# ├── "15551_mask"  → [PNG 바이너리 bytes]
# ├── "15551_name"  → b"VK652_slice_71.png"
# └── "length"      → b"15552"