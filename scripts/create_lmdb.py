import lmdb
import os
from tqdm import tqdm

import config

SWI_DIR = config.SWI_OUTPUT_DIR
ROI_DIR = config.ROI_OUTPUT_DIR
LMDB_DIR = config.LMDB_DIR


def create_lmdb(patient_ids, swi_dir, roi_dir, lmdb_path):
    """
    환자 ID 리스트를 해당 슬라이스를 받아서 LMDB로 저장
    """

    if os.path.exists(lmdb_path):
        print(f"이미 존재하므로 넘어갑니다. {lmdb_path}")
        return

    all_swi_files = sorted(os.listdir(swi_dir))
    valid_files = []

    for f in all_swi_files:
        # 파일명에서 환자 ID 추출
        # VK001_slice_0.png -> VK001
        patient_id = f.split("_slice_")[0]
        if patient_id in patient_ids:
            valid_files.append(f)

    print(f"[{lmdb_path}] 변환 시작: 총 {len(valid_files)}개 파일")

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

            str_idx = f"{idx:05d}"
            txn.put(f"{str_idx}_image".encode(), swi_bytes)
            txn.put(f"{str_idx}_mask".encode(), roi_bytes)
            txn.put(f"{str_idx}_name".encode(), filename.encode())

        # 메타 데이터
        txn.put("length".encode(), str(len(valid_files)).encode())

    env.close()
    print(f"✅ 생성 완료: {lmdb_path}")


def load_patient_ids(txt_path):
    """
    txt 파일에서 ID 로드
    """
    if not os.path.exists(txt_path):
        return set()
    with open(txt_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def main():
    if config.USE_K_FOLD:
        print(f"🔄 Creating LMDBs for {config.K_FOLDS}-Fold Cross Validation...")
        for fold_idx in range(config.K_FOLDS):
            fold_dir = os.path.join(config.SPLITS_DIR, f"fold_{fold_idx}")

            # Load IDs
            train_ids = load_patient_ids(os.path.join(fold_dir, "train.txt"))
            test_ids = load_patient_ids(os.path.join(fold_dir, "test.txt"))

            print(f"\n=== Fold {fold_idx} ===")
            print(f"Train 환자 수: {len(train_ids)}, Test 환자 수: {len(test_ids)}")

            # Create LMDBs
            create_lmdb(
                train_ids,
                SWI_DIR,
                ROI_DIR,
                os.path.join(LMDB_DIR, f"fold_{fold_idx}", "train.lmdb"),
            )
            create_lmdb(
                test_ids,
                SWI_DIR,
                ROI_DIR,
                os.path.join(LMDB_DIR, f"fold_{fold_idx}", "test.lmdb"),
            )

    else:
        print("🔄 Creating LMDBs for Fixed Split (Train/Valid/Holdout)...")
        fixed_dir = config.FIXED_SPLIT_DIR

        train_ids = load_patient_ids(os.path.join(fixed_dir, "train.txt"))
        valid_ids = load_patient_ids(os.path.join(fixed_dir, "valid.txt"))
        test_ids = load_patient_ids(os.path.join(fixed_dir, "test.txt"))

        print(f"\n=== Fixed Split ===")
        print(f"Train: {len(train_ids)}")
        print(f"Valid: {len(valid_ids)}")
        print(f"Test:  {len(test_ids)}")

        # Save to data/lmdb/fixed_split/...
        target_lmdb_dir = os.path.join(LMDB_DIR, "fixed_split")

        create_lmdb(
            train_ids, SWI_DIR, ROI_DIR, os.path.join(target_lmdb_dir, "train.lmdb")
        )
        create_lmdb(
            valid_ids, SWI_DIR, ROI_DIR, os.path.join(target_lmdb_dir, "valid.lmdb")
        )
        create_lmdb(
            test_ids, SWI_DIR, ROI_DIR, os.path.join(target_lmdb_dir, "test.lmdb")
        )


if __name__ == "__main__":
    main()
