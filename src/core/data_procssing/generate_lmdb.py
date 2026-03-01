import os
import lmdb
from tqdm import tqdm
import config
from src.utils.logger import log

swi_dir = config.SWI_OUTPUT_DIR
roi_dir = config.ROI_OUTPUT_DIR
lmdb_dir = config.LMDB_DIR
split_dir = config.SPLITS_DIR
fixed_dir = config.FIXED_SPLIT_DIR
kfolds = config.K_FOLDS


def _load_patient_ids(txt_path):
    """txt 파일에서 ID 로드"""
    if not os.path.exists(txt_path):
        return set()
    with open(txt_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def _create_lmdb(patient_ids, lmdb_path):
    """
    주어진 환자 ID 리스트에 해당하는 PNG 슬라이스들을 모아 통째로 LMDB로 묶습니다.
    (Train 파이프라인에서 파일 IO 오버헤드를 줄이기 위한 최적화)
    """
    if os.path.exists(lmdb_path):
        log.info(f"이미 LMDB가 존재합니다. 생략: {lmdb_path}")
        return

    all_swi_files = sorted(os.listdir(swi_dir))
    valid_files = []

    for f in all_swi_files:
        # 파일명에서 환자 ID 추출 (예: VK001_slice_0.png -> VK001)
        patient_id = f.split("_slice_")[0]
        if patient_id in patient_ids:
            valid_files.append(f)

    if not valid_files:
        log.warning(f"{lmdb_path} 에 쓰일 valid 파일이 없습니다.")
        return

    log.info(f"[{lmdb_path}] 변환 시작: 총 {len(valid_files)}개 PNG 파일")

    # 예상 용량 계산 (이미지 1장당 넉넉하게 산정)
    map_size = len(valid_files) * 500 * 1024 * 3
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, filename in enumerate(tqdm(valid_files)):
            swi_path = os.path.join(swi_dir, filename)
            roi_path = os.path.join(roi_dir, filename)

            with open(swi_path, "rb") as f:
                swi_bytes = f.read()
            with open(roi_path, "rb") as f:
                roi_bytes = f.read()

            str_idx = f"{idx:05d}"
            txn.put(f"{str_idx}_image".encode(), swi_bytes)
            txn.put(f"{str_idx}_mask".encode(), roi_bytes)
            txn.put(f"{str_idx}_name".encode(), filename.encode())

        # 메타 데이터 저장
        txn.put("length".encode(), str(len(valid_files)).encode())

    env.close()
    log.success(f"생성 완료: {lmdb_path}")


def run_generate_lmdb_kfold():
    """Stratified K-Fold 방식의 텍스트 파일들을 기반으로 LMDB를 생성합니다."""
    log.info(f"Creating LMDBs for {kfolds}-Fold Cross Validation...")

    for fold_idx in range(kfolds):
        fold_txt_dir = os.path.join(split_dir, f"fold_{fold_idx}")
        fold_lmdb_dir = os.path.join(lmdb_dir, f"fold_{fold_idx}")

        if not os.path.exists(fold_txt_dir):
            log.warning(
                f"{fold_txt_dir} 가 없습니다. K-Fold 데이터가 먼저 생성되어야 합니다."
            )
            continue

        train_ids = _load_patient_ids(os.path.join(fold_txt_dir, "train.txt"))
        test_ids = _load_patient_ids(os.path.join(fold_txt_dir, "test.txt"))

        log.info(f"=== Fold {fold_idx} ===")
        log.info(f"Train 환자: {len(train_ids)} 명, Test 환자: {len(test_ids)} 명")

        _create_lmdb(train_ids, os.path.join(fold_lmdb_dir, "train.lmdb"))
        _create_lmdb(test_ids, os.path.join(fold_lmdb_dir, "test.lmdb"))


def run_generate_lmdb_fixed():
    """Fixed_Split (Train/Valid/Test) 텍스트 파일들을 기반으로 LMDB를 생성합니다."""
    log.info("Creating LMDBs for Fixed Split (Train/Valid/Holdout)...")

    if not os.path.exists(fixed_dir):
        log.warning(
            f"{fixed_dir} 가 없습니다. Fixed-Split 데이터가 먼저 생성되어야 합니다."
        )
        return

    train_ids = _load_patient_ids(os.path.join(fixed_dir, "train.txt"))
    valid_ids = _load_patient_ids(os.path.join(fixed_dir, "valid.txt"))
    test_ids = _load_patient_ids(os.path.join(fixed_dir, "test.txt"))

    log.info("=== Fixed Split ===")
    log.info(f"Train 환자: {len(train_ids)} 명")
    log.info(f"Valid 환자: {len(valid_ids)} 명")
    log.info(f"Test  환자: {len(test_ids)} 명")

    target_lmdb_dir = os.path.join(lmdb_dir, "fixed_split")

    _create_lmdb(train_ids, os.path.join(target_lmdb_dir, "train.lmdb"))
    _create_lmdb(valid_ids, os.path.join(target_lmdb_dir, "valid.lmdb"))
    _create_lmdb(test_ids, os.path.join(target_lmdb_dir, "test.lmdb"))
