"""
inference.py - CMB 탐지 모델 독립 추론 스크립트

학습 완료 후 저장된 가중치로 모델을 평가합니다.
checkpoints 폴더 내 가중치 파일 목록에서 선택 가능합니다.

사용법:
    python inference.py
    
    실행 후 설정값 입력 및 가중치 선택

기능:
1. 배치 사이즈, 워커 수 설정 (Enter로 기본값 사용)
2. 저장된 가중치 목록 표시 및 선택
3. Train/Test 세트 평가 (mAP, FP/CMB)
4. FROC 곡선, Confusion Matrix 저장
"""

from evaluate import evaluate_model
from dataset import CMBsDatasetLMDB, CMBsDataset, get_transforms, detection_collate
from model import SSD_FE
from torch.utils.data import DataLoader
import torch
import sys
import os
# [설정] Albumentations 업데이트 경고 끄기 (Import 이전에 설정해야 함)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


# ==========================================
# [설정] 기본값 (main.py와 일치)
# ==========================================
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================


def get_latest_run_dir(base_dir="results"):
    """
    results 폴더 내에서 가장 최근에 생성된 폴더를 찾습니다.
    """
    if not os.path.exists(base_dir):
        return None

    dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not dirs:
        return None

    latest_dir = max(dirs, key=os.path.getmtime)
    return latest_dir


def list_checkpoints(checkpoint_dir="checkpoints"):
    """
    checkpoints 폴더 내 모든 .pth 파일을 찾아 목록화합니다.
    """
    if not os.path.exists(checkpoint_dir):
        return []

    pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    pth_files.sort(key=lambda f: os.path.getmtime(
        os.path.join(checkpoint_dir, f)), reverse=True)

    result = []
    for i, filename in enumerate(pth_files, 1):
        full_path = os.path.join(checkpoint_dir, filename)
        result.append((i, filename, full_path))

    return result


def select_checkpoint(checkpoints):
    """
    사용자에게 체크포인트 목록을 보여주고 선택받습니다.
    """
    if not checkpoints:
        print("❌ checkpoints 폴더에 가중치 파일(.pth)이 없습니다.")
        return None

    print("\n" + "=" * 50)
    print("         사용 가능한 가중치 파일 목록")
    print("=" * 50)

    for num, filename, full_path in checkpoints:
        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        print(f"  [{num}] {filename} ({size_mb:.1f} MB)")

    print("=" * 50)

    while True:
        try:
            choice = input(f">> 번호를 선택하세요 (1-{len(checkpoints)}): ").strip()
            choice_num = int(choice)

            if 1 <= choice_num <= len(checkpoints):
                selected = checkpoints[choice_num - 1]
                print(f"✅ 선택됨: {selected[1]}")
                return selected[2]
            else:
                print(f"❌ 1-{len(checkpoints)} 사이의 번호를 입력하세요.")
        except ValueError:
            print("❌ 숫자를 입력하세요.")


def get_user_settings():
    """
    사용자로부터 배치 사이즈와 워커 수를 입력받습니다.
    Enter를 누르면 기본값을 사용합니다.

    Returns:
        (batch_size, num_workers) 튜플
    """
    print("\n" + "=" * 50)
    print("         설정값 입력 (Enter: 기본값 사용)")
    print("=" * 50)

    # 배치 사이즈 입력
    batch_input = input(f">> 배치 사이즈 [{DEFAULT_BATCH_SIZE}]: ").strip()
    if batch_input == "":
        batch_size = DEFAULT_BATCH_SIZE
    else:
        try:
            batch_size = int(batch_input)
        except ValueError:
            print(f"   ⚠️ 잘못된 입력. 기본값 {DEFAULT_BATCH_SIZE} 사용")
            batch_size = DEFAULT_BATCH_SIZE

    # 워커 수 입력
    worker_input = input(f">> 워커 수 [{DEFAULT_NUM_WORKERS}]: ").strip()
    if worker_input == "":
        num_workers = DEFAULT_NUM_WORKERS
    else:
        try:
            num_workers = int(worker_input)
        except ValueError:
            print(f"   ⚠️ 잘못된 입력. 기본값 {DEFAULT_NUM_WORKERS} 사용")
            num_workers = DEFAULT_NUM_WORKERS

    print(f"   ✅ 설정: batch_size={batch_size}, num_workers={num_workers}")
    return batch_size, num_workers


def load_model(model_path, device):
    """
    모델을 로드합니다.
    """
    model = SSD_FE(num_classes=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f"   📍 저장된 에폭: {checkpoint['epoch'] + 1}")
        if 'loss' in checkpoint:
            print(f"   📍 저장된 손실: {checkpoint['loss']:.6f}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def get_data_loaders(batch_size, num_workers):
    """
    Train/Test 데이터 로더를 생성합니다.
    """
    train_transform, test_transform = get_transforms()

    if os.path.exists("train.lmdb") and os.path.exists("test.lmdb"):
        print("🚀 [Fast Mode] LMDB 사용")
        train_dataset = CMBsDatasetLMDB(
            "train.lmdb", transform=train_transform)
        test_dataset = CMBsDatasetLMDB("test.lmdb", transform=test_transform)
    else:
        print("🐢 [Normal Mode] 일반 파일 로딩")
        train_dataset = CMBsDataset(
            "output_images/swi", "output_images/roi",
            transform=train_transform
        )
        test_dataset = CMBsDataset(
            "output_images/swi_test", "output_images/roi_test",
            transform=test_transform
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        collate_fn=detection_collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        collate_fn=detection_collate
    )

    return train_loader, test_loader


def run_inference(batch_size=None, num_workers=None, model_path=None, interactive=True):
    """
    추론 실행 함수

    Args:
        batch_size: 배치 사이즈 (None이면 기본값 또는 사용자 입력)
        num_workers: 워커 수 (None이면 기본값 또는 사용자 입력)
        model_path: 모델 경로 (None이면 사용자 선택)
        interactive: True면 사용자 입력 모드, False면 자동 모드
    """
    print("\n" + "=" * 50)
    print("       CMB 탐지 모델 추론 (Inference)")
    print("=" * 50)

    # ==========================================
    # [1] 최신 결과 폴더 찾기
    # ==========================================
    result_dir = get_latest_run_dir()
    if result_dir is None:
        print("❌ 'results' 폴더를 찾을 수 없거나 학습 기록이 없습니다.")
        return

    print(f"📂 발견된 최신 결과 폴더: {result_dir}")

    # ==========================================
    # [2] 설정값 결정
    # ==========================================
    if interactive and batch_size is None and num_workers is None:
        # 사용자 입력 모드
        batch_size, num_workers = get_user_settings()
    else:
        # 자동 모드 (인자로 받은 값 또는 기본값)
        batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE
        num_workers = num_workers if num_workers is not None else DEFAULT_NUM_WORKERS

    # ==========================================
    # [3] 체크포인트 선택
    # ==========================================
    if model_path is None:
        checkpoints = list_checkpoints()
        model_path = select_checkpoint(checkpoints)
        if model_path is None:
            return

    # ==========================================
    # [4] 데이터 로드
    # ==========================================
    print("\n📦 데이터셋 로딩 중...")
    train_loader, test_loader = get_data_loaders(batch_size, num_workers)

    # ==========================================
    # [5] 모델 로드
    # ==========================================
    print("\n🧠 모델 초기화 및 가중치 로드...")
    model = load_model(model_path, DEVICE)
    print("✅ 모델 로드 완료!")

    # ==========================================
    # [6] 평가 실행
    # ==========================================
    eval_save_dir = os.path.join(result_dir, "inference_results")

    evaluate_model(
        model,
        train_loader,
        test_loader,
        DEVICE,
        save_dir=eval_save_dir,
        prefix="inf"
    )


def main():
    """
    메인 실행 함수 (독립 실행 시)
    """
    run_inference(interactive=True)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
