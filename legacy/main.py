"""
main.py - CMB(뇌미세출혈) 탐지 모델 메인 실행 스크립트

이 스크립트는 CMB 탐지 학습 파이프라인의 진입점(Entry Point)입니다.
실행 시 다음 작업을 순차적으로 수행합니다:
1. 데이터 전처리 및 로딩
2. 데이터 증강(Augmentation) 검증
3. 모델 초기화
4. 학습 실행
5. 평가 및 결과 저장

사용법:
    python main.py
"""

# ==========================================
# [환경 설정] Albumentations 경고 비활성화
# 반드시 다른 import 전에 설정해야 함
# ==========================================
from trainer import train_model                                    # 학습 함수
from data_prep import prepare_data, verify_augmentation, get_transforms  # 데이터 전처리
from model import SSD_FE                                           # SSD-FE 모델
from utils import Logger                                           # 로깅 유틸리티
import torch.optim as optim
import torch
import datetime
import sys
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# ==========================================
# [라이브러리 임포트]
# ==========================================

# 커스텀 모듈 임포트


# ==========================================
# [설정] 하이퍼파라미터
# ==========================================
BATCH_SIZE = 8          # 배치 크기 (GPU 메모리에 따라 조절)
NUM_WORKERS = 4         # 데이터 로딩 워커 수 (CPU 코어 수에 맞게 조절, Windows: 0~8 추천)
NUM_EPOCHS = 200        # 최대 학습 에폭 수
LEARNING_RATE = 1e-4    # 학습률 (Adam 옵티마이저 기본값)
EVAL_INTERVAL = 10      # mAP 평가 주기 (에폭 단위)
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # GPU 사용 여부 자동 감지
# ==========================================


if __name__ == '__main__':
    """
    메인 실행 블록
    - Windows 멀티프로세싱 지원을 위해 if __name__ == '__main__': 내부에서 실행
    """

    # Windows 멀티프로세싱 가드
    import multiprocessing
    multiprocessing.freeze_support()

    # ==========================================
    # [1] 결과 저장 폴더 생성
    # ==========================================
    # 실행 시간 기준으로 고유한 폴더명 생성
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", f"run_{run_timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # ==========================================
    # [2] 로깅 설정
    # ==========================================
    # 표준 출력을 가로채서 파일과 콘솔에 동시에 출력
    # 모든 print 문이 자동으로 log.txt에도 기록됨
    sys.stdout = Logger(os.path.join(result_dir, "log.txt"))

    # ==========================================
    # [2-1] wandb 초기화 (선택적)
    # ==========================================
    try:
        import wandb
        wandb.init(
            entity="bisnelstudent-sejong-university",  # 팀/사용자명
            project="CMB-detection",                   # 프로젝트명
            name=f"run_{run_timestamp}",               # 실행 이름
            config={
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "eval_interval": EVAL_INTERVAL,
                "architecture": "SSD-FE (VGG16)",
                "dataset": "CMB",
            }
        )
        print("✅ wandb 초기화 완료!")
    except Exception as e:
        print(f"⚠️ wandb 초기화 실패 (로컬 로깅만 사용): {e}")

    print(f"=== 실행 시작: {run_timestamp} ===")
    print(f"모든 결과(로그, 그래프 등)는 다음 폴더에 저장됩니다: {result_dir}")
    print(f"사용 디바이스: {DEVICE}")

    # ==========================================
    # [3] 데이터 준비
    # ==========================================
    # 출력 디렉토리 목록 (SWI/ROI, Train/Test)
    output_dirs = [
        "output_images/swi",       # Train SWI 이미지
        "output_images/roi",       # Train ROI 마스크
        "output_images/swi_test",  # Test SWI 이미지
        "output_images/roi_test"   # Test ROI 마스크
    ]

    # 데이터 로더 생성 (NIfTI 변환, Train/Test 분할, LMDB 생성 포함)
    train_loader, test_loader, train_dataset = prepare_data(
        BATCH_SIZE, NUM_WORKERS, output_dirs
    )

    # ==========================================
    # [4] 데이터 증강 검증
    # ==========================================
    # Train용 Transform 가져오기
    train_transform, _ = get_transforms()

    # 증강 전/후 비교 이미지 저장 (train_check 폴더)
    verify_augmentation(train_dataset, train_transform)

    # ==========================================
    # [5] 모델 초기화
    # ==========================================
    # SSD-FE 모델 생성 (Anchor 기반 다중 CMB 탐지)
    model = SSD_FE(
        anchor_scales=[0.1, 0.2, 0.4],  # CMB 크기에 맞춘 anchor 스케일
        anchor_ratios=[1.0]              # 정사각형 anchor
    ).to(DEVICE)
    print(f"\\n모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Anchor 개수: {model.num_anchors}개 (위치당)")

    # ==========================================
    # [6] 손실 함수 설정
    # ==========================================
    # 분류 손실: CrossEntropyLoss
    # - 픽셀별 분류 (배경 vs 병변)
    # - Softmax + NLLLoss 결합
    criterion = torch.nn.CrossEntropyLoss()

    # 바운딩 박스 손실: SmoothL1Loss (trainer.py에서 정의)

    # ==========================================
    # [7] 옵티마이저 설정
    # ==========================================
    # Adam 옵티마이저
    # - 적응적 학습률 (Adaptive Learning Rate)
    # - 모멘텀 + RMSprop 결합
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # [8] 학습률 스케줄러 (현재 미사용)
    # ==========================================
    scheduler = None
    # 필요시 아래와 같이 활성화 가능:
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # ==========================================
    # [9] 학습 시작
    # ==========================================
    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        result_dir=result_dir,
        eval_interval=EVAL_INTERVAL
    )
