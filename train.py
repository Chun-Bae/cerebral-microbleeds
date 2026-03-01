import argparse
import os
import datetime
import sys

import config
from src.pipelines.data_pipeline import DataPipeLine
from src.pipelines.train_pipeline import TrainPipeline
from src.utils.logger import Logger


def main():
    parser = argparse.ArgumentParser(description="CMB Detection Training")
    parser.add_argument(
        "--prepare_data",
        action="store_true",
        help="Run data preprocessing and LMDB creation before training",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional path to pretrained weights to resume training",
    )
    parser.add_argument(
        "--fixed_split",
        action="store_true",
        help="Force use fixed split instead of K-Fold",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Specific fold indices to train (e.g., --folds 0 1 2). If not provided, runs all folds.",
    )
    args = parser.parse_args()

    # 0. 가중치 이름 입력
    while True:
        run_name = input(
            "\n📝 이번 학습의 이름(목적/버전)을 입력해주세요 (예: ssd_fe_base, lr_tuning): "
        ).strip()
        if run_name:
            # 안전한 디렉토리 명으로 변환 (공백은 언더바로)
            run_name = run_name.replace(" ", "_")
            break
        print("⚠️ 학습 이름은 필수 입력 사항입니다! 공란으로 넘길 수 없습니다.")

    # 1. 로거 및 결과 폴더 준비
    run_timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    result_dir = os.path.join(config.RESULTS_DIR, f"train_{run_name}_{run_timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    log = Logger(os.path.join(result_dir, "train_log.txt"))

    log.info(f"========== TRAINING START ==========")
    log.info(f"Result Directory: {result_dir}")
    log.info(f"====================================\n")

    # 1. 데이터 파이프라인 (선택적)
    if args.prepare_data:
        log.info("🚀 [Step 1] 데이터 파이프라인 가동 (전처리 및 DB 생성)...")
        data_pipeline = DataPipeLine()
        data_pipeline.run()
    else:
        log.info("⏩ [Step 1] 데이터 전처리 스킵 (이미 구축된 LMDB 사용 가정)")

    # 2. 훈련 대상 Fold 파악
    use_fixed_split = args.fixed_split or not getattr(config, "USE_K_FOLD", False)
    if use_fixed_split:
        folds_to_run = [0]
        log.info("📌 Fixed Split 모드로 학습 진행합니다 (Fold 0)")
    else:
        if args.folds is not None:
            folds_to_run = args.folds
            log.info(f"📌 선택된 K-Fold 만 학습 진행합니다: {folds_to_run}")
        else:
            folds_to_run = list(range(getattr(config, "K_FOLDS", 5)))
            log.info(f"📌 전체 K-Fold 학습 진행합니다 (총 {len(folds_to_run)} Folds)")

    # 3. 훈련 파이프라인
    log.info("\n🚀 [Step 2] 훈련 파이프라인 가동...")
    train_pipeline = TrainPipeline(
        use_fixed_split=use_fixed_split,
        folds_to_run=folds_to_run,
        weights_path=args.weights,
        result_dir=result_dir,
    )

    train_pipeline.run()


if __name__ == "__main__":
    main()
