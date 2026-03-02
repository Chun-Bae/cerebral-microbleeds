import argparse
import os
import datetime
import sys

import config
from src.pipelines.data_pipeline import DataPipeLine
from src.pipelines.train_pipeline import TrainPipeline
from src.utils.logger import Logger


def main():
    # 0.5 모델 선택 (동적 로딩)
    import src.models as models_module

    available_models = models_module.__all__

    parser = argparse.ArgumentParser(description="CMB Detection Training")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=available_models,
        help=f"Model architecture to use for training. Options: {', '.join(available_models)}",
    )
    parser.add_argument(
        "--prepare_data",
        action="store_true",
        help="Run data preprocessing and LMDB creation before training",
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
    parser.add_argument(
        "--run_name",
        type=str,
        default="default_run",
        help="Name of this training run (used for saving weights and logs)",
    )
    args = parser.parse_args()

    # 0. 가중치 이름
    run_name = args.run_name.replace(" ", "_")

    # 0.5 모델 선택 (동적 로딩)
    model_name = args.model

    # 1. 로거 및 결과 폴더 준비
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d(%Hh-%Mm-%Ss)")
    result_dir = os.path.join(config.RESULTS_DIR, f"train_{run_name}_{run_timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    log = Logger(os.path.join(result_dir, "train_log.txt"))

    log.info(f"========== TRAINING START ==========")
    log.info(f"Result Directory: {result_dir}")
    log.info(f"====================================\n")

    # 1. 데이터 파이프라인
    if args.prepare_data:
        log.info("🚀 [Step 1] 데이터 파이프라인 가동 (전처리 및 DB 생성)...")
        data_pipeline = DataPipeLine()
        data_pipeline.run()
    else:
        log.info("⏩ [Step 1] 데이터 전처리 스킵 (이미 구축된 LMDB 사용 가정)")

    # 2. 훈련 대상 Fold 파악
    use_fixed_split = args.fixed_split
    if use_fixed_split:
        folds_to_run = [0]
        log.info("📌 Fixed Split 모드로 학습 진행합니다")
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
        result_dir=result_dir,
        model_name=model_name,
        run_name=run_name,
    )

    train_pipeline.run()


if __name__ == "__main__":
    main()
