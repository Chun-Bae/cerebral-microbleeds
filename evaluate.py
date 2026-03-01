import argparse
import os
import datetime
import sys

import config
from src.pipelines.evaluate_pipeline import EvaluatePipeline
from src.utils.logger import Logger


def main():
    parser = argparse.ArgumentParser(description="CMB Detection Evaluation")
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to pretrained weights (.pth)"
    )
    parser.add_argument(
        "--lmdb",
        type=str,
        default=os.path.join(
            getattr(config, "LMDB_DIR", "data/lmdb"), "fixed_split", "test.lmdb"
        ),
        help="Path to LMDB test dataset",
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Optional specific patient ID to evaluate (e.g., VK049)",
    )
    args = parser.parse_args()

    # 1. 평가 결과를 저장할 디렉토리 생성
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d(%Hh-%Mm-%Ss)")
    if args.patient:
        result_dir = os.path.join(
            getattr(config, "RESULTS_DIR", "results"),
            f"eval_{args.patient}_{run_timestamp}",
        )
    else:
        result_dir = os.path.join(
            getattr(config, "RESULTS_DIR", "results"), f"eval_{run_timestamp}"
        )

    os.makedirs(result_dir, exist_ok=True)

    # 2. 로거 세팅
    log = Logger(os.path.join(result_dir, "eval_log.txt"))

    log.info(f"========== EVALUATION START ==========")
    log.info(f"Weights: {args.weights}")
    log.info(f"Target LMDB: {args.lmdb}")
    if args.patient:
        log.info(f"Target Patient: {args.patient}")
    log.info(f"Result Directory: {result_dir}")
    log.info(f"======================================\n")

    # 3. 파이프라인 수행
    pipeline = EvaluatePipeline(
        weights_path=args.weights,
        lmdb_path=args.lmdb,
        patient_id=args.patient,
        result_dir=result_dir,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
