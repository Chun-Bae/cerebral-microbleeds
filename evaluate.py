import argparse
import os
import datetime

import config
from src.pipelines.evaluate_pipeline import EvaluatePipeline
from src.utils.logger import Logger


def main():
    import src.models as models_module

    available_models = models_module.__all__

    parser = argparse.ArgumentParser(description="CMB Detection Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=available_models,
        help=f"Model architecture to evaluate. Options: {', '.join(available_models)}",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="default_run",
        help="Name of the training run to evaluate",
    )
    parser.add_argument(
        "--fixed_split",
        action="store_true",
        help="Force use fixed split weights instead of K-Fold",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Specific fold indices to evaluate (e.g., --folds 0 1 2). If not provided, runs all folds.",
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
    parser.add_argument(
        "--conf",
        type=float,
        default=getattr(config, "CONF_THRESH"),
        help="Confidence threshold for filtering detections (default: 0.001)",
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=getattr(config, "NMS_THRESH", 0.45),
        help="NMS IoU threshold for suppressing overlapping detections (default: 0.45)",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="Disable visualization to speed up evaluation",
    )
    args = parser.parse_args()
    config.CONF_THRESH = args.conf
    config.NMS_THRESH = args.nms_thresh

    # 0.5 가중치 파일명 및 Fold 대상 파악
    run_name = args.run_name.replace(" ", "_")
    use_fixed_split = args.fixed_split

    if use_fixed_split:
        folds_to_run = [0]
    else:
        if args.folds is not None:
            folds_to_run = args.folds
        else:
            folds_to_run = list(range(getattr(config, "K_FOLDS", 10)))

    # 1. 평가 결과를 저장할 최상위 디렉토리 생성
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d(%Hh-%Mm-%Ss)")
    if args.patient:
        result_dir = os.path.join(
            getattr(config, "RESULTS_DIR", "results"),
            f"eval_{args.model}_{run_name}_{args.patient}_{run_timestamp}",
        )
    else:
        result_dir = os.path.join(
            getattr(config, "RESULTS_DIR", "results"),
            f"eval_{args.model}_{run_name}_{run_timestamp}",
        )

    os.makedirs(result_dir, exist_ok=True)
    log = Logger(os.path.join(result_dir, "eval_log.txt"))

    all_metrics = []

    for fold in folds_to_run:
        suffix = "fixed_split" if use_fixed_split else f"fold_{fold}"
        weights_path = os.path.join(
            getattr(config, "WEIGHTS_DIR", "weights"),
            f"{args.model}_{run_name}_{suffix}.pth",
        )

        if not os.path.exists(weights_path):
            log.warning(f"❌ 가중치 파일을 찾을 수 없습니다 스킵합니다: {weights_path}")
            continue

        # LMDB path 결정: 명시적 입력이 없다면 해당 split 폴더의 test.lmdb 자동 선택
        if args.lmdb and parser.get_default("lmdb") != args.lmdb:
            lmdb_path = args.lmdb
        else:
            lmdb_path = os.path.join(
                getattr(config, "LMDB_DIR", "data/lmdb"), suffix, "test.lmdb"
            )

        # 개별 Fold 결과 저장 디렉토리
        fold_result_dir = os.path.join(result_dir, suffix)
        os.makedirs(fold_result_dir, exist_ok=True)

        log.info(f"\n========== EVALUATION START ({suffix}) ==========")
        log.info(f"Model: {args.model}")
        log.info(f"Weights: {weights_path}")
        log.info(f"Target LMDB: {lmdb_path}")
        if args.patient:
            log.info(f"Target Patient: {args.patient}")
        log.info(f"Result Directory: {fold_result_dir}")
        log.info(f"========================================================\n")

        # 3. 파이프라인 수행
        pipeline = EvaluatePipeline(
            model_name=args.model,
            weights_path=weights_path,
            lmdb_path=lmdb_path,
            patient_id=args.patient,
            result_dir=fold_result_dir,
            visualize=not args.no_vis,
        )

        metrics_dict = pipeline.run()
        if metrics_dict:
            all_metrics.append((suffix, metrics_dict))

    # K-Fold 결과 평균 요약 출력
    if not use_fixed_split and len(all_metrics) > 1:
        log.info("\n" + "=" * 50)
        log.info(f"🌈 K-Fold (총 {len(all_metrics)} Fold) 평균 평가 결과 요약")
        log.info("=" * 50)

        avg_metrics = {}
        for _, m in all_metrics:
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    avg_metrics[k] = avg_metrics.get(k, 0) + v

        for k, v in avg_metrics.items():
            avg_metrics[k] = v / len(all_metrics)

        for k, v in avg_metrics.items():
            if isinstance(v, float):
                log.info(f"  Avg {k:<20}: {v:.4f}")
            else:
                log.info(f"  Avg {k:<20}: {v}")

        # Summary 저장
        with open(os.path.join(result_dir, "kfold_avg_metrics.txt"), "w") as f:
            for k, v in avg_metrics.items():
                f.write(f"Avg {k}: {v}\n")


if __name__ == "__main__":
    main()
