import os
import config

from src.pipelines.data_pipeline import DataPipeLine
from src.core.training.engine.loop import train_loop
from src.core.training import (
    build_dataloaders,
    compile_model,
)


# === Global ===
k_folds = config.K_FOLDS
results_dir = config.RESULTS_DIR
device = config.DEVICE


class TrainPipeline:
    """
    CMB Detection 모델 학습을 위한 전체 파이프라인 관리 클래스
    (데이터 전처리 -> 데이터 로더 구성 -> 모델/손실 셋업 -> 학습 루프 실행)
    """

    def __init__(
        self,
        use_fixed_split=False,
        folds_to_run=None,
        weights_path=None,
        result_dir=None,
        model_name="SSD_FE",
        run_name="default",
    ):
        self.use_k_fold = not use_fixed_split
        self.pretrained_weights = weights_path
        self.model_name = model_name
        self.run_name = run_name
        self.result_dir = result_dir
        self.folds_to_run = folds_to_run if folds_to_run is not None else [0]

    def run(self):
        """최종 학습 루프 실행"""
        for fold in self.folds_to_run:
            self.run_fold(fold)

    def load_data(self, fold_idx):
        """2단계: DataLoader 구성"""
        self.train_loader, self.val_loader = build_dataloaders(
            use_k_fold=self.use_k_fold, fold_idx=fold_idx
        )

    def setup_model(self):
        """3단계: 모델, 옵티마이저, 로스 구성"""
        self.model, self.criterion, self.optimizer = compile_model(self.model_name)

    def train_model(self, fold_idx):
        """4단계: 실제 학습 코어함수 호출"""

        train_loop(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=device,
            fold_idx=fold_idx,
            pretrained_weights=self.pretrained_weights,
            use_k_fold=self.use_k_fold,
            model_name=self.model_name,
            run_name=self.run_name,
        )

    def run_fold(self, fold_idx):
        """단일 Fold 사이클 실행"""
        self.load_data(fold_idx)
        self.setup_model()
        self.train_model(fold_idx)
