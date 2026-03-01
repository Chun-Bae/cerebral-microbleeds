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
use_k_fold_default = config.USE_K_FOLD
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
    ):
        self.use_k_fold = False if use_fixed_split else use_k_fold_default
        self.pretrained_weights = weights_path

        # TODO: TrainPipeline 진입 전에 학습 결과를 저장할 디렉토리(result_dir)가 미리 세팅(선언 및 생성)되어 전달되어야 합니다.
        self.result_dir = result_dir

        # TODO: 밖에서 Use K-Fold 등의 판단 로직을 거친 후, 파이프라인에는 구체적인 folds_to_run 리스트를 전달받기
        self.folds_to_run = folds_to_run if folds_to_run is not None else [0]

    def run(self):
        """최종 학습 루프 실행"""
        # TODO: TrainPipeline 진입 전 DataPipeLine().run()이 완료되어 LMDB가 미리 세팅되어 있어야 합니다.
        self.prepare_data()

        for fold in self.folds_to_run:
            self.run_fold(fold)

    def prepare_data(self):
        """1단계: 데이터 변환/준비"""
        DataPipeLine().run()

    def load_data(self, fold_idx):
        """2단계: DataLoader 구성"""
        self.train_loader, self.val_loader = build_dataloaders(
            use_k_fold=self.use_k_fold, fold_idx=fold_idx
        )

    def setup_model(self):
        """3단계: 모델, 옵티마이저, 로스 구성"""
        self.model, self.criterion, self.optimizer = compile_model()

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
        )

    def run_fold(self, fold_idx):
        """단일 Fold 사이클 실행"""
        self.load_data(fold_idx)
        self.setup_model()
        self.train_model(fold_idx)
