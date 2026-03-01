import os
import torch
from torch.utils.data import DataLoader
from src.models import SSD_FE
from src.core.evaluation import evaluate
from src.datasets import (
    CMBsDataset,
    collate_fn,
    filter_dataset_by_patient,
)
import config

# === Global ===
device = config.DEVICE
results_dir = config.RESULTS_DIR
batch_size = config.BATCH_SIZE
num_workers = config.NUM_WORKERS
bbox_json_path = config.BBOX_JSON_PATH


class EvaluatePipeline:
    """
    Evaluation 파이프라인(TP, FP, FN, mAP, FROC, 시각화) 실행
    """

    def __init__(
        self, weights_path=None, lmdb_path=None, patient_id=None, result_dir=None
    ):
        self.weights_path = weights_path
        self.lmdb_path = lmdb_path
        self.patient_id = patient_id

        # TODO: 평가 파이프라인 진입 전에 평가 결과를 저장할 디렉토리(result_dir)가 미리 세팅(선언 및 생성)되어 전달되어야 합니다.
        self.result_dir = result_dir

    def run(self):
        """평가 파이프라인 실행"""
        self.setup_model()
        self.load_data()

        evaluate(
            model=self.model,
            testloader=self.test_loader,
            dataset=self.dataset,
            device=device,
            save_dir=self.result_dir,
        )

    def setup_model(self):
        """1단계: 모델 생성 및 가중치 로드"""
        self.model = SSD_FE(num_classes=2).to(device)
        checkpoint = torch.load(
            self.weights_path, map_location=device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def load_data(self):
        """2단계: DataLoader 구성"""
        self.dataset = CMBsDataset(self.lmdb_path, bbox_json_path)

        if self.patient_id:
            self.dataset = filter_dataset_by_patient(self.dataset, self.patient_id)

        self.test_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
