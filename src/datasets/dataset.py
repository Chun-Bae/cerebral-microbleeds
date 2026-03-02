import os
import cv2
import json
import numpy as np
import lmdb
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
import config
from .utils import load_bbox_json, generate_lesion_mask


warnings.filterwarnings("ignore", message="Default grid_sample")

bbox_json_path = config.BBOX_JSON_PATH


class CMBsDataset(Dataset):
    def __init__(self, lmdb_path, bbox_json_path, split_type="train"):
        self.lmdb_path = lmdb_path
        self.env = None
        self.bboxes_dict = load_bbox_json(bbox_json_path)

        # 인덱스 → 파일명 매핑 (lmdb에 있는 key)
        self.idx_to_name = {}
        self.valid_indices = []
        self.split_type = split_type

        tmp_env = lmdb.open(
            lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
        )

        with tmp_env.begin(write=False) as txn:
            raw_length = int(txn.get(b"length").decode())

            raw_idx_to_name = {}
            pos_indices = []
            neg_indices = []

            for i in tqdm(range(raw_length), desc="인덱스 매핑"):
                str_idx = f"{i:05d}"
                name_bytes = txn.get(f"{str_idx}_name".encode())
                if name_bytes:
                    filename = name_bytes.decode()
                    raw_idx_to_name[i] = filename

                    bboxes = self.bboxes_dict.get(filename, [])
                    if len(bboxes) > 0:
                        pos_indices.append(i)
                    else:
                        neg_indices.append(i)

            if self.split_type == "test":
                import random

                # 고정 시드로 매번 동일한 holdout 세트를 구성 (재현성)
                random.seed(config.SEED)
                if len(neg_indices) > len(pos_indices):
                    sampled_neg = random.sample(neg_indices, len(pos_indices))
                else:
                    sampled_neg = neg_indices

                combined_indices = pos_indices + sampled_neg
                random.shuffle(combined_indices)
            else:
                # train, valid는 병변 있는(positive) 것만 사용
                combined_indices = pos_indices

            self.valid_indices = combined_indices
            for new_idx, real_idx in enumerate(self.valid_indices):
                self.idx_to_name[new_idx] = raw_idx_to_name[real_idx]

            self.length = len(self.valid_indices)

        tmp_env.close()

        # 파일명 → 인덱스 매핑 (2.5D)
        self.name_to_idx = {v: k for k, v in self.idx_to_name.items()}

    def get_stats(self):
        """Dataset 통계 반환: 환자 수, 슬라이스 수, GT 병변 수"""
        patient_ids = set()
        total_lesions = 0
        for name in self.idx_to_name.values():
            patient_id = (
                name.split("_slice_")[0] if "_slice_" in name else name.split("_")[0]
            )
            patient_ids.add(patient_id)
            total_lesions += len(self.bboxes_dict.get(name, []))
        return {
            "환자 수": len(patient_ids),
            "슬라이스 수 (PNG)": len(self.valid_indices),
            "GT 병변 수": total_lesions,
        }

    def _init_db(self):
        """
        lmdb 연결 초기화
        """
        self.env = lmdb.open(
            self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
        )

    def __getstate__(self):
        # 현재 변수 상태
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        데이터 로더 - 2.5D Input (Prev, Curr, Next)
        """
        if self.env is None:
            self._init_db()

        real_idx = self.valid_indices[idx]
        curr_name = self.idx_to_name[idx]

        # 1. 2.5D Logic Removed -> Single Slice duplicated to 3 channels
        with self.env.begin(write=False) as txn:
            # Helper function
            def load_img_tensor(target_idx):
                if target_idx is None:
                    return None
                str_idx = f"{target_idx:05d}"
                img_bytes = txn.get(f"{str_idx}_image".encode())
                if img_bytes is None:
                    return None
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)  # (H, W)
                return torch.from_numpy(img.astype(np.float32)).unsqueeze(
                    0
                )  # (1, H, W)

            curr_tensor = load_img_tensor(real_idx)
            if curr_tensor is None:
                raise RuntimeError(f"LMDB 읽기 실패: index {real_idx}")

            # 3채널 복사 (Grayscale -> RGB imitation)
            swi_tensor = curr_tensor.repeat(3, 1, 1)

            # 마스크 로드 (현재 슬라이스 기준)
            str_idx = f"{real_idx:05d}"
            roi_bytes = txn.get(f"{str_idx}_mask".encode())
            roi_arr = np.frombuffer(roi_bytes, dtype=np.uint8)
            roi_mask = cv2.imdecode(roi_arr, cv2.IMREAD_UNCHANGED)  # (H, W)

        # ROI Mask Tensor 변환 (H, W) -> (1, H, W)
        roi_mask_tensor = torch.from_numpy(roi_mask.astype(np.float32)).unsqueeze(0)
        # ROI Mask 값 범위: 0 or 255 -> 0 or 1
        roi_mask_tensor = (roi_mask_tensor > 0).float()

        # bbox 로드
        filename = self.idx_to_name.get(idx, "")
        bboxes_list = self.bboxes_dict.get(filename, [])
        # deep copy to avoid modifying original list during augmentation
        bboxes_list = [list(box) for box in bboxes_list]

        if len(bboxes_list) > 0:
            bboxes = torch.tensor(bboxes_list, dtype=torch.float32)
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)

        # generate_lesion_mask (Binary, 0 or 1)
        # Segmentation Target으로 사용
        lesion_mask = generate_lesion_mask(swi_tensor[1:2], bboxes_list)

        return swi_tensor, lesion_mask, roi_mask_tensor, bboxes

    def __del__(self):
        if self.env is not None:
            self.env.close()
