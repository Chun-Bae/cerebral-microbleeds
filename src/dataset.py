import os
import cv2
import json
import numpy as np
import lmdb
from tqdm import tqdm
import kornia.augmentation as K
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore", message="Default grid_sample")

import config

BBOX_JSON_PATH = config.BBOX_JSON_PATH


def get_transforms(device):
    train_transform = 
        K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomAffine(
            degrees=(-30, 30),
            translate=(0.03, 0.03),
            scale=(0.8, 0.95),
            p=0.8,
            keepdim=True,
        ),
        K.ColorJitter(brightness=(1.2, 1.5), p=0.5),
        K.RandomBrightness(brightness=(0.1, 0.3), p=0.5),
        K.RandomGaussianBlur(kernel_size=(19, 19), sigma=(0.0, 3.0), p=0.5),
        data_keys=["image", "mask", "mask"],
        same_on_batch=False,
    ).to(device)

    test_transform = None

    return train_transform, test_transform


def normalize_16bit(tensor):
    """
    16-bit (0~65535) → (-1~1) 정규화
    """
    return (tensor / 32767.5) - 1.0


def denormalize(tensor):
    """
    (-1~1) → (0~65535) 역정규화
    """
    return (tensor + 1.0) * 32767.5


def load_bbox_json(json_path=BBOX_JSON_PATH):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    else:
        print(f"⚠️ BBox JSON 파일 없음: {json_path}")
        return {}


class CMBsDatasetLMDB(Dataset):
    def __init__(self, lmdb_path, bbox_json_path, is_train=False):
        self.lmdb_path = lmdb_path
        self.env = None
        self.bboxes_dict = load_bbox_json(bbox_json_path)

        # 인덱스 → 파일명 매핑 (lmdb에 있는 key)
        self.idx_to_name = {}

        tmp_env = lmdb.open(
            lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
        )

        with tmp_env.begin(write=False) as txn:
            self.length = int(txn.get(b"length").decode())

            for i in tqdm(range(self.length), desc="인덱스 매핑"):
                str_idx = f"{i:05d}"
                name_bytes = txn.get(f"{str_idx}_name".encode())
                if name_bytes:
                    self.idx_to_name[i] = name_bytes.decode()

        tmp_env.close()

        # 파일명 → 인덱스 매핑 (2.5D)
        self.name_to_idx = {v: k for k, v in self.idx_to_name.items()}

        # Copy-Paste를 위한 병변 인덱스 뱅크
        self.is_train = is_train

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

            curr_tensor = load_img_tensor(idx)
            if curr_tensor is None:
                raise RuntimeError(f"LMDB 읽기 실패: index {idx}")

            # 3채널 복사 (Grayscale -> RGB imitation)
            swi_tensor = curr_tensor.repeat(3, 1, 1)

            # 마스크 로드 (현재 슬라이스 기준)
            str_idx = f"{idx:05d}"
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


def generate_lesion_mask(image_tensor, bboxes):
    """
    Segmentation Ground Truth 생성 (Binary Mask)
    image_tensor: (1, H, W) - Shape 참고용
    bboxes: (N, 4) [cx, cy, w, h] (0~1 scale)
    """
    _, H, W = image_tensor.shape
    mask = torch.zeros((1, H, W), dtype=torch.float32, device=image_tensor.device)

    for box in bboxes:
        cx, cy, w, h = box
        # 좌표 변환
        x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # 병변 영역 1로 채우기
        mask[:, y1:y2, x1:x2] = 1.0

    return mask


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)  # (B, C, H, W)
    lesion_masks = torch.stack([item[1] for item in batch], dim=0)  # (B, 1, H, W)
    roi_masks = torch.stack([item[2] for item in batch], dim=0)  # (B, 1, H, W)
    bboxes = [item[3] for item in batch]  # 리스트

    return images, lesion_masks, roi_masks, bboxes
