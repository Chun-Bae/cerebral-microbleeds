from utils import get_bboxes_from_mask
import os
import cv2
import numpy as np
import albumentations as A
import lmdb
import pickle
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset

# Albumentations 데이터 변환 설정 (8-bit RGB 정규화)


def get_transforms():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(0.2, 0.5),
            contrast_limit=0.0, p=0.3
        ),
        A.GaussianBlur(blur_limit=7, sigma_limit=(0, 3), p=0.2),
        A.Affine(
            translate_percent={"x": (-15/256, 15/256), "y": (-15/256, 15/256)},
            rotate=(-30, 30),
            scale=(0.8, 0.95),
            p=0.5
        ),
        # 8bit RGB 정규화 (0~255 -> -1~1)
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(
            0.5, 0.5, 0.5), max_pixel_value=255.0),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(
            0.5, 0.5, 0.5), max_pixel_value=255.0),
        ToTensorV2()
    ])

    return train_transform, test_transform


# CMB 탐지를 위한 PyTorch Dataset 클래스


class CMBsDataset(Dataset):
    def __init__(self, swi_folder, roi_folder, transform=None):
        self.swi_files = sorted(os.listdir(swi_folder))
        self.roi_files = sorted(os.listdir(roi_folder))
        self.swi_folder = swi_folder
        self.roi_folder = roi_folder
        self.transform = transform

        self.swi_files = [f for f in self.swi_files if f in self.roi_files]
        self.roi_files = [f for f in self.roi_files if f in self.swi_files]

    def __getitem__(self, idx):
        swi_path = os.path.join(self.swi_folder, self.swi_files[idx])
        roi_path = os.path.join(self.roi_folder, self.roi_files[idx])

        # 8-bit RGB 이미지 로드
        swi_img = cv2.imread(swi_path, cv2.IMREAD_GRAYSCALE)
        roi_mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)

        if swi_img is None or roi_mask is None:
            raise ValueError(f"이미지 로드 실패: {swi_path}")

        # 1채널(H, W) -> 3채널(H, W, 3) 복사
        if len(swi_img.shape) == 2:
            swi_img = np.stack([swi_img] * 3, axis=-1)

        # 데이터 변환 적용
        if self.transform:
            transformed = self.transform(image=swi_img, mask=roi_mask)
            swi_img, roi_mask = transformed["image"], transformed["mask"]

        # [최적화] Target Box 미리 생성 (Worker에서 수행하여 Main Loop 부하 제거)
        # roi_mask는 이제 Tensor (H, W) 상태임 (ToTensorV2 이후)
        # 하지만 get_bboxes_from_mask는 Numpy 입력을 선호하므로
        # Transform 이후의 Tensor -> Numpy 변환 (CPU 연산)
        # NOTE: 만약 roi_mask가 Tensor라면 .numpy() 호출. Numpy라면 그대로.
        if hasattr(roi_mask, 'numpy'):
            mask_np = roi_mask.numpy().astype(np.uint8)
        else:
            mask_np = roi_mask.astype(np.uint8)

        H, W = mask_np.shape
        # Mask에서 Box 추출 (Normalized 0~1)
        boxes = get_bboxes_from_mask(mask_np, H, W)

        # boxes를 Tensor로 변환하여 리턴 (나중에 collate에서 리스트로 처리)
        boxes = torch.tensor(boxes, dtype=torch.float32)

        return swi_img, roi_mask, boxes

    def __len__(self):
        return len(self.swi_files)


def detection_collate(batch):
    """
    가변 길이의 BBox를 처리하기 위한 Custom Collate 함수
    Args:
        batch: (image, mask, boxes) 튜플의 리스트
    Returns:
        images: (B, C, H, W) 텐서
        masks: (B, H, W) 텐서
        boxes_list: 각 이미지별 Box 텐서의 리스트 (길이가 다름)
    """
    images = []
    masks = []
    boxes_list = []

    for sample in batch:
        images.append(sample[0])
        masks.append(sample[1])
        boxes_list.append(sample[2])

    return torch.stack(images, 0), torch.stack(masks, 0), boxes_list


class CMBsDatasetLMDB(Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = None  # Worker에서 지연 초기화

        # 길이 정보만 미리 읽기 (메인 프로세스용)
        temp_env = lmdb.open(lmdb_path, readonly=True,
                             lock=False, readahead=False, meminit=False)
        with temp_env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode()).decode())
        temp_env.close()

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

    def __getstate__(self):
        # Pickle 시 env 객체는 제외하고 직렬화
        state = self.__dict__.copy()
        state['env'] = None
        return state

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()

        # Index -> 00001
        str_idx = f"{idx:05d}"

        with self.env.begin(write=False) as txn:
            try:
                swi_bytes = txn.get(f"{str_idx}_image".encode())
                roi_bytes = txn.get(f"{str_idx}_mask".encode())
            except TypeError:
                # 가끔 Transaction 실패 시 재시도 혹은 에러 처리
                raise RuntimeError(f"LMDB Read Failed at index {idx}")

        # 디코딩 (메모리에서 바로 읽음 = 빠름)
        # np.frombuffer는 복사 없이 참조만 하므로 매우 빠름
        swi_arr = np.frombuffer(swi_bytes, dtype=np.uint8)
        roi_arr = np.frombuffer(roi_bytes, dtype=np.uint8)

        # 이미지 디코딩 (8-bit RGB)
        swi_img = cv2.imdecode(swi_arr, cv2.IMREAD_GRAYSCALE)
        roi_mask = cv2.imdecode(roi_arr, cv2.IMREAD_GRAYSCALE)

        # 채널 복사 (1ch -> 3ch)
        if len(swi_img.shape) == 2:
            swi_img = np.stack([swi_img] * 3, axis=-1)

        # Augmentation
        if self.transform:
            transformed = self.transform(image=swi_img, mask=roi_mask)
            swi_img, roi_mask = transformed["image"], transformed["mask"]

        # Target Box 생성 (Dataset에서 처리)
        if hasattr(roi_mask, 'numpy'):
            mask_np = roi_mask.numpy().astype(np.uint8)
        else:
            mask_np = roi_mask.astype(np.uint8)

        H, W = mask_np.shape
        boxes = get_bboxes_from_mask(mask_np, H, W)
        boxes = torch.tensor(boxes, dtype=torch.float32)

        return swi_img, roi_mask, boxes

    def __len__(self):
        return self.length
