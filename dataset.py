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

BBOX_JSON_PATH = "data/bboxes/all_bboxes.json"


def get_transforms(device):
    train_transform = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomAffine(
            degrees=(-30, 30), translate=(0.06, 0.06), scale=(0.8, 0.95), p=0.5
        ),
        K.RandomBrightness(brightness=(0.1, 0.3), p=0.3),
        K.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 3.0), p=0.2),
        data_keys=["image", "mask"],
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
    def __init__(self, lmdb_path, bbox_json_path):
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
        데이터 로더 실질적인 부분
        """
        if self.env is None:
            self._init_db()

        str_idx = f"{idx:05d}"

        with self.env.begin(write=False) as txn:
            swi_bytes = txn.get(f"{str_idx}_image".encode())
            roi_bytes = txn.get(f"{str_idx}_mask".encode())
            if swi_bytes is None or roi_bytes is None:
                raise RuntimeError(f"LMDB 읽기 실패: index {idx}")

        # PNG 바이너리 → numpy array 디코딩
        # swi_arr이 uint8인 건, 압축된 바이너리라서 상관없음
        swi_arr = np.frombuffer(swi_bytes, dtype=np.uint8)
        roi_arr = np.frombuffer(roi_bytes, dtype=np.uint8)

        # 16-bit grayscale로 디코딩
        swi_img = cv2.imdecode(swi_arr, cv2.IMREAD_UNCHANGED)  # (H, W) uint16
        roi_mask = cv2.imdecode(roi_arr, cv2.IMREAD_UNCHANGED)  # (H, W)

        # numpy → Tensor 변환
        # (H, W) → (C, H, W), float32
        swi_tensor = torch.from_numpy(swi_img.astype(np.float32)).unsqueeze(
            0
        )  # (1, H, W)
        swi_tensor = swi_tensor.repeat(3, 1, 1)  # (3, H, W) - 3채널 복사

        # bbox 로드
        filename = self.idx_to_name.get(idx, "")
        bboxes_list = self.bboxes_dict.get(filename, [])

        if len(bboxes_list) > 0:
            bboxes = torch.tensor(bboxes_list, dtype=torch.float32)
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)

        fe_gt_mask = generate_fe_mask(swi_tensor[0:1], bboxes_list, beta=1.5)

        return swi_tensor, fe_gt_mask, bboxes

    def __del__(self):
        if self.env is not None:
            self.env.close()


def generate_fe_mask(image_tensor, bboxes, beta=1.5):
    """
    논문 수식: M = 1 - (P / B_mean)
    image_tensor: (1, H, W) - 정규화 전 16-bit 이미지 (0~65535)
    bboxes: (N, 4) [cx, cy, w, h] (0~1 scale)
    """
    # 1. 함수 내부에서 즉시 정규화 (0~1 scale)
    # 논문 수식의 1.0과 체급을 맞추기 위해 필수입니다.
    P = image_tensor.to(torch.float32) / 65535.0

    _, H, W = P.shape
    fe_mask = torch.zeros((1, H, W), dtype=torch.float32, device=P.device)

    for box in bboxes:
        cx, cy, w, h = box
        # 좌표 변환
        x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # 2. 정규화된 ROI 영역 추출
        roi = P[:, y1:y2, x1:x2]

        # 3. B_mean 계산 (논문: beta * mean(B_area))
        b_mean = beta * torch.mean(roi)

        if b_mean > 0:
            # 4. 논문 수식 적용: M = 1 - (P / B_mean)
            # 병변(어두운 곳)은 P가 작으므로 M이 커짐 -> 특징 강화
            enhanced = 1.0 - (roi / b_mean)

            # 5. beta 임계값 처리 (B_mean보다 밝은 픽셀은 강화 제외)
            enhanced[roi > b_mean] = 0

            # 5-1. 픽셀 값이 너무 낮으면(완전 검은색) 병변이 아니라 배경으로 간주하여 0 처리
            # 이걸 안하면 배경 예측이 많아짐
            # 수정: 이미지가 0~1로 정규화되어 있으므로, 아주 작은 값과 비교해야 함
            min_pixel_val = 1e-6
            enhanced[roi < min_pixel_val] = 0

            # 6. 마스크에 결과 반영
            fe_mask[:, y1:y2, x1:x2] = torch.clamp(enhanced, min=0.0)

    return fe_mask


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)  # (B, C, H, W)
    fe_masks = torch.stack([item[1] for item in batch], dim=0)  # (B, 1, H ,W)
    bboxes = [item[2] for item in batch]  # 리스트 (가변 길이)

    # images:  (4, 3, 512, 512)  # ← Tensor (고정 크기, stack 가능)
    # masks:   (4, 1, 512, 512)  # ← Tensor (고정 크기, stack 가능)
    # bboxes:  [                 # ← 리스트 (가변 길이, stack 불가!)
    # tensor([[0.3, 0.5, 0.1, 0.1],    # 샘플 0: bbox 2개
    #         [0.7, 0.2, 0.05, 0.05]]),
    # tensor([[0.4, 0.6, 0.08, 0.08]]), # 샘플 1: bbox 1개
    # tensor([]),                        # 샘플 2: bbox 0개
    # tensor([[0.2, 0.3, 0.1, 0.1],     # 샘플 3: bbox 3개
    #        [0.5, 0.5, 0.12, 0.12],
    #        [0.8, 0.1, 0.06, 0.06]])
    # ]

    return images, fe_masks, bboxes


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Transform 생성
    train_transform, test_transform = get_transforms(device)
    # Dataset 생성
    train_dataset = CMBsDatasetLMDB(
        lmdb_path="data/lmdb/fold_0/train.lmdb", bbox_json_path=BBOX_JSON_PATH
    )
    print(f"Train 데이터 개수: {len(train_dataset)}")
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )
    # 배치 가져오기
    batch_img, batch_mask, batch_bboxes = next(iter(train_loader))
    print(f"\nBefore augmentation:")
    print(
        f"  Image shape: {batch_img.shape}, min: {batch_img.min():.1f}, max: {batch_img.max():.1f}"
    )
    print(f"  Mask shape: {batch_mask.shape}")
    print(f"  BBox counts: {[len(b) for b in batch_bboxes]}")

    # 첫 번째 샘플의 bbox 출력
    if len(batch_bboxes[0]) > 0:
        print(f"  First sample bboxes:\n{batch_bboxes[0]}")

    # GPU로 이동 후 증강 적용
    batch_img = batch_img.to(device)
    batch_mask = batch_mask.to(device)
    aug_img, aug_mask = train_transform(batch_img, batch_mask)

    # 정규화 적용
    aug_img = normalize_16bit(aug_img)
    print(f"\nAfter augmentation + normalize:")
    print(
        f"  Image shape: {aug_img.shape}, min: {aug_img.min():.2f}, max: {aug_img.max():.2f}"
    )
    print(f"  Mask shape: {aug_mask.shape}")
