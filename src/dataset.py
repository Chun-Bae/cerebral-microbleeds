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
    train_transform = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomAffine(
            degrees=(-30, 30), translate=(0.1, 0.1), scale=(0.9, 2.0), p=0.8
        ),
        K.RandomBrightness(brightness=(0.1, 0.3), p=0.5),
        K.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 3.0), p=0.2),
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
        self.lesion_indices = []
        if self.is_train:
            for name, boxes in self.bboxes_dict.items():
                if len(boxes) > 0:
                    idx = self.name_to_idx.get(name)
                    if idx is not None:
                        self.lesion_indices.append(idx)
            print(
                f"🏥 Lesion Bank 구축 완료: {len(self.lesion_indices)}개 샘플 (Copy-Paste 준비됨)"
            )

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

        # 1. 인접 슬라이스 찾기 (파일명 파싱)
        # VK001_slice_0.png
        try:
            pid, s_rest = curr_name.split("_slice_")
            s_num = int(s_rest.replace(".png", ""))
            prev_name = f"{pid}_slice_{s_num - 1}.png"
            next_name = f"{pid}_slice_{s_num + 1}.png"
        except Exception:
            # 포맷 예외 시 복제
            prev_name = curr_name
            next_name = curr_name

        prev_idx = self.name_to_idx.get(prev_name)
        next_idx = self.name_to_idx.get(next_name)

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

            prev_tensor = load_img_tensor(prev_idx)
            next_tensor = load_img_tensor(next_idx)

            # 경계 처리 (없으면 현재 슬라이스 복제)
            if prev_tensor is None:
                prev_tensor = curr_tensor.clone()
            if next_tensor is None:
                next_tensor = curr_tensor.clone()

            # 마스크 로드 (현재 슬라이스 기준)
            str_idx = f"{idx:05d}"
            roi_bytes = txn.get(f"{str_idx}_mask".encode())
            roi_arr = np.frombuffer(roi_bytes, dtype=np.uint8)
            roi_mask = cv2.imdecode(roi_arr, cv2.IMREAD_UNCHANGED)  # (H, W)

            swi_tensor = torch.cat([prev_tensor, curr_tensor, next_tensor], dim=0)

        # ROI Mask Tensor 변환 (H, W) -> (1, H, W)
        roi_mask_tensor = torch.from_numpy(roi_mask.astype(np.float32)).unsqueeze(0)
        # ROI Mask 값 범위: 0 or 255 -> 0 or 1
        roi_mask_tensor = (roi_mask_tensor > 0).float()

        # bbox 로드
        filename = self.idx_to_name.get(idx, "")
        bboxes_list = self.bboxes_dict.get(filename, [])
        # deep copy to avoid modifying original list during augmentation
        bboxes_list = [list(box) for box in bboxes_list]

        # [Copy-Paste Augmentation]
        # 병변을 오려내어 현재 이미지의 빈 공간(뇌 영역)에 붙여넣기
        if self.is_train and len(self.lesion_indices) > 0 and np.random.rand() < 0.5:
            # 1. Source 선택 (병변이 있는 다른 이미지)
            src_idx = np.random.choice(self.lesion_indices)

            # --- Source Data Loading (Target과 동일한 방식) ---
            # LMDB 트랜잭션이 살아있으므로 그대로 읽음
            # 단, 함수 중복을 피하기 위해 코드가 좀 길어짐 (리팩토링 권장)
            src_name = self.idx_to_name[src_idx]
            try:
                pid, s_rest = src_name.split("_slice_")
                s_num = int(s_rest.replace(".png", ""))
                sp_name = f"{pid}_slice_{s_num - 1}.png"
                sn_name = f"{pid}_slice_{s_num + 1}.png"
            except:
                sp_name = src_name
                sn_name = src_name

            sp_idx = self.name_to_idx.get(sp_name)
            sn_idx = self.name_to_idx.get(sn_name)

            with self.env.begin(write=False) as txn2:  # 중첩 트랜잭션 가능 (readonly)

                def load_src_tensor(target_idx):
                    if target_idx is None:
                        return None
                    str_idx = f"{target_idx:05d}"
                    img_bytes = txn2.get(f"{str_idx}_image".encode())
                    if img_bytes is None:
                        return None
                    arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                    return torch.from_numpy(img.astype(np.float32)).unsqueeze(0)

                s_curr = load_src_tensor(src_idx)
                s_prev = load_src_tensor(sp_idx)
                s_next = load_src_tensor(sn_idx)
                if s_curr is not None:
                    if s_prev is None:
                        s_prev = s_curr.clone()
                    if s_next is None:
                        s_next = s_curr.clone()
                    src_tensor = torch.cat([s_prev, s_curr, s_next], dim=0)  # (3, H, W)
                else:
                    src_tensor = None

            # --- Copy & Paste ---
            src_bboxes = self.bboxes_dict.get(src_name, [])

            if src_tensor is not None and len(src_bboxes) > 0:
                # 2. Source Box 중 하나 랜덤 선택
                s_box = src_bboxes[
                    np.random.randint(len(src_bboxes))
                ]  # [cx, cy, w, h] (0~1)

                H, W = swi_tensor.shape[1:]
                scx, scy, sw, sh = s_box

                # Patch 좌표 계산 (픽셀 단위)
                # 약간 여유있게 자름 (1.2배)
                pw, ph = int(sw * W * 1.2), int(sh * H * 1.2)
                px1 = int((scx * W) - pw // 2)
                py1 = int((scy * H) - ph // 2)

                # 범위 체크
                if (
                    pw > 4
                    and ph > 4
                    and px1 >= 0
                    and py1 >= 0
                    and (px1 + pw) < W
                    and (py1 + ph) < H
                ):
                    patch = src_tensor[
                        :, py1 : py1 + ph, px1 : px1 + pw
                    ].clone()  # (3, h, w)

                    # 3. Target 위치 선정 (뇌 영역 내부)
                    # 현재 이미지(중앙 슬라이스)에서 0이 아닌 픽셀 찾기
                    brain_mask = (swi_tensor[1] > 10).nonzero(
                        as_tuple=False
                    )  # (N, 2) [y, x]

                    if len(brain_mask) > 0:
                        # 랜덤 좌표 선택 (여기가 새로운 중심점)
                        rnd_idx = np.random.randint(len(brain_mask))
                        ty, tx = brain_mask[rnd_idx]

                        # 붙여넣을 좌표 (Top-Left)
                        tx1 = int(tx - pw // 2)
                        ty1 = int(ty - ph // 2)

                        # 범위 체크
                        if tx1 >= 0 and ty1 >= 0 and (tx1 + pw) < W and (ty1 + ph) < H:
                            # 4. 합성 (Alpha Blending with Ellipse Mask)
                            # 타원형 마스크 생성 (중심 1, 가장자리 0)
                            Y, X = np.ogrid[:ph, :pw]
                            center_y, center_x = ph / 2, pw / 2
                            # 타원 방정식: ((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1
                            dist_from_center = ((X - center_x) ** 2) / (
                                (pw / 2) ** 2
                            ) + ((Y - center_y) ** 2) / ((ph / 2) ** 2)

                            # Soft edge (Gaussian Blur 효과 흉내 - 거리 기반 감쇠)
                            # dist가 1에 가까울수록(가장자리) 투명하게
                            alpha_mask = np.clip(1.0 - dist_from_center, 0, 1)
                            alpha_mask = (
                                torch.from_numpy(alpha_mask)
                                .float()
                                .to(swi_tensor.device)
                            )

                            # 합성: Target = Source * alpha + Target * (1 - alpha)
                            existing_area = swi_tensor[
                                :, ty1 : ty1 + ph, tx1 : tx1 + pw
                            ]
                            blended_patch = patch * alpha_mask + existing_area * (
                                1 - alpha_mask
                            )

                            swi_tensor[:, ty1 : ty1 + ph, tx1 : tx1 + pw] = (
                                blended_patch
                            )

                            # 5. Label 추가
                            # 새로운 cx, cy, w, h
                            new_cx = float(tx) / W
                            new_cy = float(ty) / H
                            # 너비/높이는 원본 그대로 유지 (약간의 scaling 효과는 없음)
                            new_w = float(pw) / W / 1.2  # 아까 1.2배 했으니 다시 원복
                            new_h = float(ph) / H / 1.2

                            bboxes_list.append([new_cx, new_cy, new_w, new_h])
                            # print(f"✨ Copy-Paste Applied! New bbox: {new_cx:.2f}, {new_cy:.2f}")

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
    batch_img, batch_lesion_mask, batch_roi_mask, batch_bboxes = next(
        iter(train_loader)
    )
    print(f"\nBefore augmentation:")
    print(
        f"  Image shape: {batch_img.shape}, min: {batch_img.min():.1f}, max: {batch_img.max():.1f}"
    )
    print(f"  Lesion Mask shape: {batch_lesion_mask.shape}")
    print(f"  ROI Mask shape:    {batch_roi_mask.shape}")
    print(f"  BBox counts: {[len(b) for b in batch_bboxes]}")

    # 첫 번째 샘플의 bbox 출력
    if len(batch_bboxes[0]) > 0:
        print(f"  First sample bboxes:\n{batch_bboxes[0]}")

    # GPU로 이동 후 증강 적용
    batch_img = batch_img.to(device)
    batch_lesion_mask = batch_lesion_mask.to(device)
    batch_roi_mask = batch_roi_mask.to(device)

    # Transform (img, mask1, mask2)
    aug_img, aug_lesion_mask, aug_roi_mask = train_transform(
        batch_img, batch_lesion_mask, batch_roi_mask
    )

    # 정규화 적용
    aug_img = normalize_16bit(aug_img)
    print(f"\nAfter augmentation + normalize:")
    print(
        f"  Image shape: {aug_img.shape}, min: {aug_img.min():.2f}, max: {aug_img.max():.2f}"
    )
    print(f"  Lesion Mask shape: {aug_lesion_mask.shape}")
    print(f"  ROI Mask shape:    {aug_roi_mask.shape}")
