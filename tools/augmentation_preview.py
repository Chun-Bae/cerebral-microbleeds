#!/usr/bin/env python3
"""
데이터 증강 전후 비교 스크립트.
SWI 슬라이스 + ROI를 동일하게 증강(이동/회전)한 뒤,
증강된 위치에 BBox를 그려 원본과 나란히 비교 PNG 저장.

외부 증강 라이브러리 없이 scipy.ndimage만 사용.

사용법:
  python tools/augmentation_preview.py                         # VK021 slice 30
  python tools/augmentation_preview.py VK021 --slice 30
  python tools/augmentation_preview.py VK049 --slice 45 --rotate 25 --tx 10 --ty -5
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from scipy.ndimage import affine_transform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── 경로 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SWI_PNG_DIR = os.path.join(PROJECT_ROOT, "data", "output_images", "swi")
ROI_PNG_DIR = os.path.join(PROJECT_ROOT, "data", "output_images", "roi")
BBOX_JSON = os.path.join(PROJECT_ROOT, "data", "bboxes", "all_bboxes.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output_images")


def load_png(directory, patient_id, slice_idx):
    """전처리된 PNG 슬라이스 로드 → (H, W) float64"""
    fname = f"{patient_id}_slice_{slice_idx}.png"
    path = os.path.join(directory, fname)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img.astype(np.float64)


def get_bboxes_for_slice(bbox_dict, patient_id, slice_idx):
    """해당 환자/슬라이스의 bbox 리스트 반환. [cx, cy, w, h] (정규화)"""
    key = f"{patient_id}_slice_{slice_idx}.png"
    return bbox_dict.get(key, [])


def apply_augmentation(img, angle_deg, tx_px, ty_px):
    """
    회전 + 이동 증강. img 중심 기준 회전 후 이동.
    scipy affine_transform 사용 (역변환 행렬).
    반환: 증강 이미지, 순방향 변환 행렬 (3x3 homogeneous)
    """
    h, w = img.shape
    cx, cy = w / 2.0, h / 2.0
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    # 순방향 변환: 중심 이동 → 회전 → 원점 복귀 → 이동
    # T_total = Translate(tx,ty) * Translate(cx,cy) * Rotate * Translate(-cx,-cy)
    # 좌표: (x, y) 기준
    fwd = np.array([
        [cos_a, -sin_a, -cx * cos_a + cy * sin_a + cx + tx_px],
        [sin_a,  cos_a, -cx * sin_a - cy * cos_a + cy + ty_px],
        [0,      0,      1]
    ])

    # scipy affine_transform은 역변환 사용, (row, col) = (y, x) 순서
    inv = np.linalg.inv(fwd)
    # scipy 좌표: row=y, col=x → 행렬 재배치
    mat = np.array([
        [inv[1, 1], inv[1, 0]],
        [inv[0, 1], inv[0, 0]]
    ])
    offset = np.array([inv[1, 2], inv[0, 2]])

    aug_img = affine_transform(img, mat, offset=offset, order=1, mode='constant', cval=0)
    return aug_img, fwd


def transform_bboxes(bboxes_norm, fwd_matrix, img_h, img_w):
    """
    정규화 bbox를 순방향 변환으로 이동.
    bbox: [cx, cy, w, h] (정규화) → 변환 후 [cx', cy', w, h] (정규화)
    회전 시 bbox 크기는 유지 (작은 병변이라 근사 충분).
    """
    transformed = []
    for cx_n, cy_n, w_n, h_n in bboxes_norm:
        # 정규화 → 픽셀
        cx_px = cx_n * img_w
        cy_px = cy_n * img_h

        # 순방향 변환 적용
        pt = fwd_matrix @ np.array([cx_px, cy_px, 1.0])
        new_cx_px, new_cy_px = pt[0], pt[1]

        # 다시 정규화
        transformed.append([
            new_cx_px / img_w,
            new_cy_px / img_h,
            w_n,
            h_n
        ])
    return transformed


def draw_bboxes(ax, img, bboxes_norm, color="lime", linewidth=1.5):
    """이미지 위에 bbox 사각형 그리기 (제목/축 없이)."""
    h, w = img.shape[:2]
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    for cx_n, cy_n, w_n, h_n in bboxes_norm:
        bw = w_n * w
        bh = h_n * h
        x = cx_n * w - bw / 2
        y = cy_n * h - bh / 2
        rect = Rectangle((x, y), bw, bh,
                          linewidth=linewidth, edgecolor=color, facecolor="none")
        ax.add_patch(rect)


def save_pure(fig, path):
    """여백 없이 순수 이미지 저장."""
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="증강 전후 BBox 비교")
    parser.add_argument("patient", nargs="?", default="VK021", help="환자 ID")
    parser.add_argument("--slice", type=int, default=30, help="슬라이스 번호")
    parser.add_argument("--rotate", type=float, default=20.0, help="회전 각도 (도)")
    parser.add_argument("--tx", type=float, default=15.0, help="X 이동 (픽셀)")
    parser.add_argument("--ty", type=float, default=-10.0, help="Y 이동 (픽셀)")
    args = parser.parse_args()

    # ── 데이터 로드 (전처리된 PNG) ──
    with open(BBOX_JSON, "r") as f:
        bbox_dict = json.load(f)

    actual_s = args.slice
    swi_slice = load_png(SWI_PNG_DIR, args.patient, actual_s)
    roi_slice = load_png(ROI_PNG_DIR, args.patient, actual_s)
    bboxes = get_bboxes_for_slice(bbox_dict, args.patient, actual_s)

    # bbox 없으면 자동 탐색
    if not bboxes or swi_slice is None:
        print(f"⚠️  {args.patient} slice {actual_s}에 bbox 없음 또는 PNG 없음. bbox 있는 슬라이스 탐색 중...")
        patient_keys = [k for k in bbox_dict if k.startswith(args.patient + "_")]
        for k in sorted(patient_keys):
            s_num = int(k.split("_slice_")[1].replace(".png", ""))
            if bbox_dict[k]:
                swi_tmp = load_png(SWI_PNG_DIR, args.patient, s_num)
                roi_tmp = load_png(ROI_PNG_DIR, args.patient, s_num)
                if swi_tmp is not None and roi_tmp is not None:
                    swi_slice, roi_slice = swi_tmp, roi_tmp
                    bboxes = bbox_dict[k]
                    actual_s = s_num
                    print(f"  → slice {actual_s} 사용 ({len(bboxes)}개 bbox)")
                    break

    if swi_slice is None:
        print(f"❌ {args.patient}의 PNG를 찾을 수 없습니다. ({SWI_PNG_DIR})")
        sys.exit(1)
    h, w = swi_slice.shape[:2]

    if not bboxes:
        print(f"❌ {args.patient}에 bbox가 하나도 없습니다.")
        sys.exit(1)

    print(f"📂 {args.patient} slice {actual_s} | bbox {len(bboxes)}개")
    print(f"🔄 증강: 회전 {args.rotate}°, 이동 ({args.tx}, {args.ty})px")

    # ── 증강 적용 ──
    aug_swi, fwd = apply_augmentation(swi_slice, args.rotate, args.tx, args.ty)
    aug_roi, _ = apply_augmentation(roi_slice, args.rotate, args.tx, args.ty)
    aug_bboxes = transform_bboxes(bboxes, fwd, h, w)

    # ── 개별 저장 ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prefix = f"{args.patient}_s{actual_s}"

    # 1) 원본 + bbox
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    draw_bboxes(ax, swi_slice, bboxes, color="lime")
    p1 = os.path.join(OUTPUT_DIR, f"{prefix}_original_bbox.png")
    save_pure(fig, p1)
    print(f"  📄 {p1}")

    # 2) 증강 + bbox
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    draw_bboxes(ax, aug_swi, aug_bboxes, color="cyan")
    p2 = os.path.join(OUTPUT_DIR, f"{prefix}_augmented_bbox.png")
    save_pure(fig, p2)
    print(f"  📄 {p2}")

    # 3) 원본 ROI overlay
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(swi_slice, cmap="gray")
    ax.imshow(np.ma.masked_where(roi_slice == 0, roi_slice), cmap="Reds", alpha=0.5)
    ax.axis("off")
    p3 = os.path.join(OUTPUT_DIR, f"{prefix}_original_roi.png")
    save_pure(fig, p3)
    print(f"  📄 {p3}")

    # 4) 증강 ROI overlay
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(aug_swi, cmap="gray")
    ax.imshow(np.ma.masked_where(aug_roi < 0.5, aug_roi), cmap="Reds", alpha=0.5)
    ax.axis("off")
    p4 = os.path.join(OUTPUT_DIR, f"{prefix}_augmented_roi.png")
    save_pure(fig, p4)
    print(f"  📄 {p4}")

    # 5) 나란히 비교 (원본 vs 증강, bbox 포함)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    draw_bboxes(axes[0], swi_slice, bboxes, color="lime")
    draw_bboxes(axes[1], aug_swi, aug_bboxes, color="cyan")
    p5 = os.path.join(OUTPUT_DIR, f"{prefix}_comparison.png")
    save_pure(fig, p5)
    print(f"  📄 {p5}")

    print(f"✅ 5장 저장 완료 → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
