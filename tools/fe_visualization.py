#!/usr/bin/env python3
"""
Feature Enhancement (FE) 과정 시각화.
실제 SWI 슬라이스에 FE 공식을 적용하여 단계별 결과를 PNG로 저장.

단계:
  1) 원본 SWI + BBox
  2) Conv 적용 후 축소된 특징맵 (Feature Map)
  3) 병변 BBox 평균 밝기 → 강화 계수 M 계산
  4) BBox 매핑 영역에 M 적용한 강화 특징맵
  5) 원본 vs 강화 비교

사용법:
  python tools/fe_visualization.py                      # VK021
  python tools/fe_visualization.py VK049 --slice 30
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SWI_PNG_DIR = os.path.join(PROJECT_ROOT, "data", "output_images", "swi")
BBOX_JSON = os.path.join(PROJECT_ROOT, "data", "bboxes", "all_bboxes.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output_images")

BETA = 1.5
FEATURE_MAP_SCALE = 8  # 원본 대비 축소 비율 (512 → 64)


def load_png(directory, patient_id, slice_idx):
    fname = f"{patient_id}_slice_{slice_idx}.png"
    path = os.path.join(directory, fname)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img.astype(np.float64)


def find_slice_with_bbox(bbox_dict, patient_id, preferred_slice):
    """bbox 있는 슬라이스 찾기"""
    key = f"{patient_id}_slice_{preferred_slice}.png"
    if key in bbox_dict and bbox_dict[key]:
        return preferred_slice, bbox_dict[key]
    # 탐색
    patient_keys = sorted([k for k in bbox_dict if k.startswith(patient_id + "_")])
    for k in patient_keys:
        if bbox_dict[k]:
            s = int(k.split("_slice_")[1].replace(".png", ""))
            return s, bbox_dict[k]
    return None, []


def simulate_conv(img, kernel_size=5):
    """Conv 레이어를 시뮬레이션 (uniform filter → downsample)"""
    filtered = uniform_filter(img, size=kernel_size)
    h, w = img.shape
    h_f, w_f = h // FEATURE_MAP_SCALE, w // FEATURE_MAP_SCALE
    # 다운샘플 (average pooling 시뮬레이션)
    feature_map = cv2.resize(filtered, (w_f, h_f), interpolation=cv2.INTER_AREA)
    return feature_map


def compute_fe(img_gray_01, feature_map, bboxes, beta=BETA):
    """
    FE 공식 적용 (feature_enhancement.py 기반):
      1. 병변 영역 평균 밝기 region_mean
      2. B_mean = beta * region_mean
      3. M = clamp(1 - pixel/B_mean, 0, 1)    ← 어두울수록 M 큼
      4. enhanced = feature_map * (1 + M)       ← BBox 영역만 적용
    """
    h_i, w_i = img_gray_01.shape
    h_f, w_f = feature_map.shape

    # 원본 해상도에서 병변 평균 밝기
    lesion_pixels = []
    for cx, cy, bw, bh in bboxes:
        x1, y1 = int((cx - bw/2) * w_i), int((cy - bh/2) * h_i)
        x2, y2 = int((cx + bw/2) * w_i), int((cy + bh/2) * h_i)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_i, x2), min(h_i, y2)
        if x2 > x1 and y2 > y1:
            lesion_pixels.append(img_gray_01[y1:y2, x1:x2].flatten())

    if not lesion_pixels:
        return feature_map, np.zeros_like(feature_map)

    region_mean = np.concatenate(lesion_pixels).mean()
    B_mean = max(beta * region_mean, 1e-6)

    # 특징맵 해상도로 리사이즈
    img_resized = cv2.resize(img_gray_01, (w_f, h_f), interpolation=cv2.INTER_NEAREST)

    # 강화 계수 M (전체 계산)
    M_full = np.clip(1.0 - (img_resized / B_mean), 0, 1)

    # BBox 영역만 마스크 (최소 1px 보장)
    enhanced_mask = np.zeros((h_f, w_f))
    for cx, cy, bw, bh in bboxes:
        fx1 = int(np.floor((cx - bw/2) * w_f))
        fy1 = int(np.floor((cy - bh/2) * h_f))
        fx2 = int(np.ceil((cx + bw/2) * w_f))
        fy2 = int(np.ceil((cy + bh/2) * h_f))
        # 최소 1px
        fx2 = max(fx2, fx1 + 1)
        fy2 = max(fy2, fy1 + 1)
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(w_f, fx2), min(h_f, fy2)
        if fx2 > fx1 and fy2 > fy1:
            enhanced_mask[fy1:fy2, fx1:fx2] = M_full[fy1:fy2, fx1:fx2]

    enhanced_map = feature_map * (1.0 + enhanced_mask)
    return enhanced_map, enhanced_mask


def draw_bbox_on_ax(ax, img, bboxes, color="lime", lw=1.5, is_feature_map=False):
    """이미지 위에 bbox 그리기"""
    h, w = img.shape[:2]
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    scale = 1.0
    for cx, cy, bw, bh in bboxes:
        rw = bw * w
        rh = bh * h
        # 특징맵에서는 bbox가 1px일 수 있으니 최소 크기 보장
        if is_feature_map:
            rw = max(rw, 2)
            rh = max(rh, 2)
        x = cx * w - rw / 2
        y = cy * h - rh / 2
        rect = Rectangle((x, y), rw, rh, linewidth=lw, edgecolor=color, facecolor="none")
        ax.add_patch(rect)


def main():
    parser = argparse.ArgumentParser(description="FE 시각화")
    parser.add_argument("patient", nargs="?", default="VK021", help="환자 ID")
    parser.add_argument("--slice", type=int, default=30, help="슬라이스 번호")
    parser.add_argument("--beta", type=float, default=BETA, help="β 계수")
    args = parser.parse_args()

    with open(BBOX_JSON, "r") as f:
        bbox_dict = json.load(f)

    actual_s, bboxes = find_slice_with_bbox(bbox_dict, args.patient, args.slice)
    if actual_s is None:
        print(f"❌ {args.patient}에 bbox가 없습니다.")
        sys.exit(1)

    swi = load_png(SWI_PNG_DIR, args.patient, actual_s)
    if swi is None:
        print(f"❌ PNG 없음: {args.patient} slice {actual_s}")
        sys.exit(1)

    print(f"📂 {args.patient} slice {actual_s} | bbox {len(bboxes)}개 | β={args.beta}")

    # 0~1 정규화 (FE 내부 img_0to1 재현)
    img_01 = swi / swi.max() if swi.max() > 0 else swi
    h, w = swi.shape[:2]

    # Step 1: Conv → Feature Map
    feature_map = simulate_conv(img_01)
    h_f, w_f = feature_map.shape
    print(f"   원본: {h}×{w} → 특징맵: {h_f}×{w_f}")

    # Step 2: FE 적용
    enhanced_map, M_mask = compute_fe(img_01, feature_map, bboxes, args.beta)

    # 병변 평균 밝기 계산 (출력용)
    lesion_px = []
    for cx, cy, bw, bh in bboxes:
        x1, y1 = int((cx - bw/2) * w), int((cy - bh/2) * h)
        x2, y2 = int((cx + bw/2) * w), int((cy + bh/2) * h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            lesion_px.append(img_01[y1:y2, x1:x2].flatten())
    if lesion_px:
        rmean = np.concatenate(lesion_px).mean()
        print(f"   병변 평균 밝기: {rmean:.4f}")
        print(f"   B_mean (β×mean): {args.beta * rmean:.4f}")

    # ── 시각화 (5패널) ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prefix = f"{args.patient}_s{actual_s}_fe"

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.subplots_adjust(wspace=0.05)

    # 1) 원본 + BBox
    draw_bbox_on_ax(axes[0], img_01, bboxes, color="lime", lw=1.5)
    axes[0].set_title("① Original + BBox", fontsize=11, fontweight="bold", pad=6)

    # 2) Conv → Feature Map + BBox 매핑
    draw_bbox_on_ax(axes[1], feature_map, bboxes, color="lime", lw=1.0, is_feature_map=True)
    axes[1].set_title(f"② Feature Map ({h_f}×{w_f})", fontsize=11, fontweight="bold", pad=6)

    # 3) 강화 계수 M (BBox 영역만)
    axes[2].imshow(M_mask, cmap="hot", vmin=0, vmax=1)
    axes[2].axis("off")
    axes[2].set_title("③ Enhancement M", fontsize=11, fontweight="bold", pad=6)

    # 4) 강화된 특징맵
    draw_bbox_on_ax(axes[3], enhanced_map, bboxes, color="cyan", lw=1.0, is_feature_map=True)
    axes[3].set_title("④ X × (1+M)", fontsize=11, fontweight="bold", pad=6)

    # 5) Before/After 차이 (강화 영역 강조)
    diff = enhanced_map - feature_map
    axes[4].imshow(feature_map, cmap="gray")
    # 차이를 빨간색 overlay로
    diff_norm = diff / (diff.max() + 1e-8)
    axes[4].imshow(np.ma.masked_where(diff_norm < 0.01, diff_norm),
                   cmap="Reds", alpha=0.7, vmin=0, vmax=1)
    axes[4].axis("off")
    axes[4].set_title("⑤ Δ Overlay", fontsize=11, fontweight="bold", pad=6)

    out_path = os.path.join(OUTPUT_DIR, f"{prefix}_steps.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"  📄 {out_path}")

    # ── 개별 저장 ──
    # 원본 + BBox
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    draw_bbox_on_ax(ax, img_01, bboxes, color="lime")
    p = os.path.join(OUTPUT_DIR, f"{prefix}_1_original.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"  📄 {p}")

    # Feature Map
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    draw_bbox_on_ax(ax, feature_map, bboxes, color="lime", is_feature_map=True)
    p = os.path.join(OUTPUT_DIR, f"{prefix}_2_featuremap.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"  📄 {p}")

    # M mask
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(M_mask, cmap="hot", vmin=0, vmax=1)
    ax.axis("off")
    p = os.path.join(OUTPUT_DIR, f"{prefix}_3_mask_M.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"  📄 {p}")

    # Enhanced
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    draw_bbox_on_ax(ax, enhanced_map, bboxes, color="cyan", is_feature_map=True)
    p = os.path.join(OUTPUT_DIR, f"{prefix}_4_enhanced.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"  📄 {p}")

    print(f"✅ 저장 완료 → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
