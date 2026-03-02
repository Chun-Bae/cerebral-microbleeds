import json
import numpy as np

# BBox JSON 로드
with open("data/bboxes/all_bboxes.json", "r") as f:
    all_bboxes = json.load(f)

# 모든 bbox 수집
widths = []
heights = []
ratios = []
scales = []  # sqrt(w * h)

for filename, bboxes in all_bboxes.items():
    for bbox in bboxes:
        cx, cy, w, h = bbox
        widths.append(w)
        heights.append(h)
        if h > 0:
            ratios.append(w / h)
            scales.append(np.sqrt(w * h))  # 면적 기반 scale

widths = np.array(widths)
heights = np.array(heights)
ratios = np.array(ratios)
scales = np.array(scales)

# 통계 출력
print(f"=== BBox 분석 결과 ===")
print(f"총 bbox 개수: {len(widths)}")
print()
print(
    f"Width (정규화):  min={widths.min():.4f}, max={widths.max():.4f}, mean={widths.mean():.4f}, std={widths.std():.4f}"
)
print(
    f"Height (정규화): min={heights.min():.4f}, max={heights.max():.4f}, mean={heights.mean():.4f}, std={heights.std():.4f}"
)
print(
    f"Scale (sqrt(w*h)): min={scales.min():.4f}, max={scales.max():.4f}, mean={scales.mean():.4f}, std={scales.std():.4f}"
)
print(
    f"Ratio (w/h):     min={ratios.min():.4f}, max={ratios.max():.4f}, mean={ratios.mean():.4f}, std={ratios.std():.4f}"
)
print()

# 분위수
print(
    f"Width 분위수:  25%={np.percentile(widths, 25):.4f}, 50%={np.percentile(widths, 50):.4f}, 75%={np.percentile(widths, 75):.4f}, 95%={np.percentile(widths, 95):.4f}"
)
print(
    f"Height 분위수: 25%={np.percentile(heights, 25):.4f}, 50%={np.percentile(heights, 50):.4f}, 75%={np.percentile(heights, 75):.4f}, 95%={np.percentile(heights, 95):.4f}"
)
print(
    f"Scale 분위수:  25%={np.percentile(scales, 25):.4f}, 50%={np.percentile(scales, 50):.4f}, 75%={np.percentile(scales, 75):.4f}, 95%={np.percentile(scales, 95):.4f}"
)
print(
    f"Ratio 분위수:  25%={np.percentile(ratios, 25):.4f}, 50%={np.percentile(ratios, 50):.4f}, 75%={np.percentile(ratios, 75):.4f}, 95%={np.percentile(ratios, 95):.4f}"
)
print()

# Scale 추천 (면적 기반: sqrt(w * h))
print(f"=== Scale 추천 (sqrt(w*h)) ===")
print(f"5% scale:  {np.percentile(scales, 5):.4f}")
print(f"25% scale: {np.percentile(scales, 25):.4f}")
print(f"50% scale: {np.percentile(scales, 50):.4f}")
print(f"75% scale: {np.percentile(scales, 75):.4f}")
print(f"95% scale: {np.percentile(scales, 95):.4f}")

# Ratio 추천
print(f"=== Ratio 추천 ===")
print(f"5% ratio:  {np.percentile(ratios, 5):.4f}")
print(f"25% ratio: {np.percentile(ratios, 25):.4f}")
print(f"50% ratio: {np.percentile(ratios, 50):.4f}")
print(f"75% ratio: {np.percentile(ratios, 75):.4f}")
print(f"95% ratio: {np.percentile(ratios, 95):.4f}")
