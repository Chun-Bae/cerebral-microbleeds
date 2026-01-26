"""
This module provides a Napari-based viewer for NIfTI (.nii) files.
It loads SWI and ROI images and allows navigation through cases.
"""

import os
import glob
import numpy as np
import nibabel as nib
import napari

SWI_DIR = r".\data\samsung_data\swi"
ROI_DIR = r".\data\samsung_data\roi"

def load_case(case_id: str):
    swi_path = os.path.join(SWI_DIR, f"{case_id}.nii")
    roi_path = os.path.join(ROI_DIR, f"{case_id}.nii")

    img_nii = nib.load(swi_path)
    roi_nii = nib.load(roi_path)

    img = img_nii.get_fdata()
    roi = roi_nii.get_fdata()

    # (x,y,z) -> (z,y,x)
    img = np.transpose(img, (2, 1, 0))
    roi = np.transpose(roi, (2, 1, 0))

    sx, sy, sz = img_nii.header.get_zooms()[:3]  # (x,y,z)
    scale_zyx = (sz, sy, sx)

    return img, roi, scale_zyx

def get_case_ids():
    paths = sorted(glob.glob(os.path.join(SWI_DIR, "*.nii")))
    case_ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    # ROI 없는 케이스 제거(안전)
    case_ids = [cid for cid in case_ids if os.path.exists(os.path.join(ROI_DIR, f"{cid}.nii"))]
    return case_ids

case_ids = get_case_ids()
if not case_ids:
    raise RuntimeError("SWI/ROI 케이스를 찾지 못했습니다. 폴더 경로를 확인하세요.")

idx = 0

# 최초 로드
img, roi, scale_zyx = load_case(case_ids[idx])

viewer = napari.Viewer(title=f"Samsung SWI/ROI Browser — {case_ids[idx]}")

swi_layer = viewer.add_image(
    img,
    name="SWI",
    rendering="mip",
    colormap="gray",
    contrast_limits=[0, np.percentile(img, 99)]
)
swi_layer.scale = scale_zyx

roi_layer = viewer.add_labels(
    (roi > 0).astype(np.uint8),
    name="ROI",
    opacity=0.7
)
roi_layer.scale = scale_zyx

viewer.dims.ndisplay = 3

def update_view():
    """Update the viewer with the current case data."""
    cid = case_ids[idx]
    img, roi, scale_zyx = load_case(cid)

    # 데이터만 교체 (뷰어 유지)
    swi_layer.data = img
    swi_layer.contrast_limits = (0, np.percentile(img, 99))
    swi_layer.scale = scale_zyx

    roi_layer.data = (roi > 0).astype(np.uint8)
    roi_layer.scale = scale_zyx

    viewer.title = f"Samsung SWI/ROI Browser — {cid}"
    print(f"[{idx+1}/{len(case_ids)}] {cid}")

# 키 바인딩: n=다음, p=이전, ]/[ 도 가능
@viewer.bind_key("n")
@viewer.bind_key("]")
def next_case(v):
    """Go to the next case."""
    global idx
    idx = (idx + 1) % len(case_ids)
    update_view()

@viewer.bind_key("p")
@viewer.bind_key("[")
def prev_case(v):
    """Go to the previous case."""
    global idx
    idx = (idx - 1) % len(case_ids)
    update_view()

# 케이스 목록 출력
print(f"Loaded {len(case_ids)} cases.")
print("Keys: n / ]  -> next,   p / [  -> prev")

if __name__ == "__main__":
    napari.run()
