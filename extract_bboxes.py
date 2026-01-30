import os
import cv2
import numpy as np
import json
from tqdm import tqdm

# lmdb 안씀
SWI_DIR = "data/output_images/swi"
ROI_DIR = "data/output_images/roi"
OUTPUT_DIR_SWI = "data/bbox_visualization/swi"
OUTPUT_DIR_ROI = "data/bbox_visualization/roi"
BBOX_JSON_DIR = "data/bboxes"


def mask_to_bboxes(mask, img_size=512):
    binary = (mask > 0).astype(np.uint8) * 255

    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []

    for cnt in contours:
        # bbox 계산
        x, y, w, h = cv2.boundingRect(cnt)

        cx = (x + w / 2) / img_size
        cy = (y + h / 2) / img_size

        w_norm = w / img_size
        h_norm = h / img_size

        bboxes.append([cx, cy, w_norm, h_norm])

    return bboxes, contours


def visualize_bbox(
    swi_img, roi_mask, bboxes, contours, img_size=512, draw_contour=True, draw_bbox=True
):
    # 시각화용 16-bit -> 8-bit 변환
    if swi_img.dtype == np.uint16:
        swi_8bit = (swi_img / 256).astype(np.uint8)
    else:
        swi_8bit = swi_img.astype(np.uint8)

    # 1채널 → 3채널 (빨간선 그려주기 위함)
    # len == 2는 채널이 2개인지 확인하는 뜻
    if len(swi_8bit.shape) == 2:
        vis_img = cv2.cvtColor(swi_8bit, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = swi_8bit.copy()

    # 1. 윤곽선 그리기 (초록색)
    if draw_contour:
        cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 1)

    # 2. bbox 그리기 (빨간색)
    if draw_bbox:
        for bbox in bboxes:
            cx, cy, w, h = bbox
            # 정규화 → 픽셀 좌표
            cx_px = int(cx * img_size)
            cy_px = int(cy * img_size)
            w_px = int(w * img_size)
            h_px = int(h * img_size)

            x1 = cx_px - (w_px // 2)
            y1 = cy_px - (h_px // 2)
            x2 = cx_px + (w_px // 2)
            y2 = cy_px + (h_px // 2)

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # 범례
    if draw_contour:
        cv2.putText(
            vis_img,
            "Green: Contour",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    if draw_bbox:
        cv2.putText(
            vis_img,
            "Red: BBox",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    cv2.putText(
        vis_img,
        f"Count: {len(bboxes)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return vis_img


def extract_and_visualize(
    swi_dir=SWI_DIR,
    roi_dir=ROI_DIR,
    output_dir_swi=OUTPUT_DIR_SWI,
    output_dir_roi=OUTPUT_DIR_ROI,
    bbox_dir=BBOX_JSON_DIR,
):
    os.makedirs(output_dir_swi, exist_ok=True)
    os.makedirs(output_dir_roi, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)

    # 파일 목록
    roi_files = sorted([f for f in os.listdir(roi_dir) if f.endswith(".png")])

    # 병변 있는 파일 필터링
    files_with_lesions = []
    for f in tqdm(roi_files, desc="병변 개수 필터링"):
        roi_path = os.path.join(roi_dir, f)
        roi = cv2.imread(roi_path, cv2.IMREAD_UNCHANGED)
        # 병변 없으면: 전부 0 → max() = 0
        # 병변 있으면: max() > 0
        if roi is not None and roi.max() > 0:
            files_with_lesions.append(f)

    print(f"전체 파일: {len(roi_files)}")
    print(f"병변 있는 파일: {len(files_with_lesions)}")

    # 나중에 몇 개만 추출하고 싶으면 이 부분에서 분기문 나눠주면 됌
    sample_files = files_with_lesions

    # 모든 bbox 정보 저장용
    all_bboxes = {}

    for filename in tqdm(sample_files, desc="bbox 추출 및 시각화"):
        swi_path = os.path.join(swi_dir, filename)
        roi_path = os.path.join(roi_dir, filename)

        # 이미지 로드
        swi_img = cv2.imread(swi_path, cv2.IMREAD_UNCHANGED)
        roi_mask = cv2.imread(roi_path, cv2.IMREAD_UNCHANGED)

        # 혹시 모를 예외
        if swi_img is None or roi_mask is None:
            continue

        # bbox 추출
        bboxes, contours = mask_to_bboxes(roi_mask, img_size=swi_img.shape[0])

        # dict 저장 (후에 한번에 json 저장)
        all_bboxes[filename] = bboxes

        # === SWI 시각화 ===
        save_one_visualization(
            swi_img,
            roi_mask,
            bboxes,
            contours,
            output_dir_swi,
            filename,
            draw_contour=True,
            draw_bbox=False,
            subfolder="only_contour",
        )
        save_one_visualization(
            swi_img,
            roi_mask,
            bboxes,
            contours,
            output_dir_swi,
            filename,
            draw_contour=False,
            draw_bbox=True,
            subfolder="only_bbox",
        )
        save_one_visualization(
            swi_img,
            roi_mask,
            bboxes,
            contours,
            output_dir_swi,
            filename,
            draw_contour=True,
            draw_bbox=True,
            subfolder="both",
        )

        # === ROI 시각화 ===
        save_one_visualization(
            roi_mask,
            roi_mask,
            bboxes,
            contours,
            output_dir_roi,
            filename,
            draw_contour=True,
            draw_bbox=False,
            subfolder="only_contour",
        )
        save_one_visualization(
            roi_mask,
            roi_mask,
            bboxes,
            contours,
            output_dir_roi,
            filename,
            draw_contour=False,
            draw_bbox=True,
            subfolder="only_bbox",
        )
        save_one_visualization(
            roi_mask,
            roi_mask,
            bboxes,
            contours,
            output_dir_roi,
            filename,
            draw_contour=True,
            draw_bbox=True,
            subfolder="both",
        )

    # 전체 bbox 정보 json 저장
    json_path = os.path.join(bbox_dir, "all_bboxes.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_bboxes, f, indent=2)

    print(f"\n✅ SWI 시각화 저장: {output_dir_swi}")
    print(f"✅ ROI 시각화 저장: {output_dir_roi}")
    print(f"✅ BBox JSON 저장: {json_path}")
    print(f"✅ 총 {len(all_bboxes)} 파일 처리 완료")

    return all_bboxes


def save_one_visualization(
    img,
    roi_mask,
    bboxes,
    contours,
    output_dir,
    filename,
    draw_contour,
    draw_bbox,
    subfolder,
):
    save_dir = os.path.join(output_dir, subfolder)
    os.makedirs(save_dir, exist_ok=True)

    vis_img = visualize_bbox(
        img,
        roi_mask,
        bboxes,
        contours,
        img.shape[0],
        draw_contour=draw_contour,
        draw_bbox=draw_bbox,
    )
    cv2.imwrite(os.path.join(save_dir, filename), vis_img)


def main():
    # 이미 bbox JSON이 있으면 생략
    bbox_json_path = os.path.join(BBOX_JSON_DIR, "all_bboxes.json")
    if os.path.exists(bbox_json_path):
        print(f"⏭️ 이미 BBox JSON 존재, 생략: {bbox_json_path}")
        return

    extract_and_visualize()


if __name__ == "__main__":
    main()
