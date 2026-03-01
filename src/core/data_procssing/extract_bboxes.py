import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import config
from src.utils.logger import log


swi_dir = config.SWI_OUTPUT_DIR
roi_dir = config.ROI_OUTPUT_DIR
bbox_json_path = config.BBOX_JSON_PATH

output_dir_swi = os.path.join(config.DATA_ROOT, "bbox_visualization", "swi")
output_dir_roi = os.path.join(config.DATA_ROOT, "bbox_visualization", "roi")


def _mask_to_bboxes(mask, img_size=512):
    """주어진 마스크 영역(Binary)으로부터 중심점과 정규화된 W,H (YOLO Format) 박스를 추출합니다."""
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = (x + w / 2) / img_size
        cy = (y + h / 2) / img_size
        w_norm = w / img_size
        h_norm = h / img_size
        bboxes.append([cx, cy, w_norm, h_norm])

    return bboxes, contours


def _visualize_bbox(
    img, contours, bboxes, img_size=512, draw_contour=True, draw_bbox=True
):
    """단일 이미지 위에 Bounding Box와 등고선(Contour)을 그려 반환합니다."""
    # 시각화용 16-bit -> 8-bit 변환
    if img.dtype == np.uint16:
        img_8bit = (img / 256).astype(np.uint8)
    else:
        img_8bit = img.astype(np.uint8)

    # 1채널 → 3채널 컬러로 변환
    if len(img_8bit.shape) == 2:
        vis_img = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = img_8bit.copy()

    # 1. 윤곽선 (Green)
    if draw_contour:
        cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 1)

    # 2. 바운딩 박스 (Red)
    if draw_bbox:
        for bbox in bboxes:
            cx, cy, w, h = bbox
            cx_px, cy_px = int(cx * img_size), int(cy * img_size)
            w_px, h_px = int(w * img_size), int(h * img_size)

            x1, y1 = cx_px - (w_px // 2), cy_px - (h_px // 2)
            x2, y2 = cx_px + (w_px // 2), cy_px + (h_px // 2)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # 3. 범례
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


def _save_one_visualization(
    img, contours, bboxes, output_dir, filename, draw_contour, draw_bbox, subfolder
):
    save_dir = os.path.join(output_dir, subfolder)
    os.makedirs(save_dir, exist_ok=True)

    vis_img = _visualize_bbox(
        img,
        contours,
        bboxes,
        img.shape[0],
        draw_contour=draw_contour,
        draw_bbox=draw_bbox,
    )
    cv2.imwrite(os.path.join(save_dir, filename), vis_img)


def run_extract_bboxes():
    """
    모든 데이터의 정답 마스크(ROI)를 스캔하여 바운딩 박스(Yolo-format)를 추출한 후
    중앙 BBox JSON 파일로 저장하고, 시각적 검증을 위해 이미지들을 생성합니다.
    """
    if os.path.exists(bbox_json_path):
        log.info(f"이미 BBox JSON 정보가 존재합니다, 생략: {bbox_json_path}")
        return

    log.info("BBox 추출 및 시각화 데이터 생성 시작...")
    os.makedirs(output_dir_swi, exist_ok=True)
    os.makedirs(output_dir_roi, exist_ok=True)
    os.makedirs(os.path.dirname(bbox_json_path), exist_ok=True)

    roi_files = sorted([f for f in os.listdir(roi_dir) if f.endswith(".png")])
    if not roi_files:
        log.warning(f"마스크 PNG 파일이 존재하지 않습니다: {roi_dir}")
        return

    # 병변이 있는 파일만 필터링
    files_with_lesions = []
    for f in tqdm(roi_files, desc="병변 유무 필터링"):
        roi = cv2.imread(os.path.join(roi_dir, f), cv2.IMREAD_UNCHANGED)
        if roi is not None and roi.max() > 0:
            files_with_lesions.append(f)

    log.info(
        f"전체 슬라이스: {len(roi_files)} 장 / 병변 있는 슬라이스: {len(files_with_lesions)} 장"
    )

    # BBox 정보 저장소
    all_bboxes = {}

    # 병변이 있는 파일들에 한해 BBox 추출 후 그리기 적용
    for filename in tqdm(files_with_lesions, desc="BBox 추출 및 시각화 생성"):
        swi_path = os.path.join(swi_dir, filename)
        roi_path = os.path.join(roi_dir, filename)

        swi_img = cv2.imread(swi_path, cv2.IMREAD_UNCHANGED)
        roi_mask = cv2.imread(roi_path, cv2.IMREAD_UNCHANGED)

        if swi_img is None or roi_mask is None:
            continue

        bboxes, contours = _mask_to_bboxes(roi_mask, img_size=swi_img.shape[0])
        all_bboxes[filename] = bboxes

        # SWI(뇌) 시각화 저장
        for mode, d_cnt, d_bb in [
            ("only_contour", True, False),
            ("only_bbox", False, True),
            ("both", True, True),
        ]:
            _save_one_visualization(
                swi_img, contours, bboxes, output_dir_swi, filename, d_cnt, d_bb, mode
            )

        # ROI(마스크) 시각화 저장
        for mode, d_cnt, d_bb in [
            ("only_contour", True, False),
            ("only_bbox", False, True),
            ("both", True, True),
        ]:
            _save_one_visualization(
                roi_mask, contours, bboxes, output_dir_roi, filename, d_cnt, d_bb, mode
            )

    # 최종 JSON 직렬화 저장
    with open(bbox_json_path, "w", encoding="utf-8") as f:
        json.dump(all_bboxes, f, indent=2)

    log.success(
        f"총 {len(all_bboxes)}개 슬라이스의 BBox JSON 저장 완료: {bbox_json_path}"
    )
    log.info(f"SWI 및 ROI 시각화 이미지 덤프 완료!")
