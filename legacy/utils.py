"""
utils.py - CMB 탐지 프로젝트 유틸리티 함수

이 모듈은 프로젝트 전반에서 사용되는 유틸리티 함수들을 정의합니다.

주요 기능:
1. Logger: 콘솔 + 파일 동시 로깅
2. check_image_bit_depth: 이미지 비트 깊이 확인
3. get_bboxes_from_mask: 마스크에서 바운딩 박스 추출
"""

import sys
import os
import datetime
import cv2
import numpy as np


# ==========================================
# Logger 클래스 - 이중 출력 로거
# ==========================================
class Logger(object):
    """
    이중 출력 로거 (Dual Output Logger)

    print 문의 출력을 콘솔과 파일에 동시에 기록합니다.
    sys.stdout을 이 객체로 대체하여 사용합니다.

    사용법:
        sys.stdout = Logger("log.txt")
        print("이 내용은 콘솔과 log.txt에 동시에 기록됩니다")

    특징:
        - 타임스탬프 자동 추가
        - UTF-8 인코딩 지원 (한글 등)
        - 실시간 flush로 버퍼링 문제 방지
    """

    def __init__(self, filename):
        """
        Args:
            filename: 로그를 저장할 파일 경로
        """
        self.terminal = sys.stdout  # 원본 stdout 보존
        self.log = open(filename, "a", encoding='utf-8')  # 로그 파일 (추가 모드)

    def write(self, message):
        """
        메시지를 콘솔과 파일에 동시 출력

        Args:
            message: 출력할 메시지
        """
        if message.strip():  # 빈 줄이 아닌 경우에만 타임스탬프 추가
            timestamp = datetime.datetime.now().strftime(
                "[%Y-%m-%d %H:%M:%S] ")
            self.terminal.write(f"{timestamp}{message}")
            self.log.write(f"{timestamp}{message}")
        else:
            # 빈 줄(개행 등)은 타임스탬프 없이 그대로 출력
            self.terminal.write(message)
            self.log.write(message)

    def flush(self):
        """
        출력 버퍼를 강제로 비움 (실시간 출력 보장)
        """
        self.terminal.flush()
        self.log.flush()


# ==========================================
# 이미지 비트 깊이 확인 함수
# ==========================================
def check_image_bit_depth(folder_path, num_samples=3):
    """
    폴더 내 이미지의 비트 깊이(데이터 타입) 확인

    8비트 이미지가 정상적으로 저장되었는지 검증합니다.

    Args:
        folder_path: 확인할 이미지 폴더 경로
        num_samples: 확인할 샘플 수 (기본값: 3)

    출력:
        - 파일명, dtype, 최대값 정보
    """
    files = sorted(os.listdir(folder_path))[:num_samples]
    print(f"\n[검사] '{folder_path}' 폴더 데이터 검증 중...")

    for f in files:
        path = os.path.join(folder_path, f)

        # IMREAD_UNCHANGED: 원본 비트 깊이 유지
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        dtype = img.dtype      # 데이터 타입 (uint8, uint16 등)
        max_val = img.max()    # 최대 픽셀값

        print(f" - 파일: {f} | dtype: {dtype} | 최대값: {max_val}")

        if dtype == 'uint8':
            print("   ✅ 확인: 8비트 이미지 (0~255).")
        elif dtype == 'uint16':
            print("   ⚠️ 주의: 16비트 이미지 (0~65535). 8비트로 재변환 필요.")

    print("-" * 50 + "\n")


# ==========================================
# 마스크에서 바운딩 박스 추출 함수
# ==========================================
def get_bboxes_from_mask(mask, H, W):
    """
    마스크 이미지에서 모든 병변의 바운딩 박스 추출

    ROI 마스크에서 Contour Detection을 사용하여
    각 병변 영역의 바운딩 박스 좌표를 계산합니다.

    Args:
        mask: 이진 마스크 이미지 (0: 배경, >0: 병변)
              - torch.Tensor 또는 numpy.ndarray
        H: 원본 이미지 높이 (정규화용)
        W: 원본 이미지 너비 (정규화용)

    Returns:
        boxes: 바운딩 박스 리스트
               각 박스: [x_min, y_min, x_max, y_max]
               좌표는 0~1 범위로 정규화됨

    예시:
        mask = cv2.imread("roi.png", cv2.IMREAD_GRAYSCALE)
        boxes = get_bboxes_from_mask(mask, 256, 256)
        # boxes = [[0.1, 0.2, 0.15, 0.25], [0.5, 0.6, 0.55, 0.65]]
    """
    # Tensor인 경우 Numpy로 변환
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()

    # uint8로 변환 (OpenCV 호환)
    mask = mask.astype(np.uint8)

    # 컨투어 검출
    # RETR_EXTERNAL: 외곽 컨투어만 검출 (내부 구멍 무시)
    # CHAIN_APPROX_SIMPLE: 컨투어 점 간소화 (메모리 절약)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    if len(contours) == 0:
        return boxes  # 병변 없음

    # 각 컨투어(병변)에서 바운딩 박스 추출
    for cnt in contours:
        # cv2.boundingRect: 최소 외접 사각형
        x, y, w, h = cv2.boundingRect(cnt)

        # 0~1 범위로 정규화
        x_min = x / W
        y_min = y / H
        x_max = (x + w) / W
        y_max = (y + h) / H

        boxes.append([x_min, y_min, x_max, y_max])

    return boxes
