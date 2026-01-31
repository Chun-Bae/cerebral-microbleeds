import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torchvision.ops import nms
from utils import decode, jaccard
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CMBsDatasetLMDB, collate_fn, BBOX_JSON_PATH
from model import SSD_FE
from utils import Logger
import sys
import platform

# 한글 폰트 설정 (NanumGothic이 없으면 기본 폰트 사용)
import matplotlib.font_manager as fm

available_fonts = [f.name for f in fm.fontManager.ttflist]
if "NanumGothic" in available_fonts:
    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False


def post_process(pred_locs, pred_scores, anchors, conf_thresh=0.5, nms_thresh=0.3):
    """
    약 60,000개 앵커 → 최종 예측 0~50개로 필터링

    Args:
        pred_locs: (N, 4) - 모델이 예측한 offset [dcx, dcy, dw, dh]
        pred_scores: (N, 2) - [배경 확률, 병변 확률]
        anchors: (N, 4) - 앵커 박스 [cx, cy, w, h]
        conf_thresh: confidence 임계값 (기본 0.5)
        nms_thresh: NMS IoU 임계값 (기본 0.3)

    Returns:
        final_boxes: (M, 4) - 최종 예측 박스 [cx, cy, w, h]
        final_scores: (M,) - 최종 confidence scores
    """
    # 1. Softmax → 병변 클래스 확률만 추출
    probs = F.softmax(pred_scores, dim=-1)
    scores = probs[:, 1]
    # print(f"Max score: {scores.max()}, Min score: {scores.min()}, Mean score: {scores.mean()}")
    # top_k_val = 20
    # if len(scores) > top_k_val:
    #     scores, idx = scores.topk(top_k_val)
    #     pred_locs = pred_locs[idx]
    #     anchors = anchors[idx]

    # 2. Confidence 필터링
    mask = scores > conf_thresh
    # 병변이 없으면 예측 x
    if mask.sum() == 0:
        return torch.zeros((0, 4), device=anchors.device), torch.zeros(
            (0,), device=anchors.device
        )

    # mask 되는 샘플은 다같이 지워주기
    filtered_scores = scores[mask]
    filtered_locs = pred_locs[mask]
    filter_anchors = anchors[mask]

    # 3. Offset → 실제 좌표 디코딩
    # decode: 앵커 + offset → 실제 bbox [cx, cy, w, h]
    boxes = decode(filtered_locs, filter_anchors)

    # 4. [cx, cy, w, h] → [x1, y1, x2, y2] 변환
    boxes_xyxy = torch.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    # print(f"Before NMS: {len(boxes_xyxy)}")
    # 5. NMS (Non-Maximum Suppression)
    keep = nms(boxes_xyxy, filtered_scores, nms_thresh)

    # 6. 최종 결과 (cx, cy, w, h 유지)
    final_boxes = boxes[keep]
    final_scores = filtered_scores[keep]

    return final_boxes, final_scores


def post_process_batch(
    pred_locs, pred_scores, anchors, conf_thresh=0.5, nms_thresh=0.3
):
    """
    배치 단위 후처리

    Args:
        pred_locs: (B, N, 4)
        pred_scores: (B, N, 2)
        anchors: (N, 4)

    Returns:
        all_boxes: list of (M_i, 4) tensors
        all_scores: list of (M_i,) tensors
    """
    batch_size = pred_locs.size(0)
    all_boxes = []
    all_scores = []

    for b in range(batch_size):
        boxes, scores = post_process(
            pred_locs[b], pred_scores[b], anchors, conf_thresh, nms_thresh
        )
        all_boxes.append(boxes)
        all_scores.append(scores)

    return all_boxes, all_scores


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.3):
    """
    예측 vs 정답 매칭 → TP, FP, FN 계산

    Args:
        pred_boxes: (M, 4) - 모델 예측
        gt_boxes: (K, 4) - 정답
        iou_threshold: 매칭 기준

    Returns:
        tp: True Positive 개수
        fp: False Positive 개수
        fn: False Negative 개수
    """

    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)  # 모두 놓침

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0  # 모두 오탐

    ious = jaccard(gt_boxes, pred_boxes)

    # 매칭 (Greedy 방식)
    matched_gt = set()
    tp = 0

    for pred_idx in range(len(pred_boxes)):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx in range(len(gt_boxes)):
            if gt_idx in matched_gt:
                continue

            if ious[gt_idx, pred_idx] > best_iou:
                best_iou = ious[gt_idx, pred_idx]
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn


def compute_ap(all_preds, all_gts, iou_threshold=0.3):
    """
    all_preds: [(boxes, scores, image_id), ...] 전체 예측
    all_gts: {image_id: gt_boxes} 전체 정답
    """
    # 1. 모든 예측을 confidence 순으로 정렬
    all_detections = []
    for boxes, scores, img_id in tqdm(all_preds, desc="AP 데이터 수집"):
        for box, score in zip(boxes, scores):
            all_detections.append((score, box, img_id))

    # 내림차순
    print("정렬 시작...")
    all_detections.sort(key=lambda x: x[0], reverse=True)
    print("정렬 완료!")

    # 2. 전체 GT 개수
    total_gt = sum(len(gt) for gt in all_gts.values())

    if total_gt == 0:
        return 0.0

    matched = {img_id: set() for img_id in all_gts.keys()}

    # 3. 하나씩 추가하면서　TP／FP 평가
    tp_list = []
    fp_list = []

    for score, box, img_id in tqdm(all_detections, desc="AP 계산"):
        gt_boxes = all_gts.get(img_id, [])

        # GT 아닌데 긍정 예측
        if len(gt_boxes) == 0:
            tp_list.append(0)
            fp_list.append(1)
            continue

        # 가장 IoU 높은 GT 찾기
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            # 이미 매칭
            if gt_idx in matched[img_id]:
                continue

            # 단일 IoU
            iou = jaccard(gt_box.unsqueeze(0), box.unsqueeze(0))[0, 0].item()
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            # TP
            tp_list.append(1)
            fp_list.append(0)
        else:
            # FP
            tp_list.append(0)
            fp_list.append(1)

    # 4. 누적 TP, FP 계산
    tp_cumsum = torch.cumsum(torch.tensor(tp_list), dim=0)
    fp_cumsum = torch.cumsum(torch.tensor(fp_list), dim=0)

    # Precision, Recall 계산
    total_pred = tp_cumsum + fp_cumsum

    if total_pred[-1] > 0:
        precisions = tp_cumsum / total_pred
    else:
        precisions = torch.zeros_like(tp_cumsum)

    if total_gt > 0:
        recalls = tp_cumsum / total_gt
    else:
        recalls = torch.zeros_like(tp_cumsum)

    # 5. AP 계산 (11-point interpolation 또는 all-point)
    # All-point interpolation
    precisions = torch.cat([torch.tensor([1.0]), precisions])
    recalls = torch.cat([torch.tensor([0.0]), recalls])

    # Precision을 monotonically decreasing으로 만들기
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 면적 계산
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap.item()


def plot_froc(fps_per_image, sensitivities, save_path):
    """
    FROC 곡선 그리기

    Args:
        fps_per_image: 이미지당 FP 개수 리스트
        sensitivities: 해당 시점의 sensitivity 리스트
        save_path: 저장 경로
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fps_per_image, sensitivities, "b-", linewidth=2)
    plt.xlabel("Average False Positives per Image", fontsize=12)
    plt.ylabel("Sensitivity (Recall)", fontsize=12)
    plt.title("FROC Curve", fontsize=14)
    plt.xlim([0, max(fps_per_image) + 0.5])
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)

    # 주요 FP 지점에서 sensitivity 표시
    key_fps = [0.5, 1, 2, 4, 8]
    for fp in key_fps:
        if fp <= max(fps_per_image):
            # 보간으로 해당 FP에서의 sensitivity 찾기
            idx = np.searchsorted(fps_per_image, fp)
            if idx < len(sensitivities):
                sens = sensitivities[idx]
                plt.scatter([fp], [sens], s=50, zorder=5)
                plt.annotate(
                    f"{sens:.2f}",
                    (fp, sens),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=9,
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ FROC 곡선 저장: {save_path}")


def plot_confusion_matrix(tp, fp, fn, save_path):
    """
    Confusion Matrix 그리기 (Object Detection용)

    Note: TN은 object detection에서 정의하기 어려움
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # 2x2가 아닌 바 형태로 표시
    categories = [
        "True Positive\n(정탐)",
        "False Positive\n(오탐)",
        "False Negative\n(미탐)",
    ]
    values = [tp, fp, fn]
    colors = ["#4CAF50", "#F44336", "#FF9800"]  # 초록, 빨강, 주황

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=1.2)

    # 값 표시
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Detection Results", fontsize=14)

    # Precision, Recall 표시
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    info_text = f"Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}"
    ax.text(
        0.98,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Confusion Matrix 저장: {save_path}")


def compute_froc_data(all_preds, all_gts, num_images, iou_threshold=0.3):
    """
    FROC 곡선용 데이터 계산

    Returns:
        fps_per_image: 이미지당 평균 FP 리스트
        sensitivities: sensitivity 리스트
    """
    # 모든 예측을 confidence 순으로 정렬
    all_detections = []
    for boxes, scores, img_id in tqdm(all_preds, desc="FROC 데이터 수집"):
        for box, score in zip(boxes, scores):
            all_detections.append((score.item(), box, img_id))

    all_detections.sort(key=lambda x: x[0], reverse=True)

    total_gt = sum(len(gt) for gt in all_gts.values())
    matched = {img_id: set() for img_id in all_gts.keys()}

    fps_per_image = []
    sensitivities = []

    tp_count = 0
    fp_count = 0

    for score, box, img_id in tqdm(all_detections, desc="FROC 계산"):
        gt_boxes = all_gts.get(img_id, [])

        # IoU 확인
        is_tp = False
        if len(gt_boxes) > 0:
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched[img_id]:
                    continue
                iou = jaccard(gt_box.unsqueeze(0), box.unsqueeze(0))[0, 0].item()
                if iou >= iou_threshold:
                    is_tp = True
                    matched[img_id].add(gt_idx)
                    break

        if is_tp:
            tp_count += 1
        else:
            fp_count += 1

        # 현재 시점의 FP/image, sensitivity
        fps_per_image.append(fp_count / num_images)
        sensitivities.append(tp_count / total_gt if total_gt > 0 else 0)

    return fps_per_image, sensitivities


def visualize_predictions(
    img, gt_boxes, pred_boxes, pred_scores, save_path, iou_threshold=0.3, min_box_size=8
):
    H, W = img.shape[:2]

    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    # GT 그리기 (초록색)
    for box in gt_boxes:
        cx, cy, w, h = map(float, box)
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)

        # 안전하게 정수 변환 및 범위 제한
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Pred 그리기 (빨간색)
    for box, score in zip(pred_boxes, pred_scores):
        cx, cy, w, h = map(float, box)

        # 유효하지 않은 값(NaN, Inf) 또는 크기가 0인 박스 제외
        if not all(np.isfinite([cx, cy, w, h])) or w <= 0 or h <= 0:
            continue

        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)

        # 최소 크기 보장 및 정수 변환
        try:
            ix1, iy1 = int(max(0, x1)), int(max(0, y1))
            ix2, iy2 = int(min(W - 1, x2)), int(min(H - 1, y2))

            # OpenCV 함수 호출 전 좌표 재검증
            cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                f"{float(score):.2f}",
                (ix1, max(15, iy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )
        except (ValueError, OverflowError):
            continue

    cv2.imwrite(save_path, vis)


def evaluate(
    model, testloader, dataset, device, save_dir, conf_thresh=0.5, iou_thresh=0.3
):
    """
    전체 평가: TP/FP/FN, mAP, FROC, Confusion Matrix, 시각화
    """
    import gc

    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    total_tp, total_fp, total_fn = 0, 0, 0

    # mAP, FROC용 데이터 수집
    all_preds = []  # [(boxes, scores, img_id), ...]
    all_gts = {}  # {img_id: gt_boxes}

    model.eval()
    img_id = 0
    vis_count = 0

    with torch.no_grad():
        pbar = tqdm(testloader, desc="평가+시각화")
        for batch_img, batch_mask, batch_bboxes in pbar:
            batch_img = batch_img.to(device)

            # 1. 추론
            pred_locs, pred_scores, anchors = model(batch_img, batch_mask.to(device))

            # 2. 후처리
            pred_boxes_list, pred_scores_list = post_process_batch(
                pred_locs, pred_scores, anchors, conf_thresh
            )

            # 3. 배치 내 각 이미지 처리
            for b in range(len(batch_bboxes)):
                pred_boxes = pred_boxes_list[b]
                pred_score = pred_scores_list[b]
                gt_boxes = batch_bboxes[b].to(device)

                # TP, FP, FN
                tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_thresh)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                # mAP, FROC용 (CPU로 이동하여 저장)
                all_preds.append((pred_boxes.cpu(), pred_score.cpu(), img_id))
                all_gts[img_id] = gt_boxes.cpu()

                # 시각화 저장 (MRI: 16-bit grayscale → 8-bit)
                img_np = batch_img[b].cpu().numpy()
                if len(img_np.shape) == 3:
                    img_np = img_np.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
                    # 3채널 grayscale이면 1채널만 사용
                    if img_np.shape[2] == 3:
                        img_np = img_np[:, :, 0]  # 첫 번째 채널만 사용
                    # 16-bit (0~65535) → 8-bit (0~255)
                    img_np = (img_np / 256).clip(0, 255).astype(np.uint8)

                gt_np = gt_boxes.cpu().numpy()
                pred_np = (
                    pred_boxes.cpu().numpy() if len(pred_boxes) > 0 else np.array([])
                )
                pred_s_np = (
                    pred_score.cpu().numpy() if len(pred_score) > 0 else np.array([])
                )

                # 파일명 결정
                if hasattr(dataset, "idx_to_name") and img_id in dataset.idx_to_name:
                    filename = dataset.idx_to_name[img_id]
                else:
                    filename = f"img_{img_id:05d}.png"

                save_path = os.path.join(vis_dir, filename)
                visualize_predictions(img_np, gt_np, pred_np, pred_s_np, save_path)
                vis_count += 1

                img_id += 1

            # tqdm 업데이트
            pbar.set_postfix(
                {
                    "TP": total_tp,
                    "FP": total_fp,
                    "FN": total_fn,
                    "pred_boxes": len(pred_boxes),
                    "vis": vis_count,
                }
            )

            # GPU 메모리 정리
            del pred_locs, pred_scores, batch_img
            torch.cuda.empty_cache()

    # 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()

    # 4. 지표 계산
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # 5. AP 계산
    ap = compute_ap(all_preds, all_gts, iou_thresh)

    # 6. FROC 데이터 계산 & 플롯
    fps_per_image, sensitivities = compute_froc_data(
        all_preds, all_gts, img_id, iou_thresh
    )
    if len(fps_per_image) > 0:
        plot_froc(
            fps_per_image, sensitivities, os.path.join(save_dir, "froc_curve.png")
        )

    # 7. Confusion Matrix 플롯
    plot_confusion_matrix(
        total_tp, total_fp, total_fn, os.path.join(save_dir, "confusion_matrix.png")
    )

    # 8. 결과 저장
    results = {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AP": ap,
    }

    # 로그 출력
    print("\n" + "=" * 50)
    print("  📊 평가 결과")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 50)
    print(f"  📁 시각화 저장: {vis_count}개 → {vis_dir}")
    print("=" * 50)

    # 결과 파일로 저장
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    # 최종 메모리 정리
    del all_preds, all_gts
    gc.collect()

    return results


if __name__ == "__main__":
    import datetime
    import argparse
    from torch.utils.data import Subset

    parser = argparse.ArgumentParser(description="CMB Detection Evaluation")
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Specific patient ID to evaluate (e.g., VK049)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="weights/latest_ssd_fold_0.pth",
        help="Path to model weights",
    )
    parser.add_argument(
        "--lmdb_path",
        type=str,
        default="data/lmdb/holdout_test.lmdb",
        help="Path to LMDB dataset",
    )
    args = parser.parse_args()

    # ==================== 설정 ====================
    MODEL_PATH = args.model
    LMDB_PATH = args.lmdb_path
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    CONF_THRESH = 0.5
    IOU_THRESH = 0.3

    # 타임스탬프 폴더 생성
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d(%Hh-%Mm-%Ss)")
    if args.patient:
        SAVE_DIR = os.path.join("results", f"eval_{args.patient}_{timestamp}")
    else:
        SAVE_DIR = os.path.join("results", f"eval_{timestamp}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    # ===============================================

    # 로거 설정

    sys.stdout = Logger(os.path.join(SAVE_DIR, "log.txt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 모델 로드
    model = SSD_FE(num_classes=2).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"✅ 모델 로드: {MODEL_PATH}")

    # 데이터 로드
    dataset = CMBsDatasetLMDB(LMDB_PATH, BBOX_JSON_PATH)

    # 환자 필터링
    if args.patient:
        print(f"🔍 환자 '{args.patient}' 검색 중...")
        target_indices = []
        new_idx_to_name = {}

        # 필터링 및 이름 매핑 재구성
        new_idx = 0
        for i in range(len(dataset)):
            name = dataset.idx_to_name.get(i, "")
            if args.patient in name:
                target_indices.append(i)
                new_idx_to_name[new_idx] = name
                new_idx += 1

        if len(target_indices) == 0:
            print(f"❌ Error: 환자 '{args.patient}' 데이터를 찾을 수 없습니다.")
            sys.exit(1)

        print(f"✅ 환자 '{args.patient}' 데이터 발견: {len(target_indices)}장")

        # Subset 생성
        original_dataset = dataset
        dataset = Subset(original_dataset, target_indices)
        # Subset에 idx_to_name 속성 추가 (evaluate 함수 호환성)
        dataset.idx_to_name = new_idx_to_name

    testloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"✅ 데이터 로드: {len(dataset)} samples")

    # 평가 실행
    results = evaluate(
        model,
        testloader,
        dataset,
        device,
        SAVE_DIR,
        conf_thresh=CONF_THRESH,
        iou_thresh=IOU_THRESH,
    )
