import torch
import gc
import os
import numpy as np
from tqdm import tqdm
from src.datasets.transforms import normalize_16bit
from src.utils.logger import log
import config

from .post_process import post_process_batch, get_ignored_fn
from .metrics import match_boxes, precompute_matches, calculate_ap, compute_froc_data
from .visualizer import visualize_predictions

from src.utils.plots import (
    plot_froc,
    plot_confusion_matrix_bar,
    plot_confusion_matrix_heatmap,
)


def evaluate(model, testloader, dataset, device, save_dir):
    """전체 평가: TP/FP/FN, mAP, FROC, Confusion Matrix, 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    total_tp, total_fp, total_fn = 0, 0, 0
    all_preds, all_gts = [], {}
    model.eval()
    img_id, vis_count = 0, 0
    total_gt_lesions = 0
    patient_ids = set()

    with torch.no_grad():
        pbar = tqdm(testloader, desc="평가+시각화")
        for batch_img, _, batch_roi_mask, batch_bboxes in pbar:
            batch_img = batch_img.to(device)
            batch_img = normalize_16bit(batch_img)

            pred_locs, pred_scores, anchors = model(batch_img)
            pred_boxes_list, pred_scores_list = post_process_batch(
                pred_locs, pred_scores, anchors, roi_masks=batch_roi_mask
            )

            for b in range(len(batch_bboxes)):
                pred_boxes = pred_boxes_list[b]
                pred_score = pred_scores_list[b]
                gt_boxes = batch_bboxes[b].to(device)

                tp, fp, fn = match_boxes(pred_boxes, gt_boxes)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_gt_lesions += len(gt_boxes)

                # 환자 ID 추적 (dataset에 idx_to_name이 있으면 환자 로 파싱)
                if hasattr(dataset, "idx_to_name"):
                    img_name = dataset.idx_to_name.get(img_id, "")
                    # 예시 파일명: VK049_swi_slice_012.png → VK049
                    patient_id = (
                        img_name.split("_")[0] if img_name else f"unknown_{img_id}"
                    )
                    patient_ids.add(patient_id)

                all_preds.append((pred_boxes.cpu(), pred_score.cpu(), img_id))
                all_gts[img_id] = gt_boxes.cpu()

                img_np = batch_img[b].detach().cpu().numpy()
                if len(img_np.shape) == 3:
                    img_np = img_np.transpose(1, 2, 0)
                    img_np = (
                        img_np[:, :, 1] if img_np.shape[2] == 3 else img_np[:, :, 0]
                    )
                    img_np = ((img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

                gt_np = gt_boxes.cpu().numpy()
                pred_np = (
                    pred_boxes.cpu().numpy() if len(pred_boxes) > 0 else np.array([])
                )
                pred_s_np = (
                    pred_score.cpu().numpy() if len(pred_score) > 0 else np.array([])
                )

                ign_boxes_np, ign_scores_np = get_ignored_fn(
                    pred_locs[b], pred_scores[b], anchors, gt_boxes
                )

                if len(gt_boxes) > 0 or len(pred_boxes) > 0:
                    filename = (
                        dataset.idx_to_name[img_id]
                        if hasattr(dataset, "idx_to_name")
                        else f"img_{img_id:05d}.png"
                    )
                    visualize_predictions(
                        img_np,
                        gt_np,
                        pred_np,
                        pred_s_np,
                        os.path.join(vis_dir, filename),
                        ignored_boxes=ign_boxes_np,
                        ignored_scores=ign_scores_np,
                    )
                    vis_count += 1
                img_id += 1

            pbar.set_postfix(
                {"TP": total_tp, "FP": total_fp, "FN": total_fn, "vis": vis_count}
            )
            del pred_locs, pred_scores, batch_img
            torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    log.info("\n📐 AP 계산을 위한 데이터 정렬 중...")
    all_detections = precompute_matches(all_preds, all_gts)
    ap = calculate_ap(all_detections, all_gts, config.TEST_IOU_THRESH)

    fps_per_image, sensitivities = compute_froc_data(all_preds, all_gts, img_id)
    if len(fps_per_image) > 0:
        plot_froc(
            fps_per_image, sensitivities, os.path.join(save_dir, "froc_curve.png")
        )

    plot_confusion_matrix_bar(
        total_tp, total_fp, total_fn, os.path.join(save_dir, "confusion_matrix_bar.png")
    )
    plot_confusion_matrix_heatmap(
        total_tp, total_fp, total_fn, os.path.join(save_dir, "confusion_matrix.png")
    )

    fp_per_cbm = total_fp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    results = {
        "환자 수": len(patient_ids),
        "평가 슬라이스 수 (PNG)": img_id,
        "총 GT 병변 수": total_gt_lesions,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "FP/CBM": fp_per_cbm,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        f"AP@{config.TEST_IOU_THRESH} (Target)": ap,
    }

    log.info("=" * 50)
    log.info("📊 평가 결과")
    log.info("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            log.info(f"  {k:<20}: {v:.4f}")
        else:
            log.info(f"  {k}: {v}")

    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    return results
