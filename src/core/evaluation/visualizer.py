import cv2
import numpy as np


def visualize_predictions(
    img,
    gt_boxes,
    pred_boxes,
    pred_scores,
    save_path,
    ignored_boxes=None,
    ignored_scores=None,
):
    H, W = img.shape[:2]
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()

    for box in gt_boxes:
        cx, cy, w, h = map(float, box)
        x1, y1 = max(0, int((cx - w / 2) * W)), max(0, int((cy - h / 2) * H))
        x2, y2 = min(W - 1, int((cx + w / 2) * W)), min(H - 1, int((cy + h / 2) * H))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

    for box, score in zip(pred_boxes, pred_scores):
        cx, cy, w, h = map(float, box)
        if not all(np.isfinite([cx, cy, w, h])) or w <= 0 or h <= 0:
            continue

        x1, y1 = max(0, int((cx - w / 2) * W)), max(0, int((cy - h / 2) * H))
        x2, y2 = min(W - 1, int((cx + w / 2) * W)), min(H - 1, int((cy + h / 2) * H))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(
            vis,
            f"{float(score):.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 255),
            1,
        )

    if ignored_boxes is not None and len(ignored_boxes) > 0:
        for box, score in zip(ignored_boxes, ignored_scores):
            cx, cy, w, h = map(float, box)
            if not all(np.isfinite([cx, cy, w, h])) or w <= 0 or h <= 0:
                continue

            x1, y1 = max(0, int((cx - w / 2) * W)), max(0, int((cy - h / 2) * H))
            x2, y2 = (
                min(W - 1, int((cx + w / 2) * W)),
                min(H - 1, int((cy + h / 2) * H)),
            )
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(
                vis,
                f"{float(score):.2f}",
                (x1, max(15, y2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 0, 0),
                1,
            )

    cv2.imwrite(save_path, vis)
