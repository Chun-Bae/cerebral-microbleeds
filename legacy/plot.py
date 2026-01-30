import os
import seaborn as sns
import torch.nn.functional as F
import numpy as np
import torch
from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # tkinter 백엔드 에러 방지 (반드시 pyplot import 전에 설정)

# FROC Curve (Free-response ROC) 시각화 및 저장


def plot_froc(model, test_loader, device, save_dir="./results", prefix="test", epoch=None):
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    # 파일명 생성 로직 (예: train_froc_epoch10.png 또는 test_froc.png)
    filename = f"{prefix}_froc"
    if epoch is not None:
        filename += f"_epoch{epoch}"
    save_path = os.path.join(save_dir, f"{filename}.png")

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for swi_images, roi_masks, _ in test_loader:
            swi_images, roi_masks = swi_images.to(device), roi_masks.to(device)

            # 모델 예측
            loc_preds, cls_preds = model(swi_images)
            scores = cls_preds

            # [Small Lesion Fix] MaxPool으로 작은 병변 보존 (학습과 동일한 처리)
            B, C, H, W = scores.shape
            roi_masks_resized = F.adaptive_max_pool2d(
                roi_masks.unsqueeze(1).float(),  # (B,1,H0,W0)
                output_size=(H, W)
            ).squeeze(1)                        # (B,H,W)
            roi_masks_bin = (roi_masks_resized > 0).long()

            # CMB 존재 확률만 사용
            scores_prob = torch.softmax(scores, dim=1)[:, 1, :, :]  # (B,H,W)

            # Flatten
            scores_flat = scores_prob.view(B, -1)
            labels_flat = roi_masks_bin.view(B, -1)

            all_preds.append(scores_flat.cpu().numpy())
            all_labels.append(labels_flat.cpu().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    # 디버깅용
    print(f"[{prefix.upper()} FROC] preds shape: {all_preds.shape}, labels shape: {all_labels.shape}")
    print(f"FROC 예측값 (min/max): {np.min(all_preds)} / {np.max(all_preds)}")
    print(f"FROC 정답값 (min/max): {np.min(all_labels)} / {np.max(all_labels)}")

    if all_labels.shape != all_preds.shape:
        raise ValueError(
            f"FROC 계산 시 크기가 불일치 all_labels.shape: {all_labels.shape}, all_preds.shape: {all_preds.shape}"
        )

    # 엄밀히 말하면 이건 ROC지만, 픽셀 단위로 그리면 FROC 느낌으로 쓰는 경우도 많음
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='o', linestyle='-',
             label=f"{prefix} FROC Curve")  # 라벨에 prefix 추가
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"FROC Curve ({prefix}) - CMB Detection")  # 타이틀에 prefix 추가
    plt.legend()

    # 그래프 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 메모리 해제
    print(f"✅ Saved FROC plot to {save_path}")


def plot_confusion_matrix(model, test_loader, device, save_dir="./results", prefix="test", epoch=None):
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    # 파일명 생성 로직
    filename = f"{prefix}_cm"
    if epoch is not None:
        filename += f"_epoch{epoch}"
    save_path = os.path.join(save_dir, f"{filename}.png")

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for swi_images, roi_masks, _ in test_loader:
            swi_images, roi_masks = swi_images.to(device), roi_masks.to(device)
            loc_preds, cls_preds = model(swi_images)   # (B,C,H,W)
            scores = cls_preds

            # [Small Lesion Fix] MaxPool으로 작은 병변 보존 (학습과 동일한 처리)
            B, C, H, W = scores.shape
            roi_masks_resized = F.adaptive_max_pool2d(
                roi_masks.unsqueeze(1).float(),
                output_size=(H, W)
            ).squeeze(1)                     # (B,H,W)
            roi_masks_bin = (roi_masks_resized > 0).float()

            # 확률 → 이진 예측
            scores_prob = torch.softmax(scores, dim=1)[:, 1, :, :]  # (B,H,W)
            scores_bin = (scores_prob > 0.5).float()

            # Flatten
            y_pred = scores_bin.view(B, -1)
            y_true = roi_masks_bin.view(B, -1)

            all_preds.append(y_pred.cpu().numpy())
            all_labels.append(y_true.cpu().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    print(f"[{prefix.upper()} CM] preds shape: {all_preds.shape}, labels shape: {all_labels.shape}")

    if all_labels.shape != all_preds.shape:
        raise ValueError(
            f"Confusion Matrix 계산 시 크기 불일치 all_labels.shape: {all_labels.shape}, all_preds.shape: {all_preds.shape}"
        )

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])  # 2x2 강제

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix ({prefix})")  # 타이틀에 prefix 추가

    # 그래프 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 메모리 해제
    print(f"✅ Saved Confusion Matrix to {save_path}")
