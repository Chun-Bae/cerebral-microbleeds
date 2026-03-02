import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def plot_froc(fps_per_image, sensitivities, save_path):
    """
    FROC 곡선 그리기 (Style matched to user reference)
    """
    plt.figure(figsize=(8, 6))

    # User style: Line with markers, legend
    plt.plot(
        fps_per_image,
        sensitivities,
        "o-",
        linewidth=1.5,
        markersize=4,
        label="test FROC Curve",
    )

    plt.xlabel("Average False Positives per Image", fontsize=12)
    plt.ylabel(
        "True Positive Rate (Sensitivity)", fontsize=12
    )  # Label matched to image y-axis
    plt.title(
        "FROC Curve (test) - CMB Detection", fontsize=14
    )  # Title matched to image

    # Axis limits considering the data spread
    if len(fps_per_image) > 0:
        plt.xlim([0, max(fps_per_image) * 1.05])
    plt.ylim([0, 1.05])

    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ FROC 곡선 저장: {save_path}")


def plot_confusion_matrix_bar(tp, fp, fn, save_path):
    """
    Plot Confusion Matrix as a Bar Chart.

    Args:
        tp (int): True Positives count.
        fp (int): False Positives count.
        fn (int): False Negatives count.
        save_path (str): Path to save the plot.
    """
    labels = ["True Positive (TP)", "False Positive (FP)", "False Negative (FN)"]
    counts = [tp, fp, fn]
    colors = ["green", "red", "orange"]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color=colors)

    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.title("Confusion Matrix Counts", fontsize=14)
    plt.ylabel("Count", fontsize=12)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix_heatmap(tp, fp, fn, save_path):
    """
    Plot Confusion Matrix as a Heatmap.
    This is a simplified version since TN is not applicable for object detection.

    Args:
        tp (int): True Positives count.
        fp (int): False Positives count.
        fn (int): False Negatives count.
        save_path (str): Path to save the plot.
    """
    # Create a 2x2 matrix
    # Format:
    #                 Predicted Positive    Predicted Negative
    # Actual Positive       TP                    FN
    # Actual Negative       FP                    N/A (TN)

    matrix = np.array([[tp, fn], [fp, 0]])  # 0 for TN just for shape

    # Annotations
    annot = np.array([[f"TP\n{tp}", f"FN\n{fn}"], [f"FP\n{fp}", "N/A"]])

    plt.figure(figsize=(8, 6))

    # Mask out the TN cell for coloring if desired, or just show it as grey
    sns.heatmap(
        matrix, annot=annot, fmt="", cmap="Blues", cbar=False, annot_kws={"size": 16}
    )

    plt.title("Confusion Matrix (Detection)", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.xticks([0.5, 1.5], ["Positive", "Negative"])
    plt.yticks([0.5, 1.5], ["Positive", "Negative"])

    plt.savefig(save_path)
    plt.close()
