import torch
import cv2
import os
import argparse
import config
from src.model import SSD_FE
import numpy as np


def infer(image_path, model_path, output_path):
    device = config.DEVICE

    # Load Model
    model = SSD_FE(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Preprocessing (similar to val transform)
    # ... placeholder ...

    print(f"Inference on {image_path} done. Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument("--output", type=str, required=True, help="Path to save result")
    args = parser.parse_args()

    infer(args.image, args.model, args.output)
