from src.loss import MultiBoxLoss


def test_multibox_loss():
    print("🧪 Testing MultiBoxLoss with Negative Samples (0 GT)...")

    # 1. Setup Dummy Data
    batch_size = 2
    num_anchors = 100
    num_classes = 2
    device = torch.device("cpu")  # Test on CPU for simplicity

    # Predictions
    pred_loc = torch.randn(batch_size, num_anchors, 4, device=device)
    pred_score = torch.randn(batch_size, num_anchors, num_classes, device=device)
    anchors = torch.rand(num_anchors, 4, device=device)  # [cx, cy, w, h] (0~1)

    # Ground Truth (Batch 0: Positive, Batch 1: Negative)
    gt_bboxes = [
        torch.tensor([[0.5, 0.5, 0.1, 0.1]], device=device),  # Batch 0 has 1 GT
        torch.tensor([], device=device),  # Batch 1 has 0 GT (Empty)
    ]
    gt_labels = [
        torch.tensor([1], device=device),
        torch.tensor([], dtype=torch.long, device=device),
    ]

    # Brain Mask (Dummy)
    brain_masks = torch.ones(batch_size, 1, 64, 64, device=device)

    # 2. Init Loss
    criterion = MultiBoxLoss(
        num_classes=num_classes,
        iou_threshold=0.35,  # Matches global config
        alpha=0.25,
        gamma=2.0,
    ).to(device)

    # 3. Forward
    try:
        loss, cls_loss, loc_loss = criterion(
            pred_loc, pred_score, anchors, gt_bboxes, gt_labels, brain_masks
        )
        print(f"✅ Loss Calculation Successful!")
        print(f"   Total Loss: {loss.item():.4f}")
        print(f"   Cls Loss:   {cls_loss.item():.4f}")
        print(f"   Loc Loss:   {loc_loss.item():.4f}")

        # Check if loss is not NaN
        if torch.isnan(loss):
            print("❌ Loss is NaN!")
        else:
            print("✅ Loss is valid (not NaN).")

    except Exception as e:
        print(f"❌ Error during loss calculation: {e}")
        raise e


if __name__ == "__main__":
    test_multibox_loss()
