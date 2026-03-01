import kornia.augmentation as K


def get_transforms(device):
    train_transform = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomAffine(
            degrees=(-30, 30),
            translate=(0.03, 0.03),
            scale=(0.8, 0.95),
            p=0.8,
            keepdim=True,
        ),
        K.ColorJitter(brightness=(1.2, 1.5), p=0.5),
        K.RandomBrightness(brightness=(0.1, 0.3), p=0.5),
        K.RandomGaussianBlur(kernel_size=(19, 19), sigma=(0.0, 3.0), p=0.5),
        data_keys=["image", "mask", "mask"],
        same_on_batch=False,
    ).to(device)

    test_transform = None

    return train_transform, test_transform


def normalize_16bit(tensor):
    """
    16-bit (0~65535) → (-1~1) 정규화
    """
    return (tensor / 32767.5) - 1.0


def denormalize_16bit(tensor):
    """
    (-1~1) → (0~65535) 역정규화
    """
    return (tensor + 1.0) * 32767.5
