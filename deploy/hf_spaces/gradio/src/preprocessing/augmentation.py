from __future__ import annotations

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

INPUT_SIZE = 224


def get_train_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(INPUT_SIZE, INPUT_SIZE, interpolation=cv2.INTER_LANCZOS4),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            border_mode=cv2.BORDER_REFLECT,
            p=0.7,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.2),
        A.CoarseDropout(max_holes=8, max_height=INPUT_SIZE // 8, max_width=INPUT_SIZE // 8, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(INPUT_SIZE, INPUT_SIZE, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_inference_transforms() -> A.Compose:
    return get_val_transforms()
