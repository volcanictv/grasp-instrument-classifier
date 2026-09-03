"""Crop preprocessing: reproduces exactly what each checkpoint was trained
on (bbox crop, mask-multiplied to zero out anything in the box that isn't
this instance, then padded to a square before resize) -- upstream
detection/segmentation should hand this package a box and, ideally, an
instance mask; a box alone still works; the two differ in exactly how much
of a second, overlapping instrument gets zeroed out of the crop.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def pad_to_square(crop: np.ndarray) -> np.ndarray:
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    top, left = (side - ch) // 2, (side - cw) // 2
    square[top : top + ch, left : left + cw] = crop
    return square


def crop_instance(
    image: np.ndarray,
    box_xywh: tuple[int, int, int, int],
    mask: np.ndarray | None = None,
    letterbox: bool = True,
) -> Image.Image:
    """`image`: full RGB frame, HxWx3 uint8. `box_xywh`: (x, y, w, h) in the
    frame's native pixel coordinates. `mask`: optional boolean array, same
    HxW as `image` -- pixels inside the box but outside the mask are
    zeroed, matching training (without a mask, the raw box is used as-is,
    which risks including a second, overlapping instrument in the crop).
    `letterbox`: pad to square before any resize, preserving the crop's
    true aspect ratio -- must match the flag each checkpoint was trained
    with (see WEIGHTS.md / ensemble_config.yaml).
    """
    height, width = image.shape[:2]
    x, y, w, h = box_xywh
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)

    crop = image[y0:y1, x0:x1]
    if mask is not None:
        crop = (crop * mask[y0:y1, x0:x1, None]).astype(np.uint8)
    else:
        crop = crop.astype(np.uint8)

    if letterbox:
        ch, cw = crop.shape[:2]
        aspect = max(ch, cw) / max(1, min(ch, cw))
        if aspect >= 1.0:
            crop = pad_to_square(crop)

    return Image.fromarray(crop)


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
