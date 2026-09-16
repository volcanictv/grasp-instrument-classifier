"""Minimal usage example for confidence-gated SAM2 temporal tracking.
Needs `sam2` installed (see requirements-tracking.txt) and a downloaded
checkpoint + config. Run from the repo root:

    python examples/predict_with_tracking_example.py \\
        path/to/frames_dir 10 x y w h path/to/sam2_checkpoint.pt path/to/sam2_config.yaml

`frames_dir` is a directory of consecutive video frames (filenames sort
into chronological order); the integer after it is the index, within that
sorted list, of the frame the box/mask belong to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grasp_classifier import EnsembleClassifier
from grasp_classifier.tracking import TemporalTracker, predict_with_tracking


def main() -> None:
    if len(sys.argv) != 8:
        print(f"usage: {sys.argv[0]} <frames_dir> <center_idx> <x> <y> <w> <h> <sam2_checkpoint> <sam2_config>")
        sys.exit(1)

    frames_dir, center_idx, x, y, w, h, sam2_checkpoint, sam2_config = sys.argv[1:8]
    center_idx = int(center_idx)
    box_xywh = tuple(map(int, (x, y, w, h)))

    frame_paths = sorted(Path(frames_dir).iterdir())
    image = np.array(Image.open(frame_paths[center_idx]).convert("RGB"))

    # A real mask from your own detector/segmenter goes here. This example
    # uses the box itself as a stand-in so it runs without one -- pass a
    # real instance mask in production, matching what the classifier was
    # trained on (see README's Input contract).
    mask = np.zeros(image.shape[:2], dtype=bool)
    bx, by, bw, bh = box_xywh
    mask[by : by + bh, bx : bx + bw] = True

    classifier = EnsembleClassifier(device="cuda:0")  # "cpu" also works, just slower
    tracker = TemporalTracker(classifier, sam2_checkpoint, sam2_config, device="cuda:0")

    result = predict_with_tracking(classifier, tracker, image, frames_dir, center_idx, box_xywh, mask)

    print(f"predicted: {result.class_name} ({result.confidence:.1%} confidence)")
    print("all class probabilities:")
    for name, prob in sorted(result.class_probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<28} {prob:.3f}")


if __name__ == "__main__":
    main()
