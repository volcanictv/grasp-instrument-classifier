"""Minimal usage example. Run from the repo root:

    python examples/predict_example.py path/to/frame.jpg x y w h
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grasp_classifier import EnsembleClassifier


def main() -> None:
    if len(sys.argv) != 6:
        print(f"usage: {sys.argv[0]} <image_path> <x> <y> <w> <h>")
        sys.exit(1)

    image_path, x, y, w, h = sys.argv[1], *map(int, sys.argv[2:6])
    image = np.array(Image.open(image_path).convert("RGB"))

    classifier = EnsembleClassifier()  # add device="cuda:0" for GPU, seed=0 for repeatable votes
    result = classifier.predict(image, box_xywh=(x, y, w, h))

    print(f"predicted: {result.class_name} (uncertainty {result.uncertainty:.0%}, "
          f"{'worth tracking' if result.needs_tracking else 'confident enough to keep'})")
    print("share of votes per class:")
    for name, share in sorted(result.vote_shares.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<28} {share:.3f}")


if __name__ == "__main__":
    main()
