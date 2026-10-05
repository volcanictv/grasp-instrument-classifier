"""Full pipeline on a folder of consecutive frames (about one per second) with boxes you supply.

    python examples/pipeline_example.py path/to/frames boxes.json [--no-sam3]

boxes.json maps a frame file name to its boxes, each [x, y, w, h] in pixels (see examples/boxes_example.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from grasp_classifier.evidential import EvidentialEnsembleClassifier
from grasp_pipeline.pipeline import InstrumentPipeline, semantic_map
from grasp_pipeline.segment import BoxSegmenter
from grasp_pipeline.track import EvidentialTracker


def main() -> None:
    frames_dir, boxes_path = Path(sys.argv[1]), Path(sys.argv[2])
    use_sam3 = "--no-sam3" not in sys.argv
    boxes = json.loads(boxes_path.read_text())
    paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    frames = [np.array(Image.open(p).convert("RGB")) for p in paths]
    w, dev = ROOT / "weights", "cuda:0"
    classifier = EvidentialEnsembleClassifier(ROOT / "evidential_config.yaml", device=dev)
    segmenter = BoxSegmenter(w / "sam2.1_hiera_large.pt", sam2_delta=w / "sam2_delta.pt",
                             sam3_delta=(w / "sam3_delta.pt") if use_sam3 else None, device=dev)
    tracker = EvidentialTracker(classifier, w / "sam2.1_hiera_large.pt", device=dev)
    pipeline = InstrumentPipeline(segmenter, classifier, tracker)
    for i, p in enumerate(paths):
        if p.name not in boxes:
            continue
        results = pipeline.process_frame(frames[i], boxes[p.name], context_frames=frames, center_idx=i)
        for r in results:
            print(f"{p.name}: {r.class_name:<26} epistemic {r.epistemic:.2e} tracked {r.tracked} single-pass {r.single_pass_class}")
        sem = semantic_map(frames[i].shape[:2], results)
        Image.fromarray(sem.astype(np.uint8)).save(frames_dir / f"{p.stem}_semantic.png")


if __name__ == "__main__":
    main()
