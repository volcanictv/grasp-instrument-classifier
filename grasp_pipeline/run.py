"""Runs the pipeline on a folder of consecutive frames and a JSON of boxes.

    python -m grasp_pipeline.run --frames frames/ --boxes boxes.json --out out/

`boxes.json` maps a frame file name to its boxes, each [x, y, w, h] in native pixels (from your detector, or the ground truth):

    {"00001.jpg": [[120, 80, 210, 160], [400, 300, 150, 220]], "00002.jpg": [...]}

Frames are processed in sorted file-name order and are assumed to be about one second apart. For every frame in `boxes.json` the output holds
`<frame>.json` (one entry per instrument: box, class, epistemic score, whether it was tracked, the single-pass class) and `<frame>.png`
(the semantic map, pixel values 0 = background, 1 to 7 the classes in grasp_classifier.CLASS_NAMES order).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from grasp_classifier.evidential import EvidentialEnsembleClassifier
from grasp_pipeline.pipeline import InstrumentPipeline, semantic_map
from grasp_pipeline.segment import BoxSegmenter
from grasp_pipeline.track import EvidentialTracker

ROOT = Path(__file__).resolve().parent.parent


class LazyFrames:
    """Sequence of frame arrays read from disk on demand, with a small cache (the tracker slices a window around the keyframe)."""

    def __init__(self, paths: list[Path], cache: int = 24):
        self.paths, self._cache, self._limit = paths, {}, cache

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, i: int) -> np.ndarray:
        if i not in self._cache:
            if len(self._cache) >= self._limit:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = np.array(Image.open(self.paths[i]).convert("RGB"))
        return self._cache[i]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self._load(i) for i in range(*key.indices(len(self.paths)))]
        return self._load(key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=ROOT / "weights")
    ap.add_argument("--config", type=Path, default=ROOT / "evidential_config.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-sam3", action="store_true", help="SAM2 alone as the segmenter (no gated download)")
    ap.add_argument("--no-tracking", action="store_true", help="single-pass only")
    ap.add_argument("--window", type=int, default=10, help="frames on each side of the keyframe for tracking")
    ap.add_argument("--causal", action="store_true", help="track backwards only")
    ap.add_argument("--gate", type=float, default=None, help="epistemic gate override")
    ap.add_argument("--clip-to-box", action="store_true", help="clip masks to their boxes (only meaningful for ground-truth boxes)")
    args = ap.parse_args()

    boxes = json.loads(args.boxes.read_text())
    paths = sorted(p for p in args.frames.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    index = {p.name: i for i, p in enumerate(paths)}
    sam2_base = args.weights / "sam2.1_hiera_large.pt"
    classifier = EvidentialEnsembleClassifier(args.config, device=args.device)
    segmenter = BoxSegmenter(sam2_base, sam2_delta=args.weights / "sam2_delta.pt",
                             sam3_delta=None if args.no_sam3 else args.weights / "sam3_delta.pt",
                             device=args.device, clip_to_box=args.clip_to_box)
    tracker = None if args.no_tracking else EvidentialTracker(classifier, sam2_base, device=args.device, window=args.window, causal=args.causal)
    pipeline = InstrumentPipeline(segmenter, classifier, tracker, epistemic_gate=args.gate)
    frames = LazyFrames(paths)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame_boxes in boxes.items():
        if name not in index:
            print(f"skipping {name}: no such frame in {args.frames}")
            continue
        i = index[name]
        results = pipeline.process_frame(frames[i], frame_boxes, context_frames=frames, center_idx=i)
        stem = Path(name).stem
        Image.fromarray(semantic_map(frames[i].shape[:2], results).astype(np.uint8)).save(args.out / f"{stem}.png")
        (args.out / f"{stem}.json").write_text(json.dumps([
            {"box_xywh": list(r.box_xywh), "class": r.class_name, "epistemic": r.epistemic, "tracked": r.tracked,
             "single_pass_class": r.single_pass_class, "belief": r.belief} for r in results], indent=1))
        print(f"{name}: {len(results)} instruments, {sum(r.tracked for r in results)} tracked")


if __name__ == "__main__":
    main()
