"""Training-frame statistics behind the frame decode (grasp_classifier.decode): how often each class appears in a frame, and how often twice.

    python -m grasp_pipeline.decode_stats --coco annotations/grasp_short-term_train.json --class-set grasp --rule hard \
        --allow "Large Needle Driver" --prior-shift -0.5 --out decode_stats/grasp_official_train.json

The GraSP statistics are computed from the official training cases only (the test cases are never read). For EndoVis the frames are given as a JSON list of
per-frame class-index lists (`--frames-json`), see docs/REPRODUCING_THE_PAPER.md for how the shipped EndoVis file was built.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from grasp_classifier.classes import CLASS_SETS


def frame_statistics(frames: Iterable[Sequence[int]], n_classes: int) -> tuple[list[int], list[int], list[int]]:
    """Per-frame lists of class indices -> (instances per class, frames containing the class, frames containing it more than once)."""
    counts, has, dup = np.zeros(n_classes, int), np.zeros(n_classes, int), np.zeros(n_classes, int)
    for labels in frames:
        per_class = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes)
        counts += per_class
        has += per_class > 0
        dup += per_class > 1
    return counts.tolist(), has.tolist(), dup.tolist()


def coco_frames(path: Path) -> list[list[int]]:
    """Frames of a GraSP short-term annotation file (COCO style) as lists of class indices; the index is the rank of the category id."""
    doc = json.loads(Path(path).read_text())
    index = {cid: i for i, cid in enumerate(sorted(c["id"] for c in doc["categories"]))}
    frames: dict[int, list[int]] = defaultdict(list)
    for a in doc["annotations"]:
        x, y, w, h = a["bbox"]
        if round(w) <= 0 or round(h) <= 0:
            continue
        frames[a["image_id"]].append(index[a["category_id"]])
    return list(frames.values())


def build_stats(frames: Iterable[Sequence[int]], class_set: str, rule: str, allowed: Sequence[str] = (), prior_shift: float = 0.0,
                provenance: dict | None = None) -> dict:
    names = CLASS_SETS[class_set]
    counts, has, dup = frame_statistics(frames, len(names))
    stats = {"class_set": class_set, "class_names": names, "rule": rule, "allowed_duplicates": list(allowed), "class_counts": counts,
             "frames_with_class": has, "frames_with_duplicate": dup, "prior_shift": prior_shift, "provenance": provenance or {}}
    if rule == "soft":
        stats["duplicate_penalty"] = [float(np.log((d + 1) / (h + 2))) for h, d in zip(has, dup)]
    return stats


def merge(*stats: dict) -> dict:
    """Adds the counts of several statistics of the same class set (EndoVis 2018 plus 2017 training frames)."""
    out = json.loads(json.dumps(stats[0]))
    for s in stats[1:]:
        assert s["class_set"] == out["class_set"]
        for key in ("class_counts", "frames_with_class", "frames_with_duplicate"):
            out[key] = [a + b for a, b in zip(out[key], s[key])]
    if out["rule"] == "soft":
        out["duplicate_penalty"] = [float(np.log((d + 1) / (h + 2))) for h, d in zip(out["frames_with_class"], out["frames_with_duplicate"])]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--coco", type=Path, help="GraSP short-term annotation file of the training cases")
    src.add_argument("--frames-json", type=Path, help="JSON list of per-frame class-index lists")
    ap.add_argument("--class-set", choices=sorted(CLASS_SETS), required=True)
    ap.add_argument("--rule", choices=["hard", "soft"], required=True)
    ap.add_argument("--allow", action="append", default=[], help="classes allowed to repeat in a frame (hard rule)")
    ap.add_argument("--prior-shift", type=float, default=0.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frames = coco_frames(args.coco) if args.coco else json.loads(args.frames_json.read_text())
    source = str(args.coco or args.frames_json)
    stats = build_stats(frames, args.class_set, args.rule, args.allow, args.prior_shift, {"source": Path(source).name, "frames": len(frames)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=1))
    print(f"{len(frames)} frames, instances per class {stats['class_counts']}, frames with a repeated class {stats['frames_with_duplicate']}")


if __name__ == "__main__":
    main()
