"""Turns a GraSP short-term annotation file into the inputs of `grasp_pipeline.run` for the paper's setting (ground-truth boxes and masks given).

    python -m grasp_pipeline.gt_inputs --coco annotations/grasp_short-term_test.json --out inputs/

Writes, per case directory of the annotation file (frames are tracked within one case, so run the pipeline once per case):

    inputs/<CASE>/boxes.json           {"<frame file>": [[x, y, w, h], ...]}   boxes in native pixels, one per instrument
    inputs/<CASE>/masks/<stem>.npz     masks: (instruments, H, W) bool, in the order of that frame's boxes

and then, for each case:

    python -m grasp_pipeline.run --frames <GraSP>/frames-001/frames/<CASE> --boxes inputs/<CASE>/boxes.json --masks-dir inputs/<CASE>/masks --out out/<CASE>

Annotations with an empty box are skipped, as in the research code. The frames of a case must be the 1 Hz frames in file-name order.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def decode_rle(segmentation: dict) -> np.ndarray:
    from pycocotools import mask as mask_utils

    counts = segmentation["counts"]
    rle = {"size": segmentation["size"], "counts": counts.encode("utf-8") if isinstance(counts, str) else counts}
    return mask_utils.decode(rle).astype(bool)


def build_inputs(doc: dict) -> dict[str, dict[str, tuple[list, list]]]:
    """case -> frame file name -> (boxes xywh, masks), from a loaded COCO-style annotation document."""
    images = {img["id"]: img["file_name"] for img in doc["images"]}
    per_frame: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for a in doc["annotations"]:
        x, y, w, h = (int(round(v)) for v in a["bbox"])
        if w <= 0 or h <= 0:
            continue
        boxes, masks = per_frame[images[a["image_id"]]]
        boxes.append([x, y, w, h])
        masks.append(decode_rle(a["segmentation"]))
    out: dict[str, dict[str, tuple[list, list]]] = defaultdict(dict)
    for file_name, value in per_frame.items():
        case, name = file_name.split("/")
        out[case][name] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cases = build_inputs(json.loads(args.coco.read_text()))
    for case, frames in sorted(cases.items()):
        case_dir = args.out / case
        (case_dir / "masks").mkdir(parents=True, exist_ok=True)
        (case_dir / "boxes.json").write_text(json.dumps({name: boxes for name, (boxes, _m) in sorted(frames.items())}))
        for name, (_boxes, masks) in frames.items():
            np.savez_compressed(case_dir / "masks" / f"{Path(name).stem}.npz", masks=np.stack(masks))
        print(f"{case}: {len(frames)} frames, {sum(len(b) for b, _ in frames.values())} instruments")


if __name__ == "__main__":
    main()
