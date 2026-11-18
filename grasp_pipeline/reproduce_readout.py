"""Recomputes the readout numbers of the paper from saved fused class beliefs, with no model, GPU or dataset.

    python -m grasp_pipeline.reproduce_readout              # prints the table and checks it against reproduce/expected.json
    python -m grasp_pipeline.reproduce_readout --write-expected

`reproduce/*.npz` hold, per classifier seed, the class beliefs of every test instrument after the first temporal revisit (the single-pass belief, or the
confidence-weighted fused belief of its SAM2 track for the instruments the gate flagged), the true class, and an integer id per frame. They are what the
research pipeline produced with ground-truth masks; nothing else (no file names, boxes or masks) is stored. This reproduces the accuracy and macro-F1
columns and the effect of the frame decode. The mIoU, IoU and mcIoU columns need the datasets' masks: docs/REPRODUCING_THE_PAPER.md says how.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from grasp_classifier.decode import FrameDecoder

ROOT = Path(__file__).resolve().parent.parent
TOLERANCE = 5e-4  # a twentieth of a percentage point

DATASETS = {
    "grasp": {"arrays": ROOT / "reproduce" / "grasp_test_gtmask.npz", "stats": ROOT / "decode_stats" / "grasp_official_train.json",
              "title": "GraSP official test, ground-truth masks (2,861 instruments)",
              "variants": [("revisit, argmax", None), ("revisit + decode (hard rule)", 0.0), ("revisit + decode + prior shift -0.5 (paper row)", -0.5)]},
    "endovis18": {"arrays": ROOT / "reproduce" / "endovis18_val_gtmask.npz", "stats": ROOT / "decode_stats" / "endovis18_train_plus_2017.json",
                  "title": "EndoVis 2018 ISINet-split validation, annotation masks (1,441 instruments), members trained on 2018 + 2017",
                  "variants": [("revisit, argmax", None), ("revisit + decode (soft penalties; paper row)", 0.0)]},
}


def accuracy_and_macro_f1(pred: np.ndarray, y: np.ndarray, n_classes: int = 7) -> tuple[float, float]:
    f1 = [2 * np.sum((pred == c) & (y == c)) / max(1, np.sum(pred == c) + np.sum(y == c)) for c in range(n_classes)]
    return float(np.mean(pred == y)), float(np.mean(f1))


def compute() -> dict:
    out: dict = {}
    for name, spec in DATASETS.items():
        data = np.load(spec["arrays"])
        y, group = data["y"].astype(int), data["group"]
        seeds = sorted(int(k.split("_s")[1]) for k in data.files if k.startswith("prob_s"))
        out[name] = {}
        for label, shift in spec["variants"]:
            rows = {}
            for seed in seeds:
                prob = data[f"prob_s{seed}"].astype(float)
                if shift is None:
                    pred = prob.argmax(axis=1)
                else:
                    pred = FrameDecoder.from_stats(spec["stats"], prior_shift=shift).decode_frames(prob, group)
                acc, f1 = accuracy_and_macro_f1(pred, y)
                rows[str(seed)] = {"accuracy": round(100 * acc, 2), "macro_f1": round(100 * f1, 2)}
            rows["mean"] = {k: round(float(np.mean([rows[str(s)][k] for s in seeds])), 2) for k in ("accuracy", "macro_f1")}
            out[name][label] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-expected", action="store_true")
    args = ap.parse_args()
    result = compute()
    for name, variants in result.items():
        print(f"\n{DATASETS[name]['title']}")
        print(f"  {'readout':<52}{'seed':>6}{'accuracy':>10}{'macro-F1':>10}")
        for label, rows in variants.items():
            for seed, m in rows.items():
                print(f"  {label if seed == next(iter(rows)) else '':<52}{seed:>6}{m['accuracy']:>10.2f}{m['macro_f1']:>10.2f}")
    expected_path = ROOT / "reproduce" / "expected.json"
    if args.write_expected:
        expected_path.write_text(json.dumps(result, indent=1))
        print(f"\nwrote {expected_path}")
        return
    expected = json.loads(expected_path.read_text())
    bad = [(n, v, s, k, expected[n][v][s][k], result[n][v][s][k]) for n in result for v in result[n] for s in result[n][v] for k in result[n][v][s]
           if abs(expected[n][v][s][k] - result[n][v][s][k]) > TOLERANCE * 100]
    if bad:
        for row in bad:
            print("MISMATCH", row)
        raise SystemExit(1)
    print("\nall numbers agree with reproduce/expected.json")


if __name__ == "__main__":
    main()
