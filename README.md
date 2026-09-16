# GraSP Instrument Classifier

Given a video frame and a bounding box (from your own detector/segmenter),
classifies the surgical instrument inside it into one of 7 classes.
This is a **classification-only** component: it does not locate
instruments -- it expects a box (and ideally a mask) as input from an
upstream detection/segmentation stage.

## Install

```
pip install -r requirements.txt
```

Add `pip install -r requirements-tracking.txt` only if you use the
optional temporal-tracking correction (see below) -- not needed for the
base classifier.

## Use

```python
import numpy as np
from PIL import Image
from grasp_classifier import EnsembleClassifier

classifier = EnsembleClassifier()  # pass device="cuda:0" for GPU
image = np.array(Image.open("frame.jpg").convert("RGB"))  # HxWx3 uint8

result = classifier.predict(image, box_xywh=(x, y, w, h))
print(result.class_name, result.confidence)
print(result.class_probabilities)  # dict of all 7 classes -> probability
```

Pass `mask=` (a boolean HxW array, same size as `image`) if your upstream
stage produces instance masks, not just boxes -- see "Input contract"
below for why this matters. See `examples/predict_example.py` for a
runnable end-to-end example.

## Input contract

- `image`: full RGB video frame, `HxWx3` `uint8`, native resolution (not
  pre-resized -- the box coordinates must match this frame's own pixel
  space).
- `box_xywh`: `(x, y, w, h)` in that frame's native pixel coordinates,
  one instrument instance.
- `mask` (optional but recommended): boolean array, same `HxW` as
  `image`, `True` where this specific instrument is. Without a mask, the
  raw box crop is used as-is, which risks pulling in a second,
  overlapping instrument -- GraSP frames routinely have 2-3 instruments
  at once, and the classifier was trained on mask-cleaned crops. If your
  upstream stage only gives boxes, it still works, just with a modest
  expected accuracy cost on overlapping-instrument cases (not
  independently re-measured for this package; see the parent research
  repo's `docs/error_analysis.md` for how this was characterized during
  development, "Problem 4" in `docs/imbalance_notes.md`).

## Output

A `Prediction` with `class_name` (str), `confidence` (float, 0-1), and
`class_probabilities` (dict, all 7 classes).

Class order (fixed, do not reorder -- see `grasp_classifier/classes.py`):
Bipolar Forceps, Prograsp Forceps, Large Needle Driver, Monopolar Curved
Scissors, Suction Instrument, Clip Applier, Laparoscopic Grasper.

## What this is, measured

4-model weighted ensemble (2x ResNet-50 at different resolutions, 2x
MobileNetV3-Small), evaluated on GraSP's official test set (oracle
bounding boxes/masks -- given a real detector/segmenter's own boxes
instead, expect a measurable accuracy cost from imperfect localization,
not measured for this package specifically):

| | accuracy | macro-F1 |
|---|---|---|
| current default (weighted 0.40) | 0.934 | 0.903 |
| flat-weighted alternative | 0.927 | 0.893 |

See `WEIGHTS.md` for why two numbers are listed and how to switch between
them -- this is a disclosed, open trade-off, not an oversight.

**Cost**: ~17ms/instance on a GPU (Titan Xp, no tensor cores -- this is
close to a lower bound, not an upper one, on more modern hardware).
CPU-only: roughly 70-75ms/instance (ONNX Runtime, uncontended) -- the two
ResNet-50 members account for essentially all of that; the two
MobileNetV3 members cost under 2ms each. Model size: ~200MB total across
all 4 checkpoints.

**Known limitation, not fixed**: Laparoscopic Grasper is confused with
Suction Instrument specifically when the Grasper's jaws are closed --
confirmed to be a genuine physical-state ambiguity (a closed-jaw crop is a
featureless shaft, visually indistinguishable from a suction tube), not a
training gap. Five independent fixes were tried and ruled out (see the
parent research repo's `docs/DECISIONS.md`, 2026-09-03 entries) --
resolving this would need either more training examples specifically of
this confusion, or a non-visual signal (e.g. instrument kinematic state)
this package does not have access to. Temporal tracking (below) narrows
this gap but does not close it.

## Temporal tracking (optional)

`predict()` above scores one frame. `grasp_classifier.tracking` adds an
optional, confidence-gated correction on top, with no retraining: when
the base ensemble's own confidence on an instance falls below 0.80, its
mask is propagated forward and backward across nearby video frames with
SAM2, every propagated frame is scored with the same ensemble, and the
predictions are averaged.

```python
from grasp_classifier import EnsembleClassifier
from grasp_classifier.tracking import TemporalTracker, predict_with_tracking

classifier = EnsembleClassifier(device="cuda:0")
tracker = TemporalTracker(
    classifier,
    sam2_checkpoint="path/to/sam2.1_hiera_large.pt",
    sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
    device="cuda:0",
)

result = predict_with_tracking(
    classifier, tracker, image, frames_dir="path/to/consecutive/frames",
    center_idx=10, box_xywh=(x, y, w, h), mask=mask,
)
```

`frames_dir` is a directory of consecutive video frames (filenames sort
into chronological order); `center_idx` is where, in that sorted list,
the given frame/box/mask sit. Only pay for tracking on instances that
actually need it -- `predict_with_tracking` runs the cheap single-frame
path first and only invokes SAM2 below the confidence threshold, matching
the policy this was validated under.

Needs `sam2` and a separately downloaded checkpoint, not installed by
`requirements.txt` -- see `requirements-tracking.txt`. Not required to
use the base classifier; importing `grasp_classifier.tracking` itself
doesn't need `sam2` either, only constructing a `TemporalTracker` does.

**Measured on the full official GraSP test set** (not a projection):

| | accuracy | macro-F1 |
|---|---|---|
| ensemble alone | 0.9343 | 0.903 |
| + confidence-gated tracking | 0.9567 | 0.934 |

Every one of the 7 classes improves, including the two weakest
(Laparoscopic Grasper 0.800 -> 0.838 F1, Clip Applier 0.895 -> 0.947 F1).
Applying this to every instance instead of gating it was tested and
rejected: propagation has a measured ~9.8% chance of turning an
already-correct, low-confidence prediction wrong, which costs more than
it gains once applied indiscriminately.

**Cost**: SAM2's own per-frame encoding cost alone (~398ms, reference
GPU) rules out real-time use on any hardware -- this is an offline,
batch-applied correction, not a pipeline stage for a live video feed.
Full per-instance cost when tracking does run: ~13.5s (SAM2 propagation
across a ~20-frame window plus re-scoring every frame with the ensemble).
See the parent research repo's `docs/DECISIONS.md`, 2026-09-15/16
entries, for the full derivation.

## What's not in this package

Training code, experiment configs, the dataset, and the report/decision
log live in the parent research repository -- this package is the
integration-ready output of that work, not the research process. If you
need to retrain, re-benchmark, or understand *why* a given design choice
was made, that context is there, not here.
