# GraSP Instrument Classifier

Given a video frame and a bounding box (from your own detector/segmenter),
classifies the surgical instrument inside it into one of 7 classes, and says
how uncertain it is. This is a **classification-only** component: it does not
locate instruments -- it expects a box (and ideally a mask) as input from an
upstream detection/segmentation stage.

**No softmax anywhere.** Each of the 4 ensemble members runs 20 stochastic
passes with dropout left on (MC Dropout). Every pass votes for the class of
its largest logit; the prediction is the class with the most weighted votes,
and the uncertainty is the share of votes that disagree with it. The optional
temporal tracking uses the same votes to decide when to run and to combine
frames. No class probability is computed or returned.

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

classifier = EnsembleClassifier()  # device="cuda:0" for GPU, seed=0 for repeatable votes
image = np.array(Image.open("frame.jpg").convert("RGB"))  # HxWx3 uint8

result = classifier.predict(image, box_xywh=(x, y, w, h))
print(result.class_name, result.uncertainty, result.needs_tracking)
print(result.vote_shares)  # dict of all 7 classes -> share of votes
```

Pass `mask=` (a boolean HxW array, same size as `image`) if your upstream
stage produces instance masks, not just boxes -- see "Input contract"
below for why this matters. See `examples/predict_example.py` for a
runnable end-to-end example.

Votes use random dropout masks, so two calls on the same input can differ
slightly (mostly on instances that are near a tie). Pass `seed=` to the
constructor for repeatable results.

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

A `Prediction` with:

- `class_name` (str): the class with the most weighted votes.
- `uncertainty` (float, 0-1): the share of votes that disagree with
  `class_name`; 0 means every pass of every member agreed.
- `vote_shares` (dict, all 7 classes): each class's share of the votes, sums
  to 1.
- `needs_tracking` (bool): `uncertainty` reached the configured gate
  (`uncertainty_gate` in `ensemble_config.yaml`, 0.09).
- `tracked` (bool): True only for a result returned by temporal tracking.

Class order (fixed, do not reorder -- see `grasp_classifier/classes.py`):
Bipolar Forceps, Prograsp Forceps, Large Needle Driver, Monopolar Curved
Scissors, Suction Instrument, Clip Applier, Laparoscopic Grasper.

## What this is, measured

4-model weighted ensemble (2x ResNet-50 at different resolutions, 2x
MobileNetV3-Small), each retrained with dropout inside the network,
evaluated on GraSP's official test set (oracle bounding boxes/masks -- given
a real detector/segmenter's own boxes instead, expect a measurable accuracy
cost from imperfect localization, not measured for this package
specifically):

| | accuracy | macro-F1 |
|---|---|---|
| ensemble alone (weighted 0.40, one training run) | 0.9266 | 0.8898 |
| flat-weighted alternative | 0.9238 | 0.8943 |

See `WEIGHTS.md` for why two rows are listed and how to switch between
them -- this is a disclosed, open trade-off, not an oversight. These are
single training runs, so no seed-to-seed spread is given.

`uncertainty` separates wrong from right predictions with an AUROC of 0.896
(0.5 is chance, 1.0 is perfect) on the same test set. It cannot flag the
errors on which every pass and member agrees on the wrong class (8 of the
210 errors).

**Cost**: one deterministic pass of the 4 members takes ~20ms/instance on a
Titan Xp GPU; the 80 votes (4 members x 20 passes) that `predict()` runs take
~97ms and ~640MB of GPU memory. CPU-only (4-core i5, PyTorch): ~172ms for one
pass, ~3.3s for the 80 votes -- the two ResNet-50 members account for
essentially all of it. Pass a smaller `mc_samples` to trade vote stability for
speed. Model size: ~200MB total across the 4 checkpoints.

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
optional, uncertainty-gated correction on top, with no retraining: when the
single-frame `uncertainty` is at or above the gate (0.09), the instance's mask
is propagated forward and backward across nearby video frames with SAM2, every
propagated frame gets the same 80 votes, and the frames are combined by an
uncertainty-weighted vote (each frame counts in proportion to how unanimous it
is; ties go to the dropout-off votes).

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
print(result.class_name, result.uncertainty, result.tracked)
```

`frames_dir` is a directory of consecutive video frames (filenames sort
into chronological order; the research setup used a window of about 10 frames
either side of the instance); `center_idx` is where, in that sorted list,
the given frame/box/mask sit. `predict_with_tracking` runs the cheap
single-frame path first and only invokes SAM2 at or above the gate, matching
the policy this was validated under. For a tracked result, `uncertainty` is
the share of the frames' votes, pooled evenly, that name a different class.

Needs `sam2` and a separately downloaded checkpoint, not installed by
`requirements.txt` -- see `requirements-tracking.txt`. Not required to
use the base classifier; importing `grasp_classifier.tracking` itself
doesn't need `sam2` either, only constructing a `TemporalTracker` does.

**Measured on the full official GraSP test set** (not a projection):

| | tracked | accuracy | macro-F1 |
|---|---|---|---|
| ensemble alone | none | 0.9266 | 0.8898 |
| + tracking at gate 0.20 | 18.4% | 0.9581 | 0.9295 |
| + tracking at gate 0.09 (default) | 29.1% | 0.9623 | 0.9351 |

Every one of the 7 classes improves at the default gate, including the two
weakest (Laparoscopic Grasper 0.776 -> 0.848 F1, Clip Applier 0.861 -> 0.907
F1). The default 0.09 was chosen on this test set as the cheapest gate that
matched an earlier probability-averaging pipeline's accuracy and macro-F1, so
its numbers are optimistic; choosing the gate by leave-one-case-out among
0.20, 0.15, 0.10, 0.05 and 0.03 gives 0.9633 accuracy. Below about 0.06,
tracking more instances stops helping (it breaks about as many correct
predictions as it fixes). Set `uncertainty_gate` in `ensemble_config.yaml`
(or pass `uncertainty_gate=` to `predict_with_tracking`) to change it.

**Cost**: SAM2's own per-frame encoding cost alone (~398ms, reference GPU)
rules out real-time use on any hardware -- this is an offline, batch-applied
correction, not a pipeline stage for a live video feed. Full per-instance
cost when tracking does run: ~24s (SAM2 propagation across a ~20-frame
window plus 80 votes on every frame, about 2s of that on the GPU). See the
parent research repo's `docs/DECISIONS.md`, 2026-09-20 entries, for the full
derivation.

## Migrating from the earlier (probability-averaging) version

The earlier version returned `confidence` and `class_probabilities` from an
averaged softmax, and gated tracking on `confidence < 0.80`. Both are gone:
use `uncertainty` (lower is more certain) and `needs_tracking`, and read
`vote_shares` for the per-class vote split. The checkpoints and
`ensemble_config.yaml` changed with it; the old checkpoints do not load into
the new models. The earlier version stays available in this repository's git
history (commit `92717d3`).

## What's not in this package

Training code, experiment configs, the dataset, and the report/decision
log live in the parent research repository -- this package is the
integration-ready output of that work, not the research process. If you
need to retrain, re-benchmark, or understand *why* a given design choice
was made, that context is there, not here.
