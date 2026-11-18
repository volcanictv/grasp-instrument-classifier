# GraSP Instrument Pipeline

Surgical instrument recognition for the GraSP benchmark (7 instrument classes), from frames and boxes to instance masks, classes, an uncertainty score
and a temporal correction for the uncertain cases. Everything needed to run it is in this repository: the code, the configs, the weights manifest and
a command-line runner.

```
frame + boxes ──► BoxSegmenter ──► masks ──► evidential ensemble ──► class + uncertainty ──► (uncertain?) ──► SAM2 tracking ──► corrected class
                  SAM2 (+ SAM3)               4 members, 1 pass         epistemic score S1                    ±10 frames
```

It takes boxes as input, from your own detector or from ground truth. It does not detect instruments.

**Two settings are reported here, and they are different.** The paper studies *recognition with given masks*: every method receives the ground-truth
instance masks and only has to say which instrument each one is. Those results, for GraSP and EndoVis 2018, are in "Results of the paper" below and
are what `docs/REPRODUCING_THE_PAPER.md` reproduces. The older section "Earlier results: SAM-predicted masks" scores the same classifier behind the
box-prompted segmenter and does not use the frame decode.

## Quickstart

```bash
git clone <this repo> && cd grasp-instrument-classifier
pip install -r requirements.txt -r requirements-pipeline.txt     # see "Install" for the SAM2 package
python -m grasp_pipeline.weights --fetch                        # downloads the weights (about 2.7 GB) and checks every checksum
python -m grasp_pipeline.run --frames path/to/frames --boxes boxes.json --out out/ --device cuda
```

`boxes.json` maps a frame file name to its boxes, each `[x, y, w, h]` in native pixels (see `examples/boxes_example.json`):

```json
{"00001.jpg": [[120, 80, 210, 160], [400, 300, 150, 220]]}
```

For every listed frame the output has `<frame>.json` (per instrument: box, class, epistemic score, whether it was tracked, the single-pass class, the
belief over the 7 classes) and `<frame>.png` (the semantic map: 0 background, 1 to 7 the classes in the order below). Frames in the folder are read in
sorted order and are assumed to be about one second apart; the tracker uses the 10 frames on each side of a flagged frame.

From Python (`examples/pipeline_example.py` is a runnable version):

```python
from PIL import Image
import numpy as np
from grasp_classifier.evidential import EvidentialEnsembleClassifier
from grasp_pipeline.segment import BoxSegmenter
from grasp_pipeline.track import EvidentialTracker
from grasp_pipeline.pipeline import InstrumentPipeline, semantic_map

W = "weights"
classifier = EvidentialEnsembleClassifier("evidential_config.yaml", device="cuda:0")
segmenter = BoxSegmenter(f"{W}/sam2.1_hiera_large.pt", sam2_delta=f"{W}/sam2_delta.pt", sam3_delta=f"{W}/sam3_delta.pt", device="cuda:0")
tracker = EvidentialTracker(classifier, f"{W}/sam2.1_hiera_large.pt", device="cuda:0")      # optional
pipeline = InstrumentPipeline(segmenter, classifier, tracker)

frames = [np.array(Image.open(p).convert("RGB")) for p in paths]                           # consecutive frames, about 1 per second
results = pipeline.process_frame(frames[i], boxes_xywh, context_frames=frames, center_idx=i)
for r in results:
    print(r.class_name, r.epistemic, r.tracked, r.single_pass_class)
sem = semantic_map(frames[i].shape[:2], results)                                           # per-pixel classes, what mIoU is computed on
```

Classes, in the fixed order of every output: Bipolar Forceps, Prograsp Forceps, Large Needle Driver, Monopolar Curved Scissors, Suction Instrument,
Clip Applier, Laparoscopic Grasper.

## What runs, and what you can switch off

| mode | how | needs |
|---|---|---|
| SAM2 + SAM3 segmenter (the measured configuration) | default | the gated SAM3 base weights, see below |
| SAM2 alone | `--no-sam3` | nothing gated |
| single pass, no tracking | `--no-tracking` | much faster; no SAM2 video predictor |
| causal tracking (past frames only) | `--causal` | |
| clip masks to their boxes | `--clip-to-box` | only meaningful for ground-truth boxes |
| given masks, no segmenter (the paper's setting) | `--masks-dir DIR` | `<frame stem>.npz` with `masks` (instruments, H, W) per frame |
| frame decode off | `--no-decode` | reads the largest belief per instrument instead of the joint frame labelling |

SAM3's base weights are a gated model on Hugging Face (`facebook/sam3`): accept the licence on the model page and run `huggingface-cli login` once.
Without that, use `--no-sam3`. **The measured results below are for SAM2 plus SAM3; SAM2 alone with these weights has not been scored.** For scale, adding
SAM3 to the segmenter ensemble was worth about 0.4 mIoU in the step table below.

## Results of the paper (ground-truth masks)

Instance masks are given to every method, so the scores depend only on the assigned classes. Metrics are the benchmark's mIoU, IoU and mcIoU
(`grasp_pipeline/scoring.py`, ported from the MATIS evaluation). The pipeline here: four-member evidential ensemble, SAM2.1 tracking of the instruments the
gate flags (`epistemic_gate`, 1.7e-5) over the 10 frames on each side, confidence-weighted fusion, then the frame decode (`grasp_classifier/decode.py`).
Reproduction steps and the exact scripts are in `docs/REPRODUCING_THE_PAPER.md`.

**GraSP official test** (2,861 instruments, 1,125 frames, 5 cases). Mean of three classifier seeds (42, 43, 44), mean ± SD:

| | mIoU | IoU | mcIoU |
|---|---|---|---|
| **This pipeline** (revisit, frame decode with the hard duplicate rule, prior shift -0.5) | **96.72 ± 0.14** | **95.69 ± 0.15** | 90.24 ± 0.27 |
| the same without the decode and the shift (largest belief per instrument) | 95.95 ± 0.25 | 94.87 ± 0.32 | 89.10 ± 1.09 |
| TAPIS, released model, masks given | 95.48 | 94.90 | **91.44** |
| SlowFast (TAPIS region features), masks given | 95.47 | 94.87 | 91.38 |

Per-class IoU of the first row: Bipolar Forceps 95.1, Prograsp Forceps 87.2, Large Needle Driver 93.5, Monopolar Curved Scissors 99.1, Suction Instrument
92.8, Clip Applier 88.2, Laparoscopic Grasper 75.9 (TAPIS: 94.9, 80.7, 97.0, 98.5, 89.2, 94.2, 85.6). Instance accuracy 96.89 and macro-F1 94.64, against
96.32 and 93.92 without the decode. **mcIoU is 1.2 points below TAPIS**, almost all of it the Laparoscopic Grasper, whose jaws are often not visible in the
crop (a closed grasper is a featureless shaft, like a suction tube). The ± are over classifier seeds only; TAPIS and SlowFast are one released model each.

**GraSP validation** (two-fold cross-validation over the training cases, one seed): 96.98 / 95.96 / 89.37 against TAPIS 96.38 / 95.65 / 91.15.

**EndoVis 2018, ISINet split, validation sequences 2, 5, 9, 15** (1,441 instruments, annotation masks given). Members trained on EndoVis 2018 **plus the complete
EndoVis 2017** (ISINet also used 2017 as extra data; see `WEIGHTS.md` for the class mapping). Mean of three seeds:

| | mIoU | IoU | mcIoU |
|---|---|---|---|
| **This pipeline** (revisit, frame decode with soft duplicate penalties) | **94.37 ± 0.69** | **90.31 ± 0.75** | **57.80 ± 3.43** |
| seed 42 alone | 95.16 | 91.13 | 61.75 |
| MATIS, released model, masks given | 89.28 | 84.42 | 52.70 |
| TAPIS (retrained on EndoVis 2018), masks given | 84.58 | 80.72 | 44.63 |

Per-class IoU (three-seed mean): Bipolar Forceps 97.8, Prograsp Forceps 55.7, Large Needle Driver 70.1, Monopolar Curved Scissors 96.5, Suction Instrument
66.2, Clip Applier 16.3, Ultrasound Probe 1.8. How to read it: most of the gain over members trained on 2018 alone (88.76 / 83.95 / 51.52 at seed 42) is the
extra training data, which took the Large Needle Driver from 175 to 3,581 training instances (single-pass recall 14% to 76%); the revisit and decode add about
4 to 5 mIoU on top in both setups. The baselines were not given the 2017 data, so this is not a like-for-like comparison. Clip Applier has only 15 training
instances and scores 49.0, 0.0 and 0.0 on the three seeds, which is why the mcIoU spread is large. The Ultrasound Probe appears in one training sequence and is
essentially not recognised.

**What the numbers do and do not show.**
- The prior shift (-0.5) was picked on the two validation folds; a wider grid put the best at -0.75, and on test the two differ by less than the seed spread. The
  test split was read for each setting that was scored, so treat the GraSP test row as a development-informed number, not a blind one.
- The EndoVis decode first used duplicate penalties estimated from the 2018 frames alone and was worse (93.81 / 88.59 / 58.24 at seed 42); the penalties are now
  estimated from the frames the members were trained on, which is the stated rule, but that correction was made after seeing the first result.
- Ideas that were tried and did not help are listed in `docs/REPRODUCING_THE_PAPER.md` (a second, wider revisit window was worse on test).

## Earlier results: SAM-predicted masks, no frame decode

GraSP official test set, 2,861 instruments in 1,125 frames from 5 cases, **with ground-truth boxes as input** (TAPIS and ISINet find their own boxes, so
this is an oracle-box comparison, an upper bound for a pipeline fed by a detector). Segmentation metrics are the benchmark's: mIoU, IoU, mcIoU, computed
with code ported from the MATIS evaluation (`grasp_pipeline/scoring.py`). Final configuration: all-case segmenters, tracker-style-crop classifier members,
the 833 most uncertain instruments of 2,861 tracked, three classifier seeds.

| | mIoU | IoU | mcIoU |
|---|---|---|---|
| **This pipeline, seed 44** (the seed picked in advance on a held-out fold) | 87.49 | 86.41 | 79.78 |
| This pipeline, three-seed mean ± SD | 87.37 ± 0.33 | 86.22 ± 0.46 | 78.33 ± 1.28 |
| 95% interval, bootstrap over the 5 cases | 86.85 to 88.21 | 85.56 to 87.09 | 76.33 to 79.81 |
| TAPIS (Swin-L Mask2Former + video transformer), published | 86.61 | 83.38 | 77.42 |
| TAPIS-VST, published | 86.36 | 83.51 | 77.54 |

How to read it: IoU is clearly above TAPIS. mIoU is modestly above. mcIoU is not distinguishable (TAPIS lies inside our interval). The IoU margin is the
one most inflated by oracle boxes, because that metric counts spurious classes against a method and ground-truth boxes produce none. Instance-level
accuracy is about 0.957 and macro-F1 about 0.929 (three-seed mean). With perfect classes on these masks the three scores would be 91.17, 91.17 and 88.95:
the segmentation ceiling.

Each step of the pipeline, three-seed means, same frames (differences of 0.3 are inside the noise; the paired intervals are in
`docs/reports/gtbox_sam/bootstrap_cis.json` of the research repository):

| step | mIoU | IoU | mcIoU |
|---|---|---|---|
| SAM2 masks, baseline classifier members | 86.34 | 85.31 | 77.59 |
| + SAM3 in the segmenter ensemble | 86.78 | 85.72 | 77.90 |
| + classifier trained with tracker-style crops | 87.11 | 85.97 | 78.19 |
| + segmenters trained on all 8 training cases (final) | 87.37 | 86.22 | 78.33 |

Per-class IoU of the final configuration: Bipolar Forceps 84.0, Prograsp Forceps 67.5, Large Needle Driver 87.0, Monopolar Curved Scissors 94.2,
Suction Instrument 78.1, Clip Applier 76.8, Laparoscopic Grasper 60.7. The Grasper is the weak class: with its jaws closed it is a featureless shaft,
visually the same as a suction tube.

## The parts

**Segmenter.** SAM2.1-large and SAM3, both fine-tuned on GraSP with ground-truth boxes as prompts, each run on the image and its mirror image, the mask
logits averaged and then averaged across the two models. Fine-tuning changed only some tensors, so the weights ship as deltas over the base models:
for SAM2 the decoder, prompt encoder, neck and last four encoder blocks (267 MB); for SAM3 only the mask decoder and prompt encoder (15 MB; an
encoder-training option in the training script never reached the weights, see the protocol document's correction). Mean mask IoU against the ground-truth
masks on the test set is 0.908.

**Classifier.** Four members, one deterministic pass each: ResNet-50 at 320 px, ResNet-50 at 224 px, MobileNetV3-small with a stretched crop, MobileNetV3-small
with a letterboxed crop; weights 0.40, 0.20, 0.20, 0.20. A member's 7 logits are read as Dirichlet evidence (exp of the clamped logit plus one); the
ensemble's parameters are the weighted mean, the prediction is the largest mean belief, and the uncertainty is the epistemic score
`(1 - sum(mu^2)) / (alpha0 + 1)` of Duan et al. (WACV 2024). No softmax anywhere. Members were trained on the official training cases with a
tracker-style-crop augmentation: half the time a crop is swapped for a stored crop of the same instrument a few frames away, as a tracker would produce.
The 0.40 weight was set in the first weeks on the official test set; a later check on held-out folds did not confirm it over flat weights, and it was kept
by decision (see `WEIGHTS.md`). Treat it as a default, not a validated optimum.

**Gate.** An instance whose score reaches `epistemic_gate` (`evidential_config.yaml`) is sent to tracking. 1.7e-5 was chosen on a held-out fold for the
baseline members; these members are less confident and it flags about 42% of the test instruments (seed 44). The reported results tracked a fixed share
instead, the 833 most uncertain of 2,861 (29%), which corresponds to a threshold of 1.6e-4 for seed 44 on that run, a value derived from the test
instances. Raise the gate to track fewer.

**Tracker.** SAM2-large video propagation of the flagged instrument's mask over the 10 frames on each side (about 20 seconds in total at the intended
sampling), the classifier on every propagated crop, and a confidence-weighted fusion of the frames. **It uses future frames, so it is not causal**
(`--causal` propagates backwards only; the cost of doing so was not measured with this tracker). TAPIS is non-causal as well: its 16-frame clip spans about
8 seconds around the keyframe.

**Frame decode.** After tracking, the instruments of a frame are labelled together instead of one at a time (`grasp_classifier/decode.py`). The labelling with
the largest total log-belief wins, subject to the fact that a class rarely appears twice in a frame: a second instrument of an already used class pays a
penalty. On GraSP only the Large Needle Driver repeats (414 of the 421 training frames that repeat a class), so the rule is hard (every other class may appear
at most once), and a prior shift of -0.5 (favouring frequent classes) removes rare-class false alarms. On EndoVis several classes repeat, so each class gets a
soft penalty estimated from the training frames. The search is exact and cheap (a dynamic program over the set of used classes). It needs no training and no
tuned parameter other than the GraSP shift; it added 0.8 mIoU, 0.8 IoU and 1.1 mcIoU on the GraSP test set. Switch it off with `--no-decode`.

## Cost

Measured on one Titan Xp (Pascal, no fast half precision), one frame at a time, 2.5 instruments per frame on average.

| stage | time |
|---|---|
| SAM2 masks, single pass | 0.53 s per frame (1.05 s with the mirror pass) |
| SAM3 masks, single pass | 1.16 s per frame (2.38 s with the mirror pass) |
| four-member classifier | 0.10 s per frame |
| SAM2-large tracking | about 13 s per tracked instrument (21 frames) |
| TAPIS, for reference | 0.48 s per keyframe |

So the pipeline is far slower than TAPIS: about 3.5 s per frame for the ensemble segmenter before any tracking, and tracking dominates. It is an offline,
batch tool. Newer GPUs with bf16 hardware should be several times faster; that has not been measured here. A run of the command-line tool on 4 frames and
11 instruments, model loading included, took 69 s on the same GPU.

## Install

Tested with Python 3.11, PyTorch 2.8 (CUDA 12.6), transformers 5.18.0 (SAM3 only) and SAM2 at commit `2b90b9f`. `requirements-lock.txt` is the exact
environment the results were produced in (it names the CUDA 12.6 PyTorch build; install PyTorch from its own index).

```bash
pip install -r requirements.txt
pip install -r requirements-pipeline.txt        # transformers for SAM3, huggingface_hub
pip install "git+https://github.com/facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4"
```

`python -m grasp_pipeline.weights` lists every weight file with its size, checksum state and role. `weights_manifest.json` is the source of truth. The SAM2.1
base checkpoint has Meta's public URL. The GraSP weights are hosted at https://huggingface.co/AryanB005/grasp-instrument-pipeline, with the manifest URLs
pinned to a commit. `--fetch` downloads and verifies all of them (about 2.7 GB, the default run needs about 1.4 GB of it; use `--only` to restrict).

## Tests

`pytest tests/test_evidential.py tests/test_pipeline.py tests/test_manifest.py tests/test_decode.py tests/test_reproduce.py` runs without any weights (synthetic checkpoints; the segmenter and tracker are
replaced by stand-ins). The older tests load the shipped vote-pipeline checkpoints and are marked `slow`. Against the research pipeline with the real weights:
classifier logits on 60 test instruments agree to 7e-5 with identical predictions, segmenter masks on 27 instruments are identical, and the tracker's fused
class agrees on 6 of 6 tracked instruments.

## Limits

- Boxes in, so detection quality is outside these numbers. A detector's boxes will cost accuracy; how much is not measured here.
- GraSP only: 7 classes, one dataset, five test cases. The 0.40 member weight and the tracked share of 29% were set with test-set exposure; the
  threshold-based gate (held-out) is the cleaner setting.
- The segmenters were each trained once; the seed spread above covers the classifier only.
- Clipping masks to ground-truth boxes (`--clip-to-box`) adds about 0.2 mIoU and works only because ground-truth boxes are tight bounds of the true masks.
- SAM3's licence and gating are Meta's; SAM2 is Apache 2.0. The GraSP dataset has its own terms. This repository has no licence file yet.

## More

- `docs/REPRODUCING_THE_PAPER.md`: how each number of the paper is produced, from a no-GPU check to the full pipeline, and what was tried and rejected.
- `WEIGHTS.md`: every weight file, how it was trained, the seeds, and what was and was not validated.
- `docs/LEGACY_VOTE_PIPELINE.md`: the earlier classifier-only package with MC-dropout votes (still importable as `grasp_classifier.EnsembleClassifier`).
- Research repository (training code, protocol, results, decision log): `volcanictv/grasp-lightweight-instrument-recognition`, master at the merge of
  PR #7. The protocol of this configuration is `docs/reports/gtbox_sam_protocol.md` there, written before each run it governs.
