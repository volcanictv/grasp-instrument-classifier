# Recognize, then Revisit

Code, weights and reproduction for the paper **Recognize, then Revisit: Improving Surgical Instrument Segmentation with Uncertainty-Guided Tracking**
(in preparation; GraSP and EndoVis 2018).

Instrument-type segmentation needs accurate masks *and* the right identity for every instrument, and identity is the bottleneck: many instruments share
shafts and differ only in small tip or jaw cues that are often occluded, closed or out of view. The paper therefore studies **recognition with given masks**:
the input is an RGB frame and the instance masks of its instruments, the output is a class for each mask, and the masks are never modified. The method
classifies every masked instrument with an ensemble of evidential CNNs (class probabilities plus an uncertainty score), **revisits only the uncertain
instruments** in neighbouring video frames, and finally labels the instruments of a frame together.

```
frame + given masks ──► crop each instrument ──► evidential ensemble (4 CNNs, one pass) ──► class + uncertainty S
                                                                                              │
                          S < tau: keep the class  ◄──────────────────────────────────────────┤
                          S >= tau: SAM 2.1 propagates the mask over +-10 s ──► re-classify every view ──► confidence-weighted fusion
                                                                                              │
                          all instruments of the frame ──► frame decode ──► final classes  ◄──┘
```

Masks can come from annotation, from a segmentation model or from tracking. When only boxes are available the repository also has a box-prompted segmenter
(SAM2 and SAM3 fine-tuned on GraSP), described under "Segmentation from boxes". Two Python packages: `grasp_classifier` (classifier, evidential
readout, frame decode) and `grasp_pipeline` (segmenter, tracker, command-line runner, scoring, reproduction tools).

## Results of the paper (ground-truth masks)

Every method is given the ground-truth instance masks, so the scores depend only on the assigned classes. Metrics are the benchmark's mIoU, IoU and mcIoU
(`grasp_pipeline/scoring.py`, ported from the MATIS evaluation). The pipeline is the one in the diagram above with `epistemic_gate` (tau) = 1.7e-5, which
revisits about 38% of the GraSP test instruments. How each number is produced: `docs/REPRODUCING_THE_PAPER.md`.

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

## Quickstart: recognition with given masks (the paper's setting)

```bash
git clone https://github.com/volcanictv/grasp-instrument-classifier && cd grasp-instrument-classifier
pip install -r requirements.txt -r requirements-pipeline.txt     # see "Install" for the SAM2 package
python -m grasp_pipeline.weights --fetch --only evidential_armN_seed44 sam2.1_hiera_large   # classifier members and the SAM2 base used for the revisit
python -m grasp_pipeline.gt_inputs --coco annotations/grasp_short-term_test.json --out inputs/          # boxes and masks per case from the GraSP annotations
python -m grasp_pipeline.run --frames <GraSP>/frames-001/frames/CASE041 --boxes inputs/CASE041/boxes.json --masks-dir inputs/CASE041/masks --out out/CASE041
```

`--masks-dir` skips the segmenter, so no SAM fine-tuning weights are needed; the revisit needs only the SAM2.1 base. Frames of a case are read in sorted
order and are assumed to be about one second apart; the revisit uses the 10 frames on each side of a flagged frame. For every listed frame the output has
`<frame>.json` (per instrument: box, class, uncertainty S, whether it was revisited, the single-pass class, whether the frame decode changed the class, the
belief over the classes) and `<frame>.png` (the semantic map: 0 background, 1 to 7 the classes in the classifier's class order).

For EndoVis use `evidential_config_endovis18.yaml` (its class order has an Ultrasound Probe and no Laparoscopic Grasper); its members are hosted with the
others (`python -m grasp_pipeline.weights --fetch --only evidential_endovis18_2017_seed42`).

From Python (`examples/pipeline_example.py` is a runnable version with the segmenter):

```python
from PIL import Image
import numpy as np
from grasp_classifier.decode import FrameDecoder
from grasp_classifier.evidential import EvidentialEnsembleClassifier
from grasp_pipeline.track import EvidentialTracker
from grasp_pipeline.pipeline import InstrumentPipeline, semantic_map

W = "weights"
classifier = EvidentialEnsembleClassifier("evidential_config.yaml", device="cuda:0")
tracker = EvidentialTracker(classifier, f"{W}/sam2.1_hiera_large.pt", device="cuda:0")
decoder = FrameDecoder.from_stats("decode_stats/grasp_official_train.json")       # the frame decode; leave out for the largest belief per instrument
pipeline = InstrumentPipeline(segmenter=None, classifier=classifier, tracker=tracker, decoder=decoder)

frames = [np.array(Image.open(p).convert("RGB")) for p in paths]                   # consecutive frames, about 1 per second
results = pipeline.process_frame(frames[i], boxes_xywh, context_frames=frames, center_idx=i, masks=given_masks)
for r in results:
    print(r.class_name, r.epistemic, r.tracked, r.single_pass_class, r.decoded)
sem = semantic_map(frames[i].shape[:2], results)                                   # per-pixel classes, what mIoU is computed on
```

Classes of the GraSP members, in the fixed order of every output: Bipolar Forceps, Prograsp Forceps, Large Needle Driver, Monopolar Curved Scissors,
Suction Instrument, Clip Applier, Laparoscopic Grasper.

Without any weights or data, `python -m grasp_pipeline.reproduce_readout` recomputes the instance-level readout numbers of the paper from saved class beliefs.

## The method

**Recognize.** Each instrument is cropped with its box and the pixels outside its mask are set to zero, which removes tissue and other instruments from the
crop. Four members, one deterministic pass each: ResNet-50 at 320 px, ResNet-50 at 224 px, MobileNetV3-small with a stretched crop and MobileNetV3-small
with a letterboxed crop. A member's logits are read as Dirichlet evidence (exp of the clamped logit plus one); the ensemble's parameters are the weighted
mean with weights 0.40, 0.20, 0.20, 0.20, and the prediction is the largest mean belief. No softmax anywhere. The GraSP members were trained on the official
training cases with a tracker-style-crop augmentation: half the time a crop is swapped for a stored crop of the same instrument a few frames away, as a
tracker would produce. The 0.40 weight was set in the first weeks on the official test set; a later check on held-out folds did not confirm it over flat
weights, and it was kept by decision (see `WEIGHTS.md`). Treat it as a default, not a validated optimum.

**Uncertainty.** The score is the total variance of the Dirichlet distribution (Duan et al., WACV 2024), `S = (1 - sum(P^2)) / (alpha0 + 1)`, high when the
class probabilities are spread out and the total evidence is low. An instrument is revisited when `S >= epistemic_gate` (`evidential_config.yaml`,
1.7e-5, chosen on a held-out fold); instruments below the gate keep their class from the keyframe.

**Revisit.** For a flagged instrument, the SAM2.1-large video model is initialised with its mask in the keyframe and propagates it over the 10 frames on each
side (1 Hz, about 20 seconds in total). The classifier runs on every propagated crop, and the per-frame probabilities are fused with confidence weights,
`P_final = sum_t q_t P_t / sum_t q_t` with `q_t = max_c P_t(c)`, so decisive frames count for more. **It uses future frames, so it is not causal**
(`--causal` propagates backwards only; the cost of that was not measured with this tracker). TAPIS is non-causal as well: its 16-frame clip spans about
8 seconds around the keyframe. The revisit changes only the class, never the mask.

**Frame decode.** The instruments of a frame are then labelled together instead of one at a time (`grasp_classifier/decode.py`). The labelling with the
largest total log-belief wins, subject to the fact that a class rarely appears twice in a frame: a second instrument of an already used class pays a
penalty. On GraSP only the Large Needle Driver repeats (414 of the 421 training frames that repeat a class), so the rule is hard (every other class may appear
at most once), and a prior shift of -0.5 (favouring frequent classes) removes rare-class false alarms. On EndoVis several classes repeat, so each class gets a
soft penalty estimated from the training frames. The search is exact and cheap (a dynamic program over the set of used classes). It needs no training and no
tuned parameter other than the GraSP shift; it added 0.8 mIoU, 0.8 IoU and 1.1 mcIoU on the GraSP test set. Switch it off with `--no-decode`.

## What runs, and what you can switch off

| mode | how | needs |
|---|---|---|
| given masks, no segmenter (the paper's setting) | `--masks-dir DIR` | `<frame stem>.npz` with `masks` (instruments, H, W) per frame |
| frame decode off | `--no-decode` | reads the largest belief per instrument |
| single pass, no revisit | `--no-tracking` | much faster; no SAM2 video predictor |
| causal revisit (past frames only) | `--causal` | |
| masks from boxes with SAM2 + SAM3 | default when `--masks-dir` is absent | the gated SAM3 base weights, see below |
| masks from boxes with SAM2 alone | `--no-sam3` | nothing gated |
| clip masks to their boxes | `--clip-to-box` | only meaningful for ground-truth boxes |

## Reproducing the paper

`docs/REPRODUCING_THE_PAPER.md` has three levels: a check without GPU or data (`python -m grasp_pipeline.reproduce_readout`, seconds), re-scoring saved model
outputs with the research repository's scripts, and the full run with the shipped weights. It also lists the protocol and selection history, which of the
paper's choices were made with test-set exposure, and the ideas that were tried and rejected. `WEIGHTS.md` says which weight files are behind which numbers
(the hosted GraSP checkpoints have the same sha256 as the ones the paper's numbers were produced with).

## Segmentation from boxes (earlier results, predicted masks)

When no masks are available, `BoxSegmenter` turns each box into a mask: SAM2.1-large and SAM3, both fine-tuned on GraSP with ground-truth boxes as prompts, each
run on the image and its mirror image, the mask logits averaged and then averaged across the two models. Fine-tuning changed only some tensors, so the weights
ship as deltas over the base models: for SAM2 the decoder, prompt encoder, neck and last four encoder blocks (267 MB); for SAM3 only the mask decoder and prompt
encoder (15 MB; an encoder-training option in the training script never reached the weights, see the protocol document's correction). Mean mask IoU against the
ground-truth masks on the test set is 0.908. SAM3's base weights are a gated model on Hugging Face (`facebook/sam3`): accept the licence on the model page and
run `huggingface-cli login` once, or use `--no-sam3`. The results below are for SAM2 plus SAM3; SAM2 alone with these weights has not been scored.

These numbers are **not** the paper's setting: masks are predicted from ground-truth boxes (TAPIS and ISINet find their own boxes, so this is an oracle-box
comparison, an upper bound for a pipeline fed by a detector), there is no frame decode, and the 833 most uncertain instruments of 2,861 were revisited (a fixed
share, which corresponds to a threshold derived from the test instances). Three classifier seeds, GraSP official test:

| | mIoU | IoU | mcIoU |
|---|---|---|---|
| **This pipeline, seed 44** (the seed picked in advance on a held-out fold) | 87.49 | 86.41 | 79.78 |
| This pipeline, three-seed mean ± SD | 87.37 ± 0.33 | 86.22 ± 0.46 | 78.33 ± 1.28 |
| 95% interval, bootstrap over the 5 cases | 86.85 to 88.21 | 85.56 to 87.09 | 76.33 to 79.81 |
| TAPIS (Swin-L Mask2Former + video transformer), published | 86.61 | 83.38 | 77.42 |
| TAPIS-VST, published | 86.36 | 83.51 | 77.54 |

IoU is clearly above TAPIS, mIoU modestly, and mcIoU is not distinguishable (TAPIS lies inside our interval). The IoU margin is the one most inflated by oracle
boxes, because that metric counts spurious classes against a method and ground-truth boxes produce none. With perfect classes on these masks the three scores would
be 91.17, 91.17 and 88.95: the segmentation ceiling. Each step of the pipeline, three-seed means (differences of 0.3 are inside the noise; the paired
intervals are in `docs/reports/gtbox_sam/bootstrap_cis.json` of the research repository):

| step | mIoU | IoU | mcIoU |
|---|---|---|---|
| SAM2 masks, baseline classifier members | 86.34 | 85.31 | 77.59 |
| + SAM3 in the segmenter ensemble | 86.78 | 85.72 | 77.90 |
| + classifier trained with tracker-style crops | 87.11 | 85.97 | 78.19 |
| + segmenters trained on all 8 training cases (final) | 87.37 | 86.22 | 78.33 |

Per-class IoU of the final configuration: Bipolar Forceps 84.0, Prograsp Forceps 67.5, Large Needle Driver 87.0, Monopolar Curved Scissors 94.2,
Suction Instrument 78.1, Clip Applier 76.8, Laparoscopic Grasper 60.7.

## Cost

Measured on one Titan Xp (Pascal, no fast half precision), one frame at a time, 2.5 instruments per frame on average.

| stage | time |
|---|---|
| SAM2 masks from boxes, single pass | 0.53 s per frame (1.05 s with the mirror pass) |
| SAM3 masks from boxes, single pass | 1.16 s per frame (2.38 s with the mirror pass) |
| four-member classifier | 0.10 s per frame |
| SAM2-large revisit | about 13 s per revisited instrument (21 frames) |
| TAPIS, for reference | 0.48 s per keyframe |

The classifier costs 12.5 GFLOPs per crop; a revisit costs about 25,584 GFLOPs per instrument (6.09 s on an A100), so the gate (about 38% of the GraSP test
instruments) reduces the cost but does not make revisiting cheap. The pipeline is far slower than TAPIS and is an offline, batch tool. Newer GPUs with bf16
hardware should be several times faster; that has not been measured here. A run of the command-line tool on 4 frames and 11 instruments, model loading
included, took 69 s on the Titan Xp.

## Install

Tested with Python 3.11, PyTorch 2.8 (CUDA 12.6), transformers 5.18.0 (SAM3 only) and SAM2 at commit `2b90b9f`. `requirements-lock.txt` is the exact
environment the results were produced in (it names the CUDA 12.6 PyTorch build; install PyTorch from its own index).

```bash
pip install -r requirements.txt
pip install -r requirements-pipeline.txt        # transformers for SAM3, huggingface_hub, pycocotools
pip install "git+https://github.com/facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4"
```

`python -m grasp_pipeline.weights` lists every weight file with its size, checksum state and role. `weights_manifest.json` is the source of truth. The SAM2.1
base checkpoint has Meta's public URL. The GraSP and EndoVis weights are hosted at https://huggingface.co/AryanB005/grasp-instrument-pipeline, with the manifest
URLs pinned to a commit. `--fetch` downloads and verifies the files (about 3.3 GB for everything; use `--only` to restrict, it matches substrings of the path).

## Tests

`pytest tests/test_evidential.py tests/test_pipeline.py tests/test_manifest.py tests/test_decode.py tests/test_gt_inputs.py tests/test_reproduce.py` runs without
any weights (synthetic checkpoints; the segmenter and tracker are replaced by stand-ins; the decoder is also checked against brute-force enumeration). The older
tests load the shipped vote-pipeline checkpoints and are marked `slow`. Against the research pipeline with the real weights: classifier logits on 60 GraSP test
instruments agree to 7e-5 with identical predictions, the EndoVis members agree to 1.1e-4 on 48 validation instruments, segmenter masks on 27 instruments are
identical, and the tracker's fused class agrees on 6 of 6 tracked instruments.

## Limits

- The paper's setting gives the masks, so segmentation quality is outside those numbers; with predicted masks (previous section) the scores are lower.
- Two datasets, 7 classes each, five GraSP test cases. The 0.40 member weight and the GraSP prior shift were set with test-set exposure (see `docs/REPRODUCING_THE_PAPER.md`).
- The seed spread covers the classifier only; the baselines (TAPIS, SlowFast, MATIS, ISINet) are single runs of released or re-run models, matched to the
  given masks by overlap. The GraSP validation row is one seed.
- The EndoVis rows use extra training data (EndoVis 2017) that the baselines did not get, and Clip Applier (15 training instances) and Ultrasound Probe (one
  training sequence) are unstable or essentially unrecognised there.
- The revisit is offline and non-causal, and expensive (see Cost).
- The command-line pipeline was not re-run end to end on the test set when the frame decode was added; its parts were checked separately.
- Clipping masks to ground-truth boxes (`--clip-to-box`) adds about 0.2 mIoU and works only because ground-truth boxes are tight bounds of the true masks.
- SAM3's licence and gating are Meta's; SAM2 is Apache 2.0. The GraSP and EndoVis datasets have their own terms. This repository has no licence file yet.

## More

- `docs/REPRODUCING_THE_PAPER.md`: how each number of the paper is produced, from a no-GPU check to the full pipeline, and what was tried and rejected.
- `WEIGHTS.md`: every weight file, how it was trained, the seeds, and what was and was not validated.
- `docs/LEGACY_VOTE_PIPELINE.md`: the earlier classifier-only package with MC-dropout votes (still importable as `grasp_classifier.EnsembleClassifier`).
- Research repository (training code, protocol, results, decision log): `volcanictv/grasp-lightweight-instrument-recognition`. The protocol of the
  box-prompted configuration is `docs/reports/gtbox_sam_protocol.md` there, written before each run it governs.
