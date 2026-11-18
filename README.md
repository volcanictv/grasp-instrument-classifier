# Recognize, then Revisit

Code and weights for **Recognize, then Revisit: Improving Surgical Instrument Segmentation with Uncertainty-Guided Tracking**.

## Abstract

Instrument-type segmentation in endoscopic video requires accurate instance masks and the correct identity of every instrument, and identity is the
bottleneck. Many instruments share similar shafts and differ mainly in small tip or jaw cues that are often occluded, closed, or outside the field of view,
making instrument recognition challenging. We study recognition in isolation: given an RGB frame and the ground-truth instance masks, assign a class to each
instrument. Our method classifies each masked instrument with an ensemble of evidential CNNs that outputs class probabilities and an uncertainty score. Only
uncertain instruments are revisited: their mask is propagated to neighboring frames within ±10 s with a video segmentation model, each view is re-classified,
and the predictions are fused with confidence weights, while confident instruments keep their class.

## Install

Python 3.11, PyTorch 2.8 (CUDA 12.6), SAM2 at commit `2b90b9f`. `requirements-lock.txt` is the exact environment used.

```bash
git clone https://github.com/volcanictv/grasp-instrument-classifier && cd grasp-instrument-classifier
pip install -r requirements.txt -r requirements-pipeline.txt
pip install "git+https://github.com/facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4"
```

## Weights

```bash
python -m grasp_pipeline.weights                       # lists every weight file with its size and checksum state
python -m grasp_pipeline.weights --fetch --only evidential_armN_seed44 sam2.1_hiera_large     # GraSP classifier members and the SAM2 base
python -m grasp_pipeline.weights --fetch --only evidential_endovis18_2017_seed42              # EndoVis 2018 members
```

`--only` matches substrings of the file path. Every download is checked against `weights_manifest.json`.

## Use it with given masks (GraSP)

```bash
python -m grasp_pipeline.gt_inputs --coco annotations/grasp_short-term_test.json --out inputs/       # boxes and masks per case from the annotations
python -m grasp_pipeline.run --frames <GraSP>/frames-001/frames/CASE041 --boxes inputs/CASE041/boxes.json --masks-dir inputs/CASE041/masks --out out/CASE041
```

Run it once per case. Frames are read in sorted order and are assumed to be about one second apart. For every frame the output has `<frame>.json` (per
instrument: box, class, uncertainty, whether it was revisited, the single-pass class, the belief over the classes) and `<frame>.png` (the semantic map: 0 is
background, 1 to 7 are the classes). `--masks-dir` holds one `<frame stem>.npz` per frame with an array `masks` of shape (instruments, H, W), in the order of that
frame's boxes. For EndoVis 2018 pass `--config evidential_config_endovis18.yaml` with your own boxes and masks.

## Use it with boxes only

```bash
python -m grasp_pipeline.weights --fetch --only evidential_armN_seed44 sam2.1_hiera_large sam2_delta.pt sam3_delta.pt
python -m grasp_pipeline.run --frames path/to/frames --boxes boxes.json --out out/ --device cuda
```

`boxes.json` maps a frame file name to its boxes, each `[x, y, w, h]` in native pixels (`examples/boxes_example.json`):

```json
{"00001.jpg": [[120, 80, 210, 160], [400, 300, 150, 220]]}
```

SAM3's base weights are a gated model on Hugging Face (`facebook/sam3`): accept the licence and run `huggingface-cli login` once, or pass `--no-sam3`.

## Use it from Python

```python
from PIL import Image
import numpy as np
from grasp_classifier.decode import FrameDecoder
from grasp_classifier.evidential import EvidentialEnsembleClassifier
from grasp_pipeline.track import EvidentialTracker
from grasp_pipeline.pipeline import InstrumentPipeline, semantic_map

W = "weights"
classifier = EvidentialEnsembleClassifier("evidential_config.yaml", device="cuda:0")
tracker = EvidentialTracker(classifier, f"{W}/sam2.1_hiera_large.pt", device="cuda:0")      # optional: the revisit
decoder = FrameDecoder.from_stats("decode_stats/grasp_official_train.json")                 # optional: labels the instruments of a frame jointly
pipeline = InstrumentPipeline(segmenter=None, classifier=classifier, tracker=tracker, decoder=decoder)

frames = [np.array(Image.open(p).convert("RGB")) for p in paths]                            # consecutive frames, about 1 per second
results = pipeline.process_frame(frames[i], boxes_xywh, context_frames=frames, center_idx=i, masks=given_masks)
for r in results:
    print(r.class_name, r.epistemic, r.tracked, r.single_pass_class)
sem = semantic_map(frames[i].shape[:2], results)                                            # per-pixel class map
```

`examples/pipeline_example.py` is a runnable version that also segments from boxes. The GraSP classes, in the order of every output: Bipolar Forceps,
Prograsp Forceps, Large Needle Driver, Monopolar Curved Scissors, Suction Instrument, Clip Applier, Laparoscopic Grasper.

## Options of `grasp_pipeline.run`

| option | effect |
|---|---|
| `--masks-dir DIR` | use given masks, no segmenter |
| `--no-tracking` | single pass only, no revisit |
| `--causal` | revisit with past frames only |
| `--window N` | frames on each side used for the revisit (default 10) |
| `--gate X` | uncertainty threshold above which an instrument is revisited |
| `--no-decode` | take the largest belief per instrument instead of the joint frame labelling |
| `--no-sam3` | segment from boxes with SAM2 alone |
| `--config FILE` | classifier configuration (`evidential_config.yaml`, or `evidential_config_endovis18.yaml`) |

## Documentation

- `docs/REPRODUCING_THE_PAPER.md`: how the paper's numbers are produced, and a check that runs without a GPU (`python -m grasp_pipeline.reproduce_readout`).
- `WEIGHTS.md`: every weight file and how it was trained.
- Tests: `pytest tests/test_evidential.py tests/test_pipeline.py tests/test_manifest.py tests/test_decode.py tests/test_gt_inputs.py tests/test_reproduce.py` (no weights needed).
