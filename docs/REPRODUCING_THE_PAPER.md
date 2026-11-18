# Reproducing the paper's numbers

The paper studies **recognition with given masks**: every method receives the ground-truth instance masks and assigns a class to each. The headline tables
are in the README ("Results of the paper"). This document says how each number is produced, at three levels of effort, what was checked when this
package was updated (2026-11-18), what was tried and rejected, and what is still missing.

## What the method is

Per instrument: four evidential members (ResNet-50 at 320 and 224 px, two MobileNetV3-small) read as Dirichlet evidence, mixed with weights
0.4 / 0.2 / 0.2 / 0.2, giving a mean belief and the epistemic score `S1`. Instruments with `S1 >= 1.7e-5` are tracked by SAM2.1-large from their
mask over the 10 frames on each side (1 Hz) and take the confidence-weighted fused belief of the track. Then the **frame decode**
(`grasp_classifier/decode.py`): the instruments of one frame are labelled jointly by the largest total log-belief, where a second instrument of an
already used class pays a penalty, and an optional prior shift is added.

| | duplicate rule | prior shift | statistics file |
|---|---|---|---|
| GraSP | hard: only the Large Needle Driver may repeat in a frame (414 of 421 training frames that repeat a class) | -0.5 | `decode_stats/grasp_official_train.json` |
| EndoVis 2018 | soft: penalty `log((frames with the class twice + 1) / (frames with the class + 2))` per class | none | `decode_stats/endovis18_train_plus_2017.json` |

Both files are computed from training frames only. Regenerate the GraSP one with
`python -m grasp_pipeline.decode_stats --coco grasp_short-term_train.json --class-set grasp --rule hard --allow "Large Needle Driver" --prior-shift -0.5 --out ...`
(checked on 2026-11-18: 2,324 frames, repeated-class frames BF 3, PF 4, LND 414, the numbers the research code gave).

## Level 0: no GPU, no data (seconds)

```bash
python -m grasp_pipeline.reproduce_readout
```

Recomputes instance accuracy and macro-F1 of the readouts from `reproduce/*.npz` (fused class beliefs after the first revisit, labels, and an integer frame
id per instrument, for three classifier seeds; no file names, boxes or masks) using this package's decoder, and compares with `reproduce/expected.json`.
`pytest tests/test_reproduce.py` runs the same check. Expected (percent):

| | readout | seed 42 | 43 | 44 | mean |
|---|---|---|---|---|---|
| GraSP test accuracy | argmax | 96.12 | 96.40 | 96.43 | 96.32 |
| | decode + shift -0.5 | 96.82 | 96.89 | 96.96 | 96.89 |
| EndoVis validation accuracy | argmax | 89.52 | 89.10 | 87.23 | 88.62 |
| | decode (soft) | 89.59 | 89.04 | 88.34 | 88.99 |

These agree with what the research scripts printed independently (`scripts/frame_decode_eval.py`, `scripts/endovis_score_decode.py`).

## Level 1: from model outputs to the paper's IoUs (CPU, needs the datasets' masks)

mIoU, IoU and mcIoU need ground-truth masks, which cannot be shipped. Given the saved member logits and track logits of a run:

- GraSP: `scripts/frame_decode_eval.py --split test --masks masks.pkl --logits logits_s<seed>.npz --tracked-dir <tracked dir> --shifts -0.5 --out ...` in the
  research repository (it prints argmax, decode and decode + shift rows with mIoU / IoU / mcIoU and the fixed / broken counts).
- EndoVis: `scripts/endovis_score_decode.py --instances instances.pkl --single single_ens_s<seed>.npz --tracked tracked_s<seed>.npz --stats decode_stats/endovis18_train_plus_2017.json --out ...`.
  For seed 42 this gives 95.16 / 91.13 / 61.75 (checked on 2026-11-18).

Three-seed means and SDs of the README tables are the means and sample SDs of the three per-seed outputs.

## Level 2: with the shipped weights, end to end

GraSP, per test case (frames are tracked within a case):

```bash
python -m grasp_pipeline.gt_inputs --coco annotations/grasp_short-term_test.json --out inputs/
python -m grasp_pipeline.run --frames <GraSP>/frames-001/frames/CASE041 --boxes inputs/CASE041/boxes.json --masks-dir inputs/CASE041/masks \
    --out out/CASE041 --config evidential_config.yaml --device cuda        # repeat per case; the paper uses seeds 42, 43, 44 (edit the checkpoint paths)
```

`--masks-dir` skips the segmenter; the decode is on by default with `decode_stats/grasp_official_train.json`. The semantic maps in `out/` are scored with
`grasp_pipeline/scoring.py` against the annotation maps. **The command-line path was not re-run on the test set for this update**; its parts were checked
separately (below). The EndoVis members are hosted with the others (`WEIGHTS.md`; `python -m grasp_pipeline.weights --fetch --only evidential_endovis18_2017_seed42`),
and `evidential_config_endovis18.yaml` selects them and the EndoVis class set; the EndoVis inputs (annotation masks as `--masks-dir`) come from the ISINet
annotation package, which has no converter in this repository.

## Training the members and regenerating the outputs (research repository)

GraSP members (`weights/evidential_armN_seed*`): `scripts/run_training_arms.py` with the arm N configs (`scripts/make_armN_ensemble_configs.py`;
tracker-style crops from `scripts/build_temporal_neighbors.py`). Test-split outputs with ground-truth masks: `scripts/gt_masks_to_pkl.py`,
`scripts/extract_logits_from_masks.py`, SAM2 tracking by `scripts/evaluate_temporal_track_ensemble.py` (window 10, driver pattern in
`scripts/run_gtfold_gtmask.sh`), `scripts/reclassify_tracked_masks.py` to re-classify the tracks with the other seeds' members.

EndoVis members: neighbour crops by `scripts/endovis_gtbox_neighbours.py` (EdgeTAM, k = 5), members by
`scripts/endovis_gtbox_classifiers_grasp_e17.py` (the GraSP recipe plus the 2017 crops), then logits, tracking and the track logits as in
`scripts/run_endovis_gtmask.sh`. **At the time of writing the EndoVis scripts `endovis_gtbox_*.py`, `run_endovis_gtbox*.sh` and the GraSP test-split driver
were untracked or only on the lab workstation, not committed to the research repository; commit them before sharing, otherwise this level cannot be followed.**

## Protocol and selection history (read before citing)

- GraSP: the prior shift -0.5 was chosen on the two validation folds (the best of a coarse grid, later a wider grid put the best at -0.75), but the test split
  was scored for every setting tried, so the test row is development-informed, not blind. -0.75 gives 96.63 / 95.55 / 89.62 on test, within the seed spread.
- EndoVis: the members use extra training data (the complete EndoVis 2017; ISINet did the same). 2017 type 7 "Other" is treated as Ultrasound Probe, which was
  judged visually only. The decode first used penalties from the 2018 frames alone and was worse (93.81 / 88.59 / 58.24 at seed 42); the penalties are now
  estimated from the frames the members were trained on, a correction made after seeing that result. The baselines were not trained on 2017.
- Baselines (TAPIS, SlowFast, MATIS, ISINet) with given masks are single runs of released or re-run models, matched to the given masks by overlap; their seed
  spread is unknown, ours is over classifier seeds only. The GraSP validation row is one seed.
- Clip Applier (15 EndoVis training instances) is unstable across seeds (IoU 49.0, 0.0, 0.0), so EndoVis mcIoU has a large spread (57.80 +- 3.43).

## Tried and rejected (all on GraSP unless stated; development folds first)

| idea | result |
|---|---|
| second revisit of still-uncertain instruments over +-20 s | dev folds +0.22 accuracy (narrow pass), **test worse on every seed**: 96.46 / 95.39 / 89.32 vs 96.72 / 95.69 / 90.24 |
| SAM3 instead of SAM2.1 as the tracker | no consistent gain over three seeds (predicted-mask setting) |
| other fusion rules (plain mean, evidence sum, log mean, max pool) | within noise or worse; evidence for a missed Laparoscopic Grasper is mostly absent from its frames |
| soft-context crop instead of the isolated crop | three seeds x two folds: +0.3 accuracy, +0.1 F1 (noise; one seed had looked good) |
| strong augmentation | -0.7 accuracy, -0.7 F1 |
| per-case class-prior adaptation (EM) | worse on one fold, flat on the other |
| class-set prior over a whole frame | accuracy +0.1, macro-F1 -0.4 |
| instrument position likelihood | +0.02 accuracy on the two-fold mean, worse at full weight |
| EndoVis Ultrasound Probe: logit adjustment, size likelihood | no gain; the probe is in one training sequence and looks like a plain shaft |

## What was checked on 2026-11-18

- `pytest tests/test_decode.py tests/test_gt_inputs.py tests/test_reproduce.py` plus the existing weight-free tests: 45 passed. The decoder's dynamic program was
  compared with brute-force enumeration on random frames for the hard and soft rules and several prior shifts.
- Checksums: the 12 GraSP checkpoints of `weights_manifest.json` equal the research checkpoints behind the GraSP numbers.
- Hosting: the 12 EndoVis members were uploaded to the Hugging Face repository in one commit (`be03603`). The stored sha256 and size of every file match the
  originals, all 12 manifest URLs resolve with the right size, and `python -m grasp_pipeline.weights --fetch` downloaded two of them and verified them.
- Parity: this package's inference path (class set, crops, member loading) against the research logits of the EndoVis seed 42 members on 48 validation
  instruments: max |logit difference| 1.1e-4 (CPU vs the original GPU run), same predicted class on 48 of 48.
- The two statistics files and the `reproduce/` arrays were produced by this package's own code (GraSP) or from the research outputs (EndoVis, below).

## Still missing

- The EndoVis penalties were derived from the research cache (2018 frames) and the 2017 per-frame type masks with scripts that are not in this package; the
  merged counts are in the statistics file, and `python -m grasp_pipeline.decode_stats --frames-json` regenerates the 2018 part from a list of per-frame
  class lists.
- GraSP validation rows are one seed; the end-to-end command-line run was not repeated on the test set.
