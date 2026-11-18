# Weights

`python -m grasp_pipeline.weights` prints every file below with its size and checksum state; `--fetch` downloads the ones that have a URL. The source of
truth is `weights_manifest.json` (path, bytes, sha256, role, url). The GraSP files are hosted at https://huggingface.co/AryanB005/grasp-instrument-pipeline with
URLs pinned to a commit; the SAM2.1 base uses Meta's URL. A file with a null `url` would have to be copied into `weights/` by hand; the checksum is
verified either way.

## Files

| what | notes |
|---|---|
| SAM2.1 Hiera-large base checkpoint | Meta, public URL, Apache 2.0 |
| SAM2 delta (decoder, prompt encoder, neck, last four encoder blocks), 267 MB | applied over the base at load time |
| SAM3 delta (mask decoder and prompt encoder), 15 MB | applied over `facebook/sam3` (gated, Hugging Face) |
| `weights/evidential_armN_seed{42,43,44}/`, four classifier members per seed | seed 44 is the default in `evidential_config.yaml` |

The manifest also lists two sets that the default run does not use, for reproducing the research ladder: `weights/evidential_baseline_seed{42,43,44}/`
(members trained without the tracker-style-crop augmentation) and `weights/sam{2,3}_delta_fold2.pt` (the segmenters of the earlier ladder rungs, trained on the
fold2 cases only; the manifest role string is authoritative). Exact file names, sizes and checksums are in `weights_manifest.json`.

The segmenter deltas were trained on all 8 GraSP training cases with ground-truth boxes as prompts. Fine-tuning was run once per model (no seed
repetition). The SAM3 encoder was never trained: an option for it existed in the training script but `get_image_embeddings` is `no_grad`, so only the
mask decoder changed (the research repository's protocol document records this as a correction).

## Classifier members

| file | architecture | input | crop |
|---|---|---|---|
| `resnet50_320.pt` | ResNet-50 | 320 px | letterboxed |
| `resnet50_224.pt` | ResNet-50 | 224 px | letterboxed |
| `mobilenet_baseline.pt` | MobileNetV3-small | 224 px | stretched |
| `mobilenet_letterbox.pt` | MobileNetV3-small | 224 px | letterboxed |

Trained on the official GraSP training cases with the tracker-style-crop augmentation. Logits are read as Dirichlet evidence,
`alpha = exp(clip(logit, -10, 10)) + 1`.

Seed 44 is the registered default because it had the best fold-1 accuracy of the three-member ensembles, a rule fixed before the test run. Seeds 42 and
43 are included so the seed spread can be reproduced; the three-seed mean is in the README.

## The 0.40 member weight: what was and was not validated

`weight_resnet50_320: 0.40` (the other three members 0.20 each) was set in the first weeks of the project on the official test set, for the earlier
vote-based package. Every result reported for the evidential pipeline used it. A later check on held-out folds did not confirm it over flat weights
(0.25 each). The flat-weights version of the final configuration was not run on the test set, so how much the 0.40 matters here is not measured. To try
flat weights, set `weight_resnet50_320: 0.25` in `evidential_config.yaml`; no code change is needed.

## Which weights are behind the paper's numbers

**GraSP.** The paper's GraSP rows (three seeds, ground-truth masks) use `weights/evidential_armN_seed{42,43,44}/`. Their sha256 equals that of the
research checkpoints the numbers were produced with (`configs/arms/ens4_N_official_s{42,43,44}.yaml` of the research repository, 12 of 12 files compared
on 2026-11-18), so the hosted files are the same bytes. The segmenter deltas are not involved: the paper's setting gives the masks.

**EndoVis 2018.** The EndoVis rows use four members per seed (3 seeds) trained on the EndoVis 2018 ISINet-split training sequences plus the complete
EndoVis 2017 (2017 types 1, 2, 3, 6 keep their class, type 7 "Other" is treated as Ultrasound Probe, vessel sealer and grasping retractor are dropped;
2017 crops have no tracker-style neighbours). Recipe: `scripts/endovis_gtbox_classifiers_grasp_e17.py` of the research repository (the GraSP recipe, 20
epochs). The 12 files (604 MB) are hosted in the same Hugging Face repository as the GraSP weights, under
`weights/evidential_endovis18_2017_seed{42,43,44}/` with the same file names as the GraSP members (commit `be03603`, pinned in `weights_manifest.json`;
sizes and sha256 of every file were checked against the stored copies after the upload). `evidential_config_endovis18.yaml` points at seed 42; fetch with
`python -m grasp_pipeline.weights --fetch --only evidential_endovis18_2017_seed42`. A parity check of the shipped inference path (class set, crop code, member
loading) against the research logits of seed 42 is in `docs/REPRODUCING_THE_PAPER.md`.

## Legacy vote-pipeline weights

The earlier MC-dropout package (`weights/*_dropout.pt`, `ensemble_config.yaml`, `grasp_classifier.EnsembleClassifier`) is unchanged and documented in
`docs/LEGACY_WEIGHTS.md` and `docs/LEGACY_VOTE_PIPELINE.md`.
