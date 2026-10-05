# Weights

`python -m grasp_pipeline.weights` prints every file below with its size and checksum state; `--fetch` downloads the ones that have a URL. The source of
truth is `weights_manifest.json` (path, bytes, sha256, role, url). A `url` of null means the file is not published yet and must be copied into
`weights/` by hand; the checksum is still verified.

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

## Legacy vote-pipeline weights

The earlier MC-dropout package (`weights/*_dropout.pt`, `ensemble_config.yaml`, `grasp_classifier.EnsembleClassifier`) is unchanged and documented in
`docs/LEGACY_WEIGHTS.md` and `docs/LEGACY_VOTE_PIPELINE.md`.
