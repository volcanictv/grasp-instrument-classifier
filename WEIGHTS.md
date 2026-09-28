# Ensemble weight: kept at 0.40, decided 2026-09-28

`ensemble_config.yaml` sets `weight_resnet50_320: 0.40` (the other 3
members split the remaining 0.60 equally, 0.20 each). This is the
**confirmed shipped default**, not an unreviewed carryover: it was
re-examined against the alternative below and kept deliberately.

Vote-based prediction (weighted plurality of the stochastic votes), measured
on GraSP's official test set with oracle boxes and masks:

| configuration | accuracy | macro-F1 | share of instances at uncertainty >= 0.09 |
|---|---|---|---|
| weighted (0.40 / 0.20 / 0.20 / 0.20), **current default** | 0.9266 | 0.8898 | 29.1% |
| flat (0.25 each) | 0.9238 | 0.8943 | 29.7% |

The weighted setting scores higher on accuracy, the flat one on macro-F1;
the gaps are small (0.003 and 0.005) and mixed, not a clear win either way.
An earlier ensemble's held-out check on a fresh split favoured flat
weighting slightly; that check was not repeated for these retrained
checkpoints. Decision: keep 0.40. Every result reported for this pipeline
(the tracking-gate confirmation, the final accuracy/macro-F1 numbers, the
EndoVis generalization check, and the uncertainty-method comparisons) was
computed with 0.40, and re-deriving all of it to chase a gap this small was
judged not worth it. To use the flat configuration anyway, change
`weight_resnet50_320` to `0.25` in `ensemble_config.yaml`; no code change is
needed.

# Checkpoints

The four files in `weights/` (about 200 MB together) are the retrained
members, each trained with dropout inside the network (Dropout(0.2) before
the classifier plus channel-wise Dropout2d(0.1) after two mid/late stages),
on GraSP's official training cases:

| file | architecture | input | crop |
|---|---|---|---|
| `resnet50_320_dropout.pt` | ResNet-50 | 320 px | letterboxed |
| `resnet50_224_dropout.pt` | ResNet-50 | 224 px | letterboxed |
| `mobilenet_baseline_dropout.pt` | MobileNetV3-small | 224 px | stretched |
| `mobilenet_letterbox_dropout.pt` | MobileNetV3-small | 224 px | letterboxed |

They are not compatible with the earlier softmax-averaging ensemble's
checkpoints (different layer structure), and the reverse. Checkpoint
selection for these members used the official test set as validation, as it
did for the earlier ones, so the test numbers in the README are not
independent of that choice.
