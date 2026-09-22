# Ensemble weight: a disclosed, open trade-off

`ensemble_config.yaml` sets `weight_resnet50_320: 0.40` (the other 3
members split the remaining 0.60 equally, 0.20 each). This is an
**operational default carried over from the earlier ensemble, not a value
tuned or validated for these retrained checkpoints**. Both settings are
real, and neither has been checked on data outside the official test set.

Vote-based prediction (weighted plurality of the stochastic votes), measured
on GraSP's official test set with oracle boxes and masks:

| configuration | accuracy | macro-F1 | share of instances at uncertainty >= 0.09 |
|---|---|---|---|
| weighted (0.40 / 0.20 / 0.20 / 0.20), **current default** | 0.9266 | 0.8898 | 29.1% |
| flat (0.25 each) | 0.9238 | 0.8943 | 29.7% |

The weighted setting scores higher on accuracy, the flat one on macro-F1,
so which is better depends on which metric you care about; the gaps are
small (0.003 and 0.005). The earlier ensemble had a held-out check on a
fresh split that favoured flat weighting; that check was not repeated for
the retrained checkpoints. To use the flat configuration, change
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
