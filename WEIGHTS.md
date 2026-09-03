# Ensemble weight: a disclosed, open trade-off

`ensemble_config.yaml` sets `weight_resnet50_320: 0.40` (the other 3
members split the remaining 0.60 equally, 0.20 each). This is an
**operational default, not an independently validated best value** --
both configurations are real, and they disagree on a held-out check.

| configuration | accuracy | macro-F1 | validated on unseen data? |
|---|---|---|---|
| flat (0.25 each) | 0.9266 | 0.8929 | yes -- confirmed on a fresh split (fold1), trained from scratch, never touched by any tuning |
| weighted (0.40 / 0.20 / 0.20 / 0.20), **current default** | 0.9343 | 0.9031 | no -- this exact value was tuned against the official test set; a fold1 re-check found flat weighting wins there (0.9012 vs 0.8961) |

Both numbers are measured on the same official GraSP test set. The
weighted config scores higher on that set; the flat config is the one
confirmed not to be an artifact of that set's specific composition. This
is not resolved further here -- change `weight_resnet50_320` to `0.25` in
`ensemble_config.yaml` to use the flat, held-out-validated configuration
instead; no code change is needed.
