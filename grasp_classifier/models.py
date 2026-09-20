"""Model definitions for the two backbones used by the ensemble.

Standalone from the research repo's registry pattern on purpose: this
package ships a fixed, known set of 4 checkpoints, not an experiment
surface, so a plain builder per architecture is simpler than bringing
over a registry built for trying many configs.

Both backbones carry dropout inside the network (and were trained with it),
because the pipeline's predictions and uncertainty come from votes over
stochastic passes with dropout left on (see grasp_classifier.ensemble).
The layer structure here must match the checkpoints exactly: wrapping a
stage in nn.Sequential renames its state-dict keys.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import mobilenet_v3_small, resnet50

NUM_CLASSES = 7
HEAD_DROPOUT = 0.2
FEATURE_DROPOUT = 0.1


def build_resnet50_dropout(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Dropout(0.2) before the classifier, plus channel-wise Dropout2d(0.1)
    after layer3 and layer4."""
    model = resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=HEAD_DROPOUT), nn.Linear(model.fc.in_features, num_classes))
    model.layer3 = nn.Sequential(model.layer3, nn.Dropout2d(p=FEATURE_DROPOUT))
    model.layer4 = nn.Sequential(model.layer4, nn.Dropout2d(p=FEATURE_DROPOUT))
    return model


def build_mobilenet_v3_small_dropout(num_classes: int = NUM_CLASSES) -> nn.Module:
    """torchvision's own Dropout(0.2) before the classifier, plus channel-wise
    Dropout2d(0.1) after features[8] and at the end of features."""
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    layers = list(model.features)
    model.features = nn.Sequential(
        *layers[:9], nn.Dropout2d(p=FEATURE_DROPOUT), *layers[9:], nn.Dropout2d(p=FEATURE_DROPOUT)
    )
    return model


BUILDERS = {
    "resnet50_deepdropout": build_resnet50_dropout,
    "mobilenet_v3_small_deepdropout": build_mobilenet_v3_small_dropout,
}
