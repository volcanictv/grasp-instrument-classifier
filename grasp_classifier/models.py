"""Model definitions for the two backbones used by the ensemble.

Standalone from the research repo's registry pattern on purpose: this
package ships a fixed, known set of 4 checkpoints, not an experiment
surface, so a plain builder per architecture is simpler than bringing
over a registry built for trying many configs.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import mobilenet_v3_small, resnet50

NUM_CLASSES = 7


def build_resnet50(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_mobilenet_v3_small(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


BUILDERS = {
    "resnet50": build_resnet50,
    "mobilenet_v3_small": build_mobilenet_v3_small,
}
