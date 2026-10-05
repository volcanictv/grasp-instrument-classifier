import numpy as np
import pytest
import torch
import yaml

from grasp_classifier.evidential import EvidentialEnsembleClassifier
from grasp_classifier.models import BUILDERS


@pytest.fixture(scope="session")
def synthetic_classifier(tmp_path_factory):
    """Four randomly initialised small members saved as checkpoints: exercises loading, cropping and scoring without any shipped weights."""
    tmp = tmp_path_factory.mktemp("members")
    torch.manual_seed(0)
    spec = [("resnet50_320", 320, True), ("resnet50_224", 224, True), ("baseline", 224, False), ("letterbox_crop", 224, True)]
    members = []
    for label, size, letterbox in spec:
        model = BUILDERS["mobilenet_v3_small"](num_classes=7)
        torch.save(model.state_dict(), tmp / f"{label}.pt")
        members.append({"checkpoint": f"{label}.pt", "model": "mobilenet_v3_small", "image_size": size, "letterbox": letterbox, "label": label})
    config = tmp / "config.yaml"
    config.write_text(yaml.safe_dump({"weight_resnet50_320": 0.4, "epistemic_gate": 1e-3, "members": members}))
    return EvidentialEnsembleClassifier(config, device="cpu")


@pytest.fixture
def frame():
    return np.random.default_rng(1).integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
