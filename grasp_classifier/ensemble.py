"""Loads the 4-checkpoint weighted ensemble and exposes a single
`predict` call: given a frame, a box, and (ideally) a mask, returns the
instrument class.

This is the only entry point most integrations need -- see
examples/predict_example.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from grasp_classifier.classes import CLASS_NAMES
from grasp_classifier.models import BUILDERS
from grasp_classifier.preprocess import build_eval_transform, crop_instance

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Prediction:
    class_name: str
    confidence: float
    class_probabilities: dict[str, float]


class EnsembleClassifier:
    def __init__(self, config_path: Path | str = PACKAGE_ROOT / "ensemble_config.yaml", device: str = "cpu"):
        config = yaml.safe_load(Path(config_path).read_text())
        self.weight_resnet50_320 = config["weight_resnet50_320"]
        self.device = torch.device(device)

        self._members = []
        for member in config["members"]:
            model = BUILDERS[member["model"]](num_classes=len(CLASS_NAMES))
            checkpoint_path = PACKAGE_ROOT / member["checkpoint"]
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            # strict=False: some checkpoints predate a fix that stopped
            # duplicating the classification head's weights under a second
            # "head.*" key alongside its real name ("fc.*"/"classifier.*") --
            # those are harmless extras, never a missing required weight.
            result = model.load_state_dict(state_dict, strict=False)
            assert not result.missing_keys, f"{checkpoint_path} is missing required weights: {result.missing_keys}"
            model.eval().to(self.device)
            self._members.append({
                "model": model,
                "transform": build_eval_transform(member["image_size"]),
                "letterbox": member["letterbox"],
                "label": member["label"],
            })

        n_rest = len(self._members) - 1
        w_rest = (1 - self.weight_resnet50_320) / n_rest
        self._weights = [
            self.weight_resnet50_320 if m["label"] == "resnet50_320" else w_rest
            for m in self._members
        ]

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        box_xywh: tuple[int, int, int, int],
        mask: np.ndarray | None = None,
    ) -> Prediction:
        """`image`: full RGB frame, HxWx3 uint8. `box_xywh`: (x, y, w, h) in
        the frame's native pixel coordinates -- the box a detector/segmenter
        already produced upstream. `mask`: optional boolean instance mask,
        same HxW as `image`; strongly recommended when instruments overlap
        (see preprocess.crop_instance's docstring)."""
        probs_per_member = []
        for member in self._members:
            crop = crop_instance(image, box_xywh, mask=mask, letterbox=member["letterbox"])
            tensor = member["transform"](crop).unsqueeze(0).to(self.device)
            logits = member["model"](tensor)
            probs_per_member.append(torch.softmax(logits, dim=1)[0].cpu().numpy())

        avg = sum(w * p for w, p in zip(self._weights, probs_per_member))
        pred_idx = int(avg.argmax())
        return Prediction(
            class_name=CLASS_NAMES[pred_idx],
            confidence=float(avg[pred_idx]),
            class_probabilities={name: float(p) for name, p in zip(CLASS_NAMES, avg)},
        )
