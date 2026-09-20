"""Loads the 4-checkpoint weighted ensemble and exposes a single
`predict` call: given a frame, a box, and (ideally) a mask, returns the
instrument class and how uncertain the ensemble is about it.

No softmax anywhere. Each member runs `mc_samples` stochastic passes with
dropout left on; every pass votes for the class of its largest logit, and
the votes are weighted by member weight. The prediction is the class with the
most votes, and the uncertainty is the share of votes that disagree with it.

This is the only entry point most integrations need -- see
examples/predict_example.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from grasp_classifier.classes import CLASS_NAMES
from grasp_classifier.models import BUILDERS
from grasp_classifier.preprocess import build_eval_transform, crop_instance
from grasp_classifier.voting import plurality, vote_shares

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNCERTAINTY_GATE = 0.09


@dataclass
class Prediction:
    class_name: str
    uncertainty: float
    """Share of the weighted stochastic votes that disagree with `class_name`
    (0 = unanimous). For a tracked prediction, the share of frame votes,
    pooled evenly over the frames, that name a different class."""
    vote_shares: dict[str, float]
    """Weighted share of the stochastic votes each class received (sums to 1)."""
    needs_tracking: bool
    """True when `uncertainty` reaches the configured gate, i.e. this
    instance is one temporal tracking is meant for."""
    tracked: bool = False


def _enable_mc_dropout(model: nn.Module) -> None:
    """Dropout layers stochastic, everything else (BatchNorm) in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()


class EnsembleClassifier:
    def __init__(
        self,
        config_path: Path | str = PACKAGE_ROOT / "ensemble_config.yaml",
        device: str = "cpu",
        mc_samples: int | None = None,
        seed: int | None = None,
    ):
        """`mc_samples`: stochastic passes per member (default from the
        config, 20). `seed`: seeds torch's global RNG, which draws the dropout
        masks, for repeatable votes; leave None for fresh draws each call."""
        config = yaml.safe_load(Path(config_path).read_text())
        self.mc_samples = int(mc_samples if mc_samples is not None else config.get("mc_samples", 20))
        self.uncertainty_gate = float(config.get("uncertainty_gate", DEFAULT_UNCERTAINTY_GATE))
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)

        self._members = []
        for member in config["members"]:
            model = BUILDERS[member["model"]](num_classes=len(CLASS_NAMES))
            checkpoint_path = PACKAGE_ROOT / member["checkpoint"]
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            # strict=False: the checkpoints also carry a duplicate "head.*"
            # alias of the classifier weights -- harmless extras, never a
            # missing required weight.
            result = model.load_state_dict(state_dict, strict=False)
            assert not result.missing_keys, f"{checkpoint_path} is missing required weights: {result.missing_keys}"
            model.eval().to(self.device)
            self._members.append({
                "model": model,
                "transform": build_eval_transform(member["image_size"]),
                "letterbox": member["letterbox"],
                "label": member["label"],
            })

        weight_320 = config["weight_resnet50_320"]
        n_rest = len(self._members) - 1
        w_rest = (1 - weight_320) / n_rest
        raw = np.array([weight_320 if m["label"] == "resnet50_320" else w_rest for m in self._members])
        self._weights = raw / raw.sum()

    @torch.no_grad()
    def vote_distribution(
        self,
        image: np.ndarray,
        box_xywh: tuple[int, int, int, int],
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Weighted vote shares over the 7 classes, as
        (stochastic passes, dropout-off pass). Both sum to 1."""
        n_classes = len(CLASS_NAMES)
        stochastic = np.zeros(n_classes)
        deterministic = np.zeros(n_classes)
        for weight, member in zip(self._weights, self._members):
            crop = crop_instance(image, box_xywh, mask=mask, letterbox=member["letterbox"])
            x = member["transform"](crop).unsqueeze(0).to(self.device)
            model = member["model"]
            model.eval()
            deterministic[int(model(x).argmax(dim=1))] += weight
            _enable_mc_dropout(model)
            votes = model(x.repeat(self.mc_samples, 1, 1, 1)).argmax(dim=1).cpu().numpy()
            model.eval()
            stochastic += weight * vote_shares(votes, n_classes)
        return stochastic, deterministic

    def prediction_from_votes(self, stochastic: np.ndarray, deterministic: np.ndarray) -> Prediction:
        pred_idx = plurality(stochastic, deterministic)
        uncertainty = float(np.round(1 - stochastic[pred_idx], 9))
        return Prediction(
            class_name=CLASS_NAMES[pred_idx],
            uncertainty=uncertainty,
            vote_shares={name: float(v) for name, v in zip(CLASS_NAMES, stochastic)},
            needs_tracking=uncertainty >= self.uncertainty_gate,
        )

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
        return self.prediction_from_votes(*self.vote_distribution(image, box_xywh, mask))
