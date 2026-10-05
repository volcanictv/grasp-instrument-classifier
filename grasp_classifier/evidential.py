"""Single-pass evidential ensemble, the research default.

Each member runs one deterministic pass and its 7 output logits are read as Dirichlet evidence: evidence = exp(clip(logit, -10, 10)),
alpha = evidence + 1. The ensemble's alpha is the weighted mean of the members' alphas. The prediction is the class with the largest
mean belief mu = alpha / alpha0, and the uncertainty is the epistemic score of Duan et al. (WACV 2024, arXiv 2311.11367),

    S1 = (1 - sum(mu ** 2)) / (alpha0 + 1),

high when the members carry little evidence or disagree. No softmax is involved. An instance whose S1 reaches the gate is one temporal
tracking is meant for (`EvidentialPrediction.needs_tracking`).

Over a tracked instance, each frame's mixed alpha gives a belief mu_f, and the frames are combined by confidence-weighted belief:
combined = sum_f max(mu_f) * mu_f (`fuse_frames`). More decisive frames count for more.

The numbers here are the same ones the research pipeline uses; tests/test_evidential_parity.py (needs the checkpoints) compares them.
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
EVIDENCE_CLAMP = 10.0
DEFAULT_EPISTEMIC_GATE = 1.7e-5  # chosen on a held-out fold, not on the test set


def alpha_from_logits(logits: np.ndarray, clamp: float = EVIDENCE_CLAMP) -> np.ndarray:
    return np.exp(np.clip(logits, -clamp, clamp)) + 1.0


def mixture_alpha(member_logits: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(members, classes) logits and (members,) weights summing to 1 -> (classes,) mixed Dirichlet parameters."""
    return (weights[:, None] * alpha_from_logits(member_logits)).sum(axis=0)


def epistemic_score(alpha: np.ndarray) -> float:
    """The variance-based epistemic score S1 of a (classes,) Dirichlet parameter vector."""
    alpha0 = float(alpha.sum())
    mu = alpha / alpha0
    return float((1.0 - (mu ** 2).sum()) / (alpha0 + 1.0))


def fuse_frames(frame_member_logits: np.ndarray, weights: np.ndarray) -> tuple[int, float, np.ndarray]:
    """(frames, members, classes) logits -> (class index, the class's share of the combined belief, combined belief of all classes)."""
    alpha = (weights[None, :, None] * alpha_from_logits(frame_member_logits)).sum(axis=1)  # (frames, classes)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    combined = (mu.max(axis=1)[:, None] * mu).sum(axis=0)
    combined = combined / combined.sum()
    idx = int(combined.argmax())
    return idx, float(combined[idx]), combined


@dataclass
class EvidentialPrediction:
    class_name: str
    epistemic: float
    """The epistemic score S1; larger means less certain."""
    belief: dict[str, float]
    """Mean belief mu of the mixed Dirichlet for each class (sums to 1). Not a softmax output."""
    needs_tracking: bool
    """True when `epistemic` reaches the configured gate."""
    member_logits: np.ndarray
    """(members, 7) raw logits, kept so that tracked frames can be fused with `fuse_frames` without another forward pass."""
    tracked: bool = False


class EvidentialEnsembleClassifier:
    def __init__(self, config_path: Path | str = PACKAGE_ROOT / "evidential_config.yaml", device: str = "cpu"):
        config = yaml.safe_load(Path(config_path).read_text())
        self.epistemic_gate = float(config.get("epistemic_gate", DEFAULT_EPISTEMIC_GATE))
        self.device = torch.device(device)
        base = Path(config_path).resolve().parent
        self._members = []
        for member in config["members"]:
            model = BUILDERS[member["model"]](num_classes=len(CLASS_NAMES))
            checkpoint = Path(member["checkpoint"])
            checkpoint = checkpoint if checkpoint.is_absolute() else base / checkpoint
            state_dict = torch.load(checkpoint, map_location=self.device)
            # strict=False: the checkpoints also hold a duplicate "head.*" alias of the classifier layer; a missing required weight is still an error
            result = model.load_state_dict(state_dict, strict=False)
            assert not result.missing_keys, f"{checkpoint} is missing required weights: {result.missing_keys}"
            model.eval().to(self.device)
            self._members.append({"model": model, "transform": build_eval_transform(member["image_size"]),
                                  "letterbox": member["letterbox"], "label": member["label"]})
        weight_320 = float(config["weight_resnet50_320"])
        rest = (1.0 - weight_320) / (len(self._members) - 1)
        raw = np.array([weight_320 if m["label"] == "resnet50_320" else rest for m in self._members])
        self.weights = raw / raw.sum()

    @property
    def member_labels(self) -> list[str]:
        return [m["label"] for m in self._members]

    @torch.no_grad()
    def member_logits(self, image: np.ndarray, box_xywh: tuple[int, int, int, int], mask: np.ndarray | None = None) -> np.ndarray:
        """One deterministic pass of every member on the instance crop: (members, 7) logits.
        `image`: full RGB frame, HxWx3 uint8. `box_xywh`: (x, y, w, h) in native pixels. `mask`: optional boolean HxW instance mask."""
        out = []
        for member in self._members:
            crop = crop_instance(image, box_xywh, mask=mask, letterbox=member["letterbox"])
            x = member["transform"](crop).unsqueeze(0).to(self.device)
            out.append(member["model"](x).float().cpu().numpy()[0])
        return np.stack(out)

    def prediction_from_logits(self, logits: np.ndarray) -> EvidentialPrediction:
        alpha = mixture_alpha(logits, self.weights)
        mu = alpha / alpha.sum()
        score = epistemic_score(alpha)
        return EvidentialPrediction(class_name=CLASS_NAMES[int(mu.argmax())], epistemic=score,
                                    belief={name: float(v) for name, v in zip(CLASS_NAMES, mu)},
                                    needs_tracking=score >= self.epistemic_gate, member_logits=logits)

    def predict(self, image: np.ndarray, box_xywh: tuple[int, int, int, int], mask: np.ndarray | None = None) -> EvidentialPrediction:
        return self.prediction_from_logits(self.member_logits(image, box_xywh, mask))

    def combine_track(self, frame_logits: list[np.ndarray]) -> EvidentialPrediction:
        """Combines the per-frame (members, 7) logits of one tracked instance into a single prediction (tracked=True)."""
        idx, _share, combined = fuse_frames(np.stack(frame_logits), self.weights)
        pooled = np.stack(frame_logits).mean(axis=0)
        alpha = mixture_alpha(pooled, self.weights)
        return EvidentialPrediction(class_name=CLASS_NAMES[idx], epistemic=epistemic_score(alpha),
                                    belief={name: float(v) for name, v in zip(CLASS_NAMES, combined)}, needs_tracking=False,
                                    member_logits=pooled, tracked=True)
