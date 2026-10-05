"""The three stages together: boxes -> masks -> classes with uncertainty -> temporal correction of the uncertain ones.

For one keyframe: `BoxSegmenter` turns each box into an instance mask, `EvidentialEnsembleClassifier` classifies every mask in one pass and scores
its uncertainty, and instances whose epistemic score reaches the gate are re-classified from SAM2-propagated masks over the neighbouring frames by
`EvidentialTracker`. `semantic_map` paints the result into the per-pixel class map that mIoU, IoU and mcIoU are computed on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from grasp_classifier.classes import CLASS_NAMES
from grasp_classifier.evidential import EvidentialEnsembleClassifier


@dataclass
class InstanceResult:
    box_xywh: tuple[int, int, int, int]
    mask: np.ndarray
    class_index: int
    class_name: str
    epistemic: float
    """Epistemic score of the single-pass prediction (larger = less certain), also for tracked instances."""
    belief: dict[str, float]
    score: float
    """Belief of the chosen class, used to order overlapping masks when painting the semantic map."""
    tracked: bool
    single_pass_class: str
    """The class before temporal correction; differs from `class_name` when tracking changed the answer."""


class InstrumentPipeline:
    def __init__(self, segmenter, classifier: EvidentialEnsembleClassifier, tracker=None, epistemic_gate: float | None = None):
        """`tracker` may be None (single-pass only). `epistemic_gate` overrides the classifier's configured gate."""
        self.segmenter = segmenter
        self.classifier = classifier
        self.tracker = tracker
        self.gate = classifier.epistemic_gate if epistemic_gate is None else float(epistemic_gate)

    def process_frame(
        self,
        image: np.ndarray,
        boxes_xywh: Sequence[Sequence[float]],
        context_frames: Sequence[np.ndarray] | None = None,
        center_idx: int | None = None,
        masks: Sequence[np.ndarray] | None = None,
    ) -> list[InstanceResult]:
        """`image`: the keyframe, RGB uint8. `boxes_xywh`: one (x, y, w, h) per instrument, in native pixels. `context_frames` and `center_idx`: the
        consecutive frames around the keyframe (the keyframe included, at `center_idx`), needed only if tracking is on and an instance is flagged.
        `masks`: precomputed instance masks to use instead of segmenting."""
        boxes = [tuple(int(round(v)) for v in b) for b in boxes_xywh]
        if masks is None:
            masks = self.segmenter.segment(image, boxes)
        results = []
        for box, mask in zip(boxes, masks):
            if not mask.any():
                continue
            single = self.classifier.predict(image, box, mask)
            final = single
            if single.epistemic >= self.gate and self.tracker is not None and context_frames is not None and center_idx is not None:
                final = self.tracker.track(context_frames, center_idx, mask, box)
            idx = CLASS_NAMES.index(final.class_name)
            results.append(InstanceResult(box_xywh=box, mask=mask, class_index=idx, class_name=final.class_name, epistemic=single.epistemic,
                                          belief=final.belief, score=final.belief[final.class_name], tracked=final.tracked,
                                          single_pass_class=single.class_name))
        return results


def semantic_map(shape: tuple[int, int], results: Sequence[InstanceResult]) -> np.ndarray:
    """Per-pixel class map (0 background, 1 to 7 the classes in CLASS_NAMES order); a higher score is painted last, so it wins an overlap."""
    sem = np.zeros(shape, dtype=np.int8)
    for r in sorted(results, key=lambda r: r.score):
        sem[r.mask] = r.class_index + 1
    return sem
