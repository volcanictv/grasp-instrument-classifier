"""The three stages together: boxes -> masks -> classes with uncertainty -> temporal correction of the uncertain ones.

For one keyframe: `BoxSegmenter` turns each box into an instance mask, `EvidentialEnsembleClassifier` classifies every mask in one pass and scores
its uncertainty, and instances whose epistemic score reaches the gate are re-classified from SAM2-propagated masks over the neighbouring frames by
`EvidentialTracker`. An optional `FrameDecoder` then labels the instruments of the frame jointly (a class rarely repeats within a frame, see
`grasp_classifier.decode`). `semantic_map` paints the result into the per-pixel class map that mIoU, IoU and mcIoU are computed on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from grasp_classifier.decode import FrameDecoder
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
    decoded: bool = False
    """True when the frame decode changed the class that the tracked or single-pass belief alone would have given."""


class InstrumentPipeline:
    def __init__(self, segmenter, classifier: EvidentialEnsembleClassifier, tracker=None, epistemic_gate: float | None = None,
                 decoder: FrameDecoder | None = None):
        """`tracker` may be None (single-pass only). `epistemic_gate` overrides the classifier's configured gate. `decoder` (optional) labels the
        instruments of each frame jointly after tracking; its class order must be the classifier's."""
        if decoder is not None:
            assert decoder.class_names == classifier.class_names, "the decoder was built for another class set than the classifier"
        self.segmenter = segmenter
        self.classifier = classifier
        self.tracker = tracker
        self.decoder = decoder
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
        staged = []
        for box, mask in zip(boxes, masks):
            if not mask.any():
                continue
            single = self.classifier.predict(image, box, mask)
            final = single
            if single.epistemic >= self.gate and self.tracker is not None and context_frames is not None and center_idx is not None:
                final = self.tracker.track(context_frames, center_idx, mask, box)
            staged.append((box, mask, single, final))
        names = self.classifier.class_names
        labels = None
        if self.decoder is not None and staged:
            labels = self.decoder.decode(np.array([[final.belief[n] for n in names] for _, _, _, final in staged]))
        results = []
        for k, (box, mask, single, final) in enumerate(staged):
            idx = int(labels[k]) if labels is not None else names.index(final.class_name)
            results.append(InstanceResult(box_xywh=box, mask=mask, class_index=idx, class_name=names[idx], epistemic=single.epistemic,
                                          belief=final.belief, score=final.belief[names[idx]], tracked=final.tracked,
                                          single_pass_class=single.class_name, decoded=names[idx] != final.class_name))
        return results


def semantic_map(shape: tuple[int, int], results: Sequence[InstanceResult]) -> np.ndarray:
    """Per-pixel class map (0 background, 1 to 7 the classes in the classifier's class order); a higher score is painted last, so it wins an overlap."""
    sem = np.zeros(shape, dtype=np.int8)
    for r in sorted(results, key=lambda r: r.score):
        sem[r.mask] = r.class_index + 1
    return sem
