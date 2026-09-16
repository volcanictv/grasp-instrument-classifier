"""Optional SAM2-propagated temporal-track correction, gated on the
ensemble's own prediction confidence. Not a required dependency of the
base package -- only import this module if you use it; it needs `sam2`
installed and a SAM2 checkpoint downloaded separately (see README's
"Temporal tracking" section).

Validated policy (parent research repo, docs/DECISIONS.md 2026-09-16):
running SAM2 propagation on every prediction is a net negative -- it has
a measured ~9.8% chance of turning an already-correct, low-confidence
prediction wrong, which costs more than it gains once applied
indiscriminately. Only run it when the ensemble's own confidence falls
below `CONFIDENCE_THRESHOLD`. At that threshold, measured on the full
official test set: accuracy 0.9343 -> 0.9567, macro-F1 0.903 -> 0.934,
for ~13.5s of extra latency per corrected instance (offline/batch use
only -- SAM2's own per-frame encoding cost alone rules out real-time use
on any hardware).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from grasp_classifier.classes import CLASS_NAMES
from grasp_classifier.ensemble import EnsembleClassifier, Prediction
from grasp_classifier.preprocess import box_from_mask

CONFIDENCE_THRESHOLD = 0.80


class TemporalTracker:
    """Wraps a SAM2 video predictor to propagate one frame's instance mask
    across nearby frames, then reuses an already-loaded `EnsembleClassifier`
    to score every propagated frame and average the result (avg-softmax --
    the aggregation rule validated as stronger than majority-vote in the
    parent research repo's measurements).
    """

    def __init__(
        self,
        classifier: EnsembleClassifier,
        sam2_checkpoint: str,
        sam2_config: str,
        device: str = "cpu",
    ):
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise ImportError(
                "SAM2 temporal tracking needs the `sam2` package, which is not installed "
                "by default (pip install git+https://github.com/facebookresearch/sam2). "
                "See README's Temporal tracking section."
            ) from exc
        self.classifier = classifier
        self._predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=device)

    def predict(
        self,
        frames_dir: str | Path,
        center_idx: int,
        box_xywh: tuple[int, int, int, int],
        mask: np.ndarray,
    ) -> Prediction:
        """`frames_dir`: a directory of consecutive video frames, named so
        that sorting the filenames gives chronological order (SAM2's video
        predictor reads a frame directory, not in-memory arrays).
        `center_idx`: index into the sorted frame list of the frame `mask`
        and `box_xywh` belong to (the real, already-known instance).

        Propagates the mask forward and backward from `center_idx` across
        every other frame in `frames_dir`, classifies each resulting crop
        with the wrapped ensemble, and returns the track-averaged
        prediction.
        """
        frame_paths = sorted(Path(frames_dir).iterdir())
        if not (0 <= center_idx < len(frame_paths)):
            raise ValueError(f"center_idx {center_idx} out of range for {len(frame_paths)} frames in {frames_dir}")

        state = self._predictor.init_state(video_path=str(frames_dir))
        self._predictor.add_new_mask(state, frame_idx=center_idx, obj_id=1, mask=mask.astype(bool))

        track_masks: dict[int, np.ndarray] = {center_idx: mask.astype(bool)}
        for frame_idx, _obj_ids, mask_logits in self._predictor.propagate_in_video(state):
            if frame_idx != center_idx:
                track_masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()
        for frame_idx, _obj_ids, mask_logits in self._predictor.propagate_in_video(state, reverse=True):
            if frame_idx != center_idx:
                track_masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()

        per_frame_probs = []
        for frame_idx, frame_mask in track_masks.items():
            if not frame_mask.any():
                continue
            image = np.array(Image.open(frame_paths[frame_idx]).convert("RGB"))
            if frame_idx == center_idx:
                result = self.classifier.predict(image, box_xywh=box_xywh, mask=frame_mask)
            else:
                result = self.classifier.predict(image, box_xywh=box_from_mask(frame_mask), mask=frame_mask)
            per_frame_probs.append([result.class_probabilities[name] for name in CLASS_NAMES])

        if not per_frame_probs:
            raise RuntimeError("propagation produced no usable frames -- mask was lost on every frame including the seed")

        avg = np.mean(per_frame_probs, axis=0)
        pred_idx = int(avg.argmax())
        return Prediction(
            class_name=CLASS_NAMES[pred_idx],
            confidence=float(avg[pred_idx]),
            class_probabilities={name: float(p) for name, p in zip(CLASS_NAMES, avg)},
        )


def predict_with_tracking(
    classifier: EnsembleClassifier,
    tracker: TemporalTracker,
    image: np.ndarray,
    frames_dir: str | Path,
    center_idx: int,
    box_xywh: tuple[int, int, int, int],
    mask: np.ndarray,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Prediction:
    """The full validated policy in one call: score the single center frame
    first (cheap), and only pay for SAM2 propagation when the ensemble's
    own confidence is below `confidence_threshold`. `image` is the center
    frame's own array (already available to most callers without a re-read);
    `frames_dir`/`center_idx` are only touched if tracking actually runs.
    """
    single = classifier.predict(image, box_xywh=box_xywh, mask=mask)
    if single.confidence >= confidence_threshold:
        return single
    return tracker.predict(frames_dir, center_idx, box_xywh, mask)
