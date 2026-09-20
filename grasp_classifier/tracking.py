"""Optional SAM2-propagated temporal-track correction, gated on the
ensemble's own vote uncertainty. Not a required dependency of the base
package -- only import this module if you use it; it needs `sam2`
installed and a SAM2 checkpoint downloaded separately (see README's
"Temporal tracking" section).

Policy (parent research repo, docs/DECISIONS.md 2026-09-20): tracking every
instance is not worth its cost, so only instances whose vote uncertainty
reaches the gate (`EnsembleClassifier.uncertainty_gate`, 0.09 by default) are
tracked. On the official GraSP test set that is 29.1% of instances, and it
takes accuracy from 0.9266 to 0.9623 and macro-F1 from 0.890 to 0.935, for
about 24s of GPU time per tracked instance (offline/batch use only -- SAM2's
own per-frame encoding cost alone rules out real-time use on any hardware).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from grasp_classifier.classes import CLASS_NAMES
from grasp_classifier.ensemble import EnsembleClassifier, Prediction
from grasp_classifier.preprocess import box_from_mask
from grasp_classifier.voting import combine_frame_votes


class TemporalTracker:
    """Wraps a SAM2 video predictor to propagate one frame's instance mask
    across nearby frames, then reuses an already-loaded `EnsembleClassifier`
    to get every propagated frame's votes and combine them with
    `grasp_classifier.voting.combine_frame_votes` (frames that are more
    unanimous count for more).
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
        every other frame in `frames_dir`, gets each resulting crop's votes
        from the wrapped ensemble, and returns the combined prediction with
        `tracked=True`.
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

        frame_mc, frame_det = [], []
        for frame_idx, frame_mask in track_masks.items():
            if not frame_mask.any():
                continue
            image = np.array(Image.open(frame_paths[frame_idx]).convert("RGB"))
            box = box_xywh if frame_idx == center_idx else box_from_mask(frame_mask)
            stochastic, deterministic = self.classifier.vote_distribution(image, box, mask=frame_mask)
            frame_mc.append(stochastic)
            frame_det.append(deterministic)

        if not frame_mc:
            raise RuntimeError("propagation produced no usable frames -- mask was lost on every frame including the seed")

        frame_mc, frame_det = np.stack(frame_mc), np.stack(frame_det)
        pred_idx, uncertainty = combine_frame_votes(frame_mc, frame_det)
        pooled = frame_mc.mean(axis=0)
        return Prediction(
            class_name=CLASS_NAMES[pred_idx],
            uncertainty=uncertainty,
            vote_shares={name: float(v) for name, v in zip(CLASS_NAMES, pooled)},
            needs_tracking=uncertainty >= self.classifier.uncertainty_gate,
            tracked=True,
        )


def predict_with_tracking(
    classifier: EnsembleClassifier,
    tracker: TemporalTracker,
    image: np.ndarray,
    frames_dir: str | Path,
    center_idx: int,
    box_xywh: tuple[int, int, int, int],
    mask: np.ndarray,
    uncertainty_gate: float | None = None,
) -> Prediction:
    """The full policy in one call: get the single center frame's votes first
    (cheap), and only pay for SAM2 propagation when its uncertainty reaches
    the gate. `image` is the center frame's own array (already available to
    most callers without a re-read); `frames_dir`/`center_idx` are only
    touched if tracking actually runs. `uncertainty_gate` overrides the
    classifier's configured gate for this call.
    """
    single = classifier.predict(image, box_xywh=box_xywh, mask=mask)
    gate = classifier.uncertainty_gate if uncertainty_gate is None else uncertainty_gate
    if single.uncertainty < gate:
        return single
    return tracker.predict(frames_dir, center_idx, box_xywh, mask)
