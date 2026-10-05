"""Temporal correction of an uncertain instance: SAM2 video propagation of its mask across the neighbouring frames, the evidential ensemble on
every propagated crop, and the frame-wise fusion of grasp_classifier.evidential.fuse_frames.

Frames are expected at the sampling the classifier was developed on (about 1 per second). By default the window reaches `window` frames on each
side of the keyframe (10 each way, as in the reported results), which means the correction uses future frames and is not causal; `causal=True`
propagates backwards only.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from grasp_classifier.evidential import EvidentialEnsembleClassifier, EvidentialPrediction
from grasp_classifier.preprocess import box_from_mask

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


class EvidentialTracker:
    def __init__(
        self,
        classifier: EvidentialEnsembleClassifier,
        sam2_checkpoint: Path | str,
        sam2_config: str = SAM2_CONFIG,
        device: str = "cuda",
        window: int = 10,
        causal: bool = False,
    ):
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise ImportError("tracking needs the `sam2` package: pip install git+https://github.com/facebookresearch/sam2") from exc
        self.classifier = classifier
        self.window = window
        self.causal = causal
        self.device = device
        self._predictor = build_sam2_video_predictor(sam2_config, str(sam2_checkpoint), device=device)

    def track(self, frames: Sequence[np.ndarray], center_idx: int, mask: np.ndarray, box_xywh: tuple[int, int, int, int]) -> EvidentialPrediction:
        """`frames`: consecutive RGB frames (HxWx3 uint8), `center_idx` the keyframe `mask` and `box_xywh` belong to. Returns the fused prediction
        (tracked=True), or the keyframe's own single-pass prediction if propagation lost the mask on every other frame."""
        if not 0 <= center_idx < len(frames):
            raise ValueError(f"center_idx {center_idx} is outside the {len(frames)} frames given")
        lo = max(0, center_idx - self.window)
        hi = center_idx if self.causal else min(len(frames) - 1, center_idx + self.window)
        local_center = center_idx - lo
        window = list(frames[lo:hi + 1])
        track_masks: dict[int, np.ndarray] = {local_center: mask.astype(bool)}
        with tempfile.TemporaryDirectory() as tmp:
            for k, frame in enumerate(window):
                Image.fromarray(frame).save(Path(tmp) / f"{k:05d}.jpg", quality=95)
            with torch.inference_mode():
                state = self._predictor.init_state(video_path=tmp)
                self._predictor.add_new_mask(state, frame_idx=local_center, obj_id=1, mask=mask.astype(bool))
                if not self.causal:
                    for idx, _ids, logits in self._predictor.propagate_in_video(state):
                        if idx != local_center:
                            track_masks[idx] = (logits[0, 0] > 0).cpu().numpy()
                for idx, _ids, logits in self._predictor.propagate_in_video(state, reverse=True):
                    if idx != local_center:
                        track_masks[idx] = (logits[0, 0] > 0).cpu().numpy()
                self._predictor.reset_state(state)

        frame_logits = []
        for idx in sorted(track_masks):
            frame_mask = track_masks[idx]
            if not frame_mask.any():
                continue
            box = box_xywh if idx == local_center else box_from_mask(frame_mask)
            frame_logits.append(self.classifier.member_logits(window[idx], box, mask=frame_mask))
        if not frame_logits:
            raise RuntimeError("the mask is empty on every frame including the keyframe; nothing to classify")
        return self.classifier.combine_track(frame_logits)
