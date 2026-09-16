"""Tests the confidence-gating policy in isolation, with stub classifier/
tracker objects -- this is orchestration logic (call the cheap path first,
only call the expensive path below a threshold), not something that needs
a real SAM2 checkpoint or GPU to verify. See test_ensemble.py for the real
model's own behavior.
"""

import numpy as np
import pytest

from grasp_classifier.ensemble import Prediction
from grasp_classifier.preprocess import box_from_mask
from grasp_classifier.tracking import CONFIDENCE_THRESHOLD, predict_with_tracking


class StubClassifier:
    def __init__(self, single_frame_prediction: Prediction):
        self._prediction = single_frame_prediction
        self.predict_calls = 0

    def predict(self, image, box_xywh, mask=None):
        self.predict_calls += 1
        return self._prediction


class StubTracker:
    def __init__(self, tracked_prediction: Prediction):
        self._prediction = tracked_prediction
        self.predict_calls = 0

    def predict(self, frames_dir, center_idx, box_xywh, mask):
        self.predict_calls += 1
        return self._prediction


def _prediction(class_name: str, confidence: float) -> Prediction:
    return Prediction(class_name=class_name, confidence=confidence, class_probabilities={class_name: confidence})


@pytest.fixture
def dummy_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def dummy_mask():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:20, 10:20] = True
    return mask


def test_high_confidence_skips_tracking(dummy_image, dummy_mask):
    single = _prediction("Bipolar Forceps", 0.95)
    tracker_result = _prediction("Prograsp Forceps", 0.5)
    classifier = StubClassifier(single)
    tracker = StubTracker(tracker_result)

    result = predict_with_tracking(
        classifier, tracker, dummy_image, frames_dir="unused", center_idx=0,
        box_xywh=(10, 10, 10, 10), mask=dummy_mask,
    )

    assert result is single
    assert classifier.predict_calls == 1
    assert tracker.predict_calls == 0


def test_low_confidence_runs_tracking(dummy_image, dummy_mask):
    single = _prediction("Bipolar Forceps", 0.5)
    tracker_result = _prediction("Prograsp Forceps", 0.7)
    classifier = StubClassifier(single)
    tracker = StubTracker(tracker_result)

    result = predict_with_tracking(
        classifier, tracker, dummy_image, frames_dir="unused", center_idx=0,
        box_xywh=(10, 10, 10, 10), mask=dummy_mask,
    )

    assert result is tracker_result
    assert classifier.predict_calls == 1
    assert tracker.predict_calls == 1


def test_confidence_exactly_at_threshold_skips_tracking(dummy_image, dummy_mask):
    """Threshold is `confidence >= threshold` -> single frame kept, matching
    the validated policy's own definition (instances *below* 0.80 get
    tracked, not <= 0.80)."""
    single = _prediction("Bipolar Forceps", CONFIDENCE_THRESHOLD)
    classifier = StubClassifier(single)
    tracker = StubTracker(_prediction("Prograsp Forceps", 0.5))

    result = predict_with_tracking(
        classifier, tracker, dummy_image, frames_dir="unused", center_idx=0,
        box_xywh=(10, 10, 10, 10), mask=dummy_mask,
    )

    assert result is single
    assert tracker.predict_calls == 0


def test_custom_threshold_overrides_default(dummy_image, dummy_mask):
    single = _prediction("Bipolar Forceps", 0.85)
    tracker_result = _prediction("Prograsp Forceps", 0.6)
    classifier = StubClassifier(single)
    tracker = StubTracker(tracker_result)

    # 0.85 confidence would normally skip tracking (default threshold 0.80),
    # but a stricter caller-supplied threshold should still trigger it.
    result = predict_with_tracking(
        classifier, tracker, dummy_image, frames_dir="unused", center_idx=0,
        box_xywh=(10, 10, 10, 10), mask=dummy_mask, confidence_threshold=0.90,
    )

    assert result is tracker_result


def test_box_from_mask_used_for_propagated_frames_matches_mask_extent(dummy_mask):
    # sanity check that the helper tracking.py relies on for non-center
    # frames actually produces a box tightly matching the mask, since
    # TemporalTracker.predict has no separate unit test path for this
    # without a real SAM2 checkpoint.
    x, y, w, h = box_from_mask(dummy_mask)
    assert (x, y, w, h) == (10, 10, 10, 10)
