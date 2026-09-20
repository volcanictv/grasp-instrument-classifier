"""Tests the uncertainty-gating policy in isolation, with stub classifier/
tracker objects -- this is orchestration logic (call the cheap path first,
only call the expensive path at or above the gate), not something that needs
a real SAM2 checkpoint or GPU to verify. See test_ensemble.py for the real
model's own behavior and test_voting.py for the vote combination rule.
"""

import numpy as np
import pytest

from grasp_classifier.ensemble import DEFAULT_UNCERTAINTY_GATE, Prediction
from grasp_classifier.preprocess import box_from_mask
from grasp_classifier.tracking import predict_with_tracking


class StubClassifier:
    uncertainty_gate = DEFAULT_UNCERTAINTY_GATE

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


def _prediction(class_name: str, uncertainty: float, tracked: bool = False) -> Prediction:
    return Prediction(
        class_name=class_name, uncertainty=uncertainty, vote_shares={class_name: 1 - uncertainty},
        needs_tracking=uncertainty >= DEFAULT_UNCERTAINTY_GATE, tracked=tracked,
    )


@pytest.fixture
def dummy_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def dummy_mask():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:20, 10:20] = True
    return mask


def _run(classifier, tracker, image, mask, **kwargs):
    return predict_with_tracking(
        classifier, tracker, image, frames_dir="unused", center_idx=0, box_xywh=(10, 10, 10, 10), mask=mask, **kwargs
    )


def test_low_uncertainty_skips_tracking(dummy_image, dummy_mask):
    single = _prediction("Bipolar Forceps", 0.02)
    classifier, tracker = StubClassifier(single), StubTracker(_prediction("Prograsp Forceps", 0.3, tracked=True))

    result = _run(classifier, tracker, dummy_image, dummy_mask)

    assert result is single
    assert classifier.predict_calls == 1
    assert tracker.predict_calls == 0


def test_high_uncertainty_runs_tracking(dummy_image, dummy_mask):
    tracker_result = _prediction("Prograsp Forceps", 0.3, tracked=True)
    classifier, tracker = StubClassifier(_prediction("Bipolar Forceps", 0.45)), StubTracker(tracker_result)

    result = _run(classifier, tracker, dummy_image, dummy_mask)

    assert result is tracker_result
    assert result.tracked
    assert classifier.predict_calls == 1
    assert tracker.predict_calls == 1


def test_uncertainty_exactly_at_gate_runs_tracking(dummy_image, dummy_mask):
    """The gate is `uncertainty >= gate` -> tracked, matching the validated
    policy (instances at or above the gate get tracked)."""
    tracker_result = _prediction("Prograsp Forceps", 0.3, tracked=True)
    classifier = StubClassifier(_prediction("Bipolar Forceps", DEFAULT_UNCERTAINTY_GATE))
    tracker = StubTracker(tracker_result)

    assert _run(classifier, tracker, dummy_image, dummy_mask) is tracker_result


def test_custom_gate_overrides_default(dummy_image, dummy_mask):
    # 0.05 uncertainty would normally skip tracking (default gate 0.09), but a
    # stricter caller-supplied gate should still trigger it.
    tracker_result = _prediction("Prograsp Forceps", 0.3, tracked=True)
    classifier, tracker = StubClassifier(_prediction("Bipolar Forceps", 0.05)), StubTracker(tracker_result)

    assert _run(classifier, tracker, dummy_image, dummy_mask, uncertainty_gate=0.03) is tracker_result


def test_box_from_mask_used_for_propagated_frames_matches_mask_extent(dummy_mask):
    # sanity check that the helper tracking.py relies on for non-center
    # frames actually produces a box tightly matching the mask, since
    # TemporalTracker.predict has no separate unit test path for this
    # without a real SAM2 checkpoint.
    x, y, w, h = box_from_mask(dummy_mask)
    assert (x, y, w, h) == (10, 10, 10, 10)
