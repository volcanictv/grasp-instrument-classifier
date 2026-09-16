import numpy as np
import pytest

from grasp_classifier import CLASS_NAMES, EnsembleClassifier

# Loads the real 4 checkpoints from weights/ -- these are committed to the
# repo, so this exercises the actual shipped package, not a mock.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def classifier() -> EnsembleClassifier:
    return EnsembleClassifier(device="cpu")


@pytest.fixture
def random_frame() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


def test_predict_returns_valid_class_name(classifier, random_frame):
    result = classifier.predict(random_frame, box_xywh=(100, 100, 150, 150))
    assert result.class_name in CLASS_NAMES


def test_predict_probabilities_sum_to_one(classifier, random_frame):
    result = classifier.predict(random_frame, box_xywh=(100, 100, 150, 150))
    total = sum(result.class_probabilities.values())
    assert total == pytest.approx(1.0, abs=1e-4)


def test_predict_confidence_matches_argmax_class(classifier, random_frame):
    result = classifier.predict(random_frame, box_xywh=(100, 100, 150, 150))
    assert result.confidence == pytest.approx(result.class_probabilities[result.class_name], abs=1e-6)


def test_predict_all_seven_classes_present(classifier, random_frame):
    result = classifier.predict(random_frame, box_xywh=(100, 100, 150, 150))
    assert set(result.class_probabilities.keys()) == set(CLASS_NAMES)


def test_predict_is_deterministic(classifier, random_frame):
    a = classifier.predict(random_frame, box_xywh=(100, 100, 150, 150))
    b = classifier.predict(random_frame, box_xywh=(100, 100, 150, 150))
    assert a.class_name == b.class_name
    assert a.confidence == pytest.approx(b.confidence, abs=1e-6)


def test_predict_with_mask_does_not_crash_and_stays_valid(classifier, random_frame):
    mask = np.zeros(random_frame.shape[:2], dtype=bool)
    mask[110:200, 110:200] = True
    result = classifier.predict(random_frame, box_xywh=(100, 100, 150, 150), mask=mask)
    assert result.class_name in CLASS_NAMES
    assert sum(result.class_probabilities.values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_box_clipped_at_frame_edge_does_not_crash(classifier, random_frame):
    h, w = random_frame.shape[:2]
    result = classifier.predict(random_frame, box_xywh=(w - 30, h - 30, 100, 100))
    assert result.class_name in CLASS_NAMES


def test_flat_weighting_config_loads_and_predicts(random_frame, tmp_path):
    """WEIGHTS.md documents weight_resnet50_320: 0.25 as a real, supported
    alternative configuration -- confirm it actually loads and runs, not
    just that the default does."""
    import shutil
    from pathlib import Path

    package_root = Path(__file__).resolve().parent.parent
    flat_config = tmp_path / "flat_ensemble_config.yaml"
    original = (package_root / "ensemble_config.yaml").read_text()
    flat_config.write_text(original.replace("weight_resnet50_320: 0.40", "weight_resnet50_320: 0.25"))

    classifier = EnsembleClassifier(config_path=flat_config, device="cpu")
    result = classifier.predict(random_frame, box_xywh=(100, 100, 150, 150))
    assert result.class_name in CLASS_NAMES
