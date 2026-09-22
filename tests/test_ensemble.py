import numpy as np
import pytest

from grasp_classifier import CLASS_NAMES, EnsembleClassifier

# Loads the real 4 checkpoints from weights/ -- these are committed to the
# repo, so this exercises the actual shipped package, not a mock. Uses a small
# number of stochastic passes to keep CPU runs short; the configured default
# is checked separately.
pytestmark = pytest.mark.slow

BOX = (100, 100, 150, 150)


@pytest.fixture(scope="module")
def classifier() -> EnsembleClassifier:
    return EnsembleClassifier(device="cpu", mc_samples=6, seed=0)


@pytest.fixture
def random_frame() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


def test_config_defaults_are_the_validated_ones():
    default = EnsembleClassifier(device="cpu")
    assert default.mc_samples == 20
    assert default.uncertainty_gate == pytest.approx(0.09)


def test_predict_returns_valid_class_name(classifier, random_frame):
    assert classifier.predict(random_frame, box_xywh=BOX).class_name in CLASS_NAMES


def test_vote_shares_cover_all_seven_classes_and_sum_to_one(classifier, random_frame):
    result = classifier.predict(random_frame, box_xywh=BOX)
    assert set(result.vote_shares) == set(CLASS_NAMES)
    assert sum(result.vote_shares.values()) == pytest.approx(1.0, abs=1e-6)


def test_uncertainty_is_the_share_of_votes_against_the_winner(classifier, random_frame):
    result = classifier.predict(random_frame, box_xywh=BOX)
    assert result.uncertainty == pytest.approx(1 - result.vote_shares[result.class_name], abs=1e-6)
    assert 0.0 <= result.uncertainty <= 1.0


def test_needs_tracking_follows_the_gate(classifier, random_frame):
    result = classifier.predict(random_frame, box_xywh=BOX)
    assert result.needs_tracking == (result.uncertainty >= classifier.uncertainty_gate)
    assert not result.tracked


def test_no_softmax_probabilities_in_the_output(classifier, random_frame):
    result = classifier.predict(random_frame, box_xywh=BOX)
    assert not hasattr(result, "confidence")
    assert not hasattr(result, "class_probabilities")


def test_same_seed_gives_identical_votes(random_frame):
    a = EnsembleClassifier(device="cpu", mc_samples=6, seed=123).predict(random_frame, box_xywh=BOX)
    b = EnsembleClassifier(device="cpu", mc_samples=6, seed=123).predict(random_frame, box_xywh=BOX)
    assert a.class_name == b.class_name
    assert a.vote_shares == b.vote_shares


def test_dropout_is_off_again_after_predict(classifier, random_frame):
    import torch.nn as nn

    classifier.predict(random_frame, box_xywh=BOX)
    for member in classifier._members:
        assert not any(m.training for m in member["model"].modules() if isinstance(m, (nn.Dropout, nn.Dropout2d)))


def test_predict_with_mask_does_not_crash_and_stays_valid(classifier, random_frame):
    mask = np.zeros(random_frame.shape[:2], dtype=bool)
    mask[110:200, 110:200] = True
    result = classifier.predict(random_frame, box_xywh=BOX, mask=mask)
    assert result.class_name in CLASS_NAMES
    assert sum(result.vote_shares.values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_box_clipped_at_frame_edge_does_not_crash(classifier, random_frame):
    h, w = random_frame.shape[:2]
    assert classifier.predict(random_frame, box_xywh=(w - 30, h - 30, 100, 100)).class_name in CLASS_NAMES


def test_flat_weighting_config_loads_and_predicts(random_frame, tmp_path):
    """WEIGHTS.md documents weight_resnet50_320: 0.25 as a real, supported
    alternative configuration -- confirm it actually loads and runs, not
    just that the default does."""
    from pathlib import Path

    package_root = Path(__file__).resolve().parent.parent
    flat_config = tmp_path / "flat_ensemble_config.yaml"
    original = (package_root / "ensemble_config.yaml").read_text()
    flat_config.write_text(original.replace("weight_resnet50_320: 0.40", "weight_resnet50_320: 0.25"))

    flat = EnsembleClassifier(config_path=flat_config, device="cpu", mc_samples=6, seed=0)
    assert flat.predict(random_frame, box_xywh=BOX).class_name in CLASS_NAMES
