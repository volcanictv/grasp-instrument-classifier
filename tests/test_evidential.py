import numpy as np
import pytest

from grasp_classifier.classes import CLASS_NAMES
from grasp_classifier.evidential import alpha_from_logits, epistemic_score, fuse_frames, mixture_alpha


def test_alpha_from_logits_is_clamped_exponential_plus_one():
    assert np.allclose(alpha_from_logits(np.zeros(7)), 2.0)
    assert alpha_from_logits(np.array([100.0]))[0] == pytest.approx(np.exp(10.0) + 1.0)
    assert alpha_from_logits(np.array([-100.0]))[0] == pytest.approx(np.exp(-10.0) + 1.0)


def test_epistemic_score_of_uniform_evidence_is_known():
    alpha = np.full(7, 2.0)  # alpha0 = 14, mu = 1/7
    assert epistemic_score(alpha) == pytest.approx((1 - 1 / 7) / 15)


def test_more_concentrated_evidence_is_less_uncertain():
    flat = mixture_alpha(np.zeros((4, 7)), np.full(4, 0.25))
    peaked = np.zeros((4, 7))
    peaked[:, 2] = 9.0
    assert epistemic_score(mixture_alpha(peaked, np.full(4, 0.25))) < epistemic_score(flat)


def test_members_that_disagree_are_more_uncertain_than_members_that_agree():
    agree = np.zeros((2, 7))
    agree[:, 0] = 6.0
    disagree = np.zeros((2, 7))
    disagree[0, 0], disagree[1, 3] = 6.0, 6.0
    w = np.array([0.5, 0.5])
    assert epistemic_score(mixture_alpha(disagree, w)) > epistemic_score(mixture_alpha(agree, w))


def test_fuse_frames_matches_the_formula():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, size=(5, 4, 7))
    weights = np.array([0.4, 0.2, 0.2, 0.2])
    idx, share, combined = fuse_frames(logits, weights)
    alpha = sum(weights[m] * alpha_from_logits(logits[:, m, :]) for m in range(4))
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    expected = (mu.max(axis=1)[:, None] * mu).sum(axis=0)
    assert idx == int(expected.argmax())
    assert share == pytest.approx(expected.max() / expected.sum())
    assert combined.sum() == pytest.approx(1.0)


def test_one_decisive_frame_outweighs_several_vague_ones():
    vague = np.zeros((4, 7))
    vague[:, 1] = 0.3
    decisive = np.zeros((4, 7))
    decisive[:, 5] = 8.0
    logits = np.stack([vague, vague, decisive])
    assert fuse_frames(logits, np.full(4, 0.25))[0] == 5


def test_member_weights_sum_to_one_and_favour_the_320_member(synthetic_classifier):
    w = synthetic_classifier.weights
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.4) and np.allclose(w[1:], 0.2)


def test_predict_returns_a_well_formed_prediction(synthetic_classifier, frame):
    pred = synthetic_classifier.predict(frame, (50, 40, 120, 90))
    assert pred.class_name in CLASS_NAMES
    assert sum(pred.belief.values()) == pytest.approx(1.0)
    assert 0 < pred.epistemic < 1
    assert pred.member_logits.shape == (4, 7)
    assert pred.needs_tracking == (pred.epistemic >= synthetic_classifier.epistemic_gate)


def test_prediction_is_deterministic(synthetic_classifier, frame):
    a = synthetic_classifier.predict(frame, (50, 40, 120, 90))
    b = synthetic_classifier.predict(frame, (50, 40, 120, 90))
    assert np.array_equal(a.member_logits, b.member_logits)


def test_a_mask_zeroes_everything_outside_the_instance_in_the_crop(frame):
    from grasp_classifier.preprocess import crop_instance

    mask = np.zeros(frame.shape[:2], bool)
    mask[60:100, 70:130] = True
    with_mask = np.array(crop_instance(frame, (50, 40, 120, 90), mask=mask, letterbox=False))
    without = np.array(crop_instance(frame, (50, 40, 120, 90), letterbox=False))
    assert with_mask.shape == without.shape == (90, 120, 3)
    assert (with_mask != without).any()
    assert with_mask[0, 0].sum() == 0 and with_mask[30, 40].tolist() == frame[70, 90].tolist()


def test_combine_track_is_marked_tracked_and_sums_to_one(synthetic_classifier, frame):
    logits = [synthetic_classifier.member_logits(frame, (50 + 5 * k, 40, 120, 90)) for k in range(3)]
    pred = synthetic_classifier.combine_track(logits)
    assert pred.tracked and not pred.needs_tracking
    assert sum(pred.belief.values()) == pytest.approx(1.0)
