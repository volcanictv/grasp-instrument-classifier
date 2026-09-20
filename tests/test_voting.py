"""Vote arithmetic in isolation: hand-built vote distributions with known
answers, no checkpoints needed."""

import numpy as np
import pytest

from grasp_classifier.voting import combine_frame_votes, plurality, vote_shares


def test_vote_shares_counts_votes_and_sums_to_one():
    shares = vote_shares(np.array([0, 0, 2, 2, 2, 5]), n_classes=7)
    assert shares.tolist() == pytest.approx([2 / 6, 0, 3 / 6, 0, 0, 1 / 6, 0])
    assert shares.sum() == pytest.approx(1.0)


def test_plurality_picks_largest_primary():
    assert plurality(np.array([0.1, 0.6, 0.3]), np.array([1.0, 0.0, 0.0])) == 1


def test_plurality_tie_goes_to_tiebreak_votes():
    primary = np.array([0.5, 0.5, 0.0])
    assert plurality(primary, np.array([0.0, 1.0, 0.0])) == 1


def test_plurality_full_tie_goes_to_lower_index():
    assert plurality(np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.5, 0.0])) == 0


def test_plurality_ignores_float_noise_below_a_billionth():
    primary = np.array([0.3, 0.3 + 1e-12, 0.4])
    tiebreak = np.array([1.0, 0.0, 0.0])
    # 0.3 vs 0.3 + 1e-12 is a tie, resolved by the dropout-off votes; class 2 still wins overall
    assert plurality(np.array([0.5, 0.5 + 1e-12, 0.0]), tiebreak) == 0
    assert plurality(primary, tiebreak) == 2


def test_combine_frame_votes_unanimous_track_has_zero_uncertainty():
    frames = np.tile(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), (5, 1))
    pred, uncertainty = combine_frame_votes(frames, frames)
    assert pred == 2
    assert uncertainty == 0.0


def test_combine_frame_votes_weights_unsure_frames_less():
    # One unanimous frame for class 0, two split frames leaning to class 1.
    # Pooled evenly the split frames win (class 1: 1.1 vs class 0: 1.0); weighted
    # by each frame's own agreement the unanimous frame wins (1.0 vs 0.605).
    frames = np.array([[1.0, 0.0, 0.0], [0.0, 0.55, 0.45], [0.0, 0.55, 0.45]])
    assert int(frames.sum(axis=0).argmax()) == 1
    winner, _ = combine_frame_votes(frames, frames)
    assert winner == 0


def test_combine_frame_votes_uncertainty_is_share_of_pooled_votes_against_winner():
    frames = np.array([[0.75, 0.25, 0.0], [0.75, 0.25, 0.0]])
    pred, uncertainty = combine_frame_votes(frames, frames)
    assert pred == 0
    assert uncertainty == pytest.approx(0.25)
