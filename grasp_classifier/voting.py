"""Vote arithmetic shared by the single-frame ensemble and the temporal
tracker. No probabilities anywhere: a vote is the index of a stochastic
pass's largest logit, and everything below is counting and weighting votes.
"""

from __future__ import annotations

import numpy as np


def plurality(primary: np.ndarray, tiebreak: np.ndarray) -> int:
    """Index of the largest `primary` entry. A tie goes to the larger
    `tiebreak` entry (the dropout-off votes), then to the lower class index."""
    key = np.round(primary, 9) * 1e3 + np.round(tiebreak, 9)
    return int(max(range(len(key)), key=lambda k: (key[k], -k)))


def vote_shares(votes: np.ndarray, n_classes: int) -> np.ndarray:
    """Share of each class among integer votes (1-D array of class indices)."""
    return np.bincount(votes, minlength=n_classes) / len(votes)


def combine_frame_votes(frame_mc: np.ndarray, frame_det: np.ndarray) -> tuple[int, float]:
    """Combine per-frame vote distributions of a tracked instance.

    `frame_mc`, `frame_det`: (frames, classes) weighted vote shares from the
    stochastic and the dropout-off passes. Each frame counts in proportion to
    how unanimous it is (its own top vote share), so an unsure frame counts
    for less; the class with the largest weighted total wins, ties going to
    the dropout-off votes.

    Returns (predicted class index, track uncertainty), where uncertainty is
    the share of the frames' stochastic votes, pooled evenly, that name a
    different class than the one returned.
    """
    weights = frame_mc.max(axis=1)
    weighted = (weights[:, None] * frame_mc).sum(axis=0)
    pred = plurality(weighted, frame_det.sum(axis=0))
    pooled = frame_mc.sum(axis=0)
    return pred, float(np.round(1 - pooled[pred] / pooled.sum(), 9))
