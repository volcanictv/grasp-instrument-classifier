"""Frame-level readout: the instruments of one frame are labelled together, not one at a time.

Each instrument has a belief over the classes (the single-pass mean belief, or the fused belief of its track). The usual readout takes the largest
belief per instrument. The frame decode instead picks the labelling of the whole frame with the largest total log-belief, subject to one structural
fact of the data: a class rarely appears twice in a frame (on GraSP only the Large Needle Driver does, in 414 of 421 training frames that repeat a
class), so a second instrument of the same class pays a penalty. Optionally a class-prior shift is added to every log-belief.

    score(labelling) = sum_i [ log(belief_i[c_i] + eps) - prior_shift * log(prior[c_i]) ]  +  sum_c max(0, n_c - 1) * duplicate_penalty[c]

duplicate_penalty is <= 0 per class. Two ways to get it (both parameter-free given the training frames, see `grasp_pipeline.decode_stats`):
    hard   every class except an explicit allow-list may not repeat (penalty -1e3, effectively forbidden); the allow-list pays nothing
    soft   penalty[c] = log((frames with c twice + 1) / (frames with c + 2)), estimated from the training frames
`prior_shift = g` adds `-g * log(prior)`: g < 0 favours frequent classes (it removed rare-class false alarms on GraSP, g = -0.5), g = 0 is off.

The search is exact: the penalty only depends on whether a class was already used in the frame, so a dynamic program over the set of used classes
(2^C states) finds the best labelling without enumerating C^n combinations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORBIDDEN = -1e3
EPS = 1e-9


@dataclass
class FrameDecoder:
    class_names: list[str]
    duplicate_penalty: np.ndarray
    log_prior: np.ndarray | None = None
    prior_shift: float = 0.0

    def __post_init__(self) -> None:
        n = len(self.class_names)
        self.duplicate_penalty = np.asarray(self.duplicate_penalty, dtype=float)
        assert self.duplicate_penalty.shape == (n,), "one duplicate penalty per class"
        if self.log_prior is not None:
            self.log_prior = np.asarray(self.log_prior, dtype=float)
            assert self.log_prior.shape == (n,), "one log prior per class"

    @classmethod
    def from_stats(cls, stats: dict | str | Path, prior_shift: float | None = None) -> "FrameDecoder":
        """Builds a decoder from a statistics file written by `python -m grasp_pipeline.decode_stats` (see decode_stats/)."""
        if not isinstance(stats, dict):
            stats = json.loads(Path(stats).read_text())
        names = list(stats["class_names"])
        if stats["rule"] == "hard":
            allowed = set(stats["allowed_duplicates"])
            penalty = np.array([0.0 if n in allowed else FORBIDDEN for n in names])
        elif stats["rule"] == "soft":
            has, dup = np.asarray(stats["frames_with_class"], float), np.asarray(stats["frames_with_duplicate"], float)
            penalty = np.log((dup + 1.0) / (has + 2.0))
        else:
            raise ValueError(f"unknown duplicate rule {stats['rule']!r}")
        shift = float(stats.get("prior_shift", 0.0) if prior_shift is None else prior_shift)
        counts = np.asarray(stats["class_counts"], float)
        log_prior = np.log(counts / counts.sum()) if shift != 0.0 else None
        return cls(names, penalty, log_prior, shift)

    def _scores(self, beliefs: np.ndarray) -> np.ndarray:
        logp = np.log(np.asarray(beliefs, dtype=float) + EPS)
        if self.log_prior is not None and self.prior_shift != 0.0:
            logp = logp - self.prior_shift * self.log_prior[None, :]
        return logp

    def decode(self, beliefs: np.ndarray) -> np.ndarray:
        """(instances, classes) beliefs of the instruments of ONE frame -> (instances,) class indices."""
        scores = self._scores(beliefs)
        n, n_classes = scores.shape
        if n == 1:
            return scores.argmax(axis=1)
        states = 1 << n_classes
        best = np.full(states, -np.inf)
        best[0] = 0.0
        back: list[dict[int, tuple[int, int]]] = []
        for i in range(n):
            nxt = np.full(states, -np.inf)
            choice: dict[int, tuple[int, int]] = {}
            for mask in np.nonzero(np.isfinite(best))[0]:
                for c in range(n_classes):
                    bit = 1 << c
                    value = best[mask] + scores[i, c] + (self.duplicate_penalty[c] if mask & bit else 0.0)
                    new_mask = mask | bit
                    if value > nxt[new_mask]:
                        nxt[new_mask] = value
                        choice[new_mask] = (int(mask), c)
            best = nxt
            back.append(choice)
        mask = int(np.argmax(best))
        labels = np.zeros(n, dtype=int)
        for i in range(n - 1, -1, -1):
            mask, labels[i] = back[i][mask]
        return labels

    def decode_frames(self, beliefs: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """(instances, classes) beliefs of many frames and the frame id of each instance -> (instances,) class indices."""
        groups = np.asarray(groups)
        out = np.zeros(len(groups), dtype=int)
        for g in np.unique(groups):
            idx = np.nonzero(groups == g)[0]
            out[idx] = self.decode(beliefs[idx])
        return out
