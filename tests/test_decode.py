import itertools
import json

import numpy as np
import pytest

from grasp_classifier.classes import CLASS_NAMES, ENDOVIS18_CLASS_NAMES
from grasp_classifier.decode import EPS, FORBIDDEN, FrameDecoder
from grasp_pipeline.decode_stats import build_stats, coco_frames, frame_statistics, merge
from grasp_pipeline.pipeline import InstrumentPipeline

LND = CLASS_NAMES.index("Large Needle Driver")


def brute_force(decoder: FrameDecoder, beliefs: np.ndarray) -> tuple[int, ...]:
    scores = decoder._scores(beliefs)
    best, best_score = None, -np.inf
    for combo in itertools.product(range(scores.shape[1]), repeat=scores.shape[0]):
        total = sum(scores[i, c] for i, c in enumerate(combo))
        for c in set(combo):
            total += (combo.count(c) - 1) * decoder.duplicate_penalty[c]
        if total > best_score + 1e-12:
            best, best_score = combo, total
    return best


def hard_decoder(prior_shift=0.0):
    penalty = np.array([0.0 if i == LND else FORBIDDEN for i in range(7)])
    log_prior = np.log(np.array([0.3, 0.15, 0.2, 0.2, 0.08, 0.02, 0.05]))
    return FrameDecoder(CLASS_NAMES, penalty, log_prior, prior_shift)


def soft_decoder(prior_shift=0.0):
    penalty = np.log((np.array([2, 7, 80, 3, 1, 0, 1]) + 1) / (np.array([100, 50, 90, 60, 20, 8, 15]) + 2))
    log_prior = np.log(np.array([0.3, 0.15, 0.2, 0.2, 0.08, 0.02, 0.05]))
    return FrameDecoder(CLASS_NAMES, penalty, log_prior, prior_shift)


@pytest.mark.parametrize("make,shift", [(hard_decoder, 0.0), (hard_decoder, -0.5), (soft_decoder, 0.0), (soft_decoder, -0.5), (soft_decoder, 0.7)])
def test_dynamic_program_matches_exhaustive_search(make, shift):
    decoder = make(shift)
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 4, 5):
        for _ in range(12):
            beliefs = rng.dirichlet(np.ones(7) * 0.4, size=n)
            assert tuple(decoder.decode(beliefs)) == brute_force(decoder, beliefs)


def test_a_repeated_class_is_split_unless_it_is_allowed_to_repeat():
    decoder = hard_decoder()
    two_bipolar = np.array([[0.7, 0.2, 0.0, 0.05, 0.0, 0.05, 0.0], [0.6, 0.3, 0.0, 0.05, 0.0, 0.05, 0.0]])
    assert decoder.decode(two_bipolar).tolist() == [0, 1]  # the less certain one takes its next best class
    two_driver = np.array([[0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0], [0.1, 0.0, 0.7, 0.2, 0.0, 0.0, 0.0]])
    assert decoder.decode(two_driver).tolist() == [LND, LND]


def test_a_single_instrument_is_just_the_argmax_without_a_prior_shift():
    beliefs = np.array([[0.1, 0.5, 0.2, 0.1, 0.05, 0.03, 0.02]])
    assert hard_decoder().decode(beliefs).tolist() == [1]


def test_a_negative_prior_shift_favours_frequent_classes():
    beliefs = np.array([[0.30, 0.0, 0.0, 0.0, 0.0, 0.70 - EPS, 0.0]])  # Clip Applier (rare) narrowly ahead of Bipolar Forceps (common)
    assert hard_decoder(0.0).decode(beliefs).tolist() == [5]
    assert hard_decoder(-0.5).decode(beliefs).tolist() == [0]


def test_from_stats_builds_the_hard_and_the_soft_rule():
    hard = FrameDecoder.from_stats({"class_names": CLASS_NAMES, "rule": "hard", "allowed_duplicates": ["Large Needle Driver"],
                                    "class_counts": [10, 5, 8, 9, 3, 1, 2], "prior_shift": -0.5})
    assert hard.duplicate_penalty[LND] == 0.0 and (hard.duplicate_penalty[np.arange(7) != LND] == FORBIDDEN).all()
    assert hard.prior_shift == -0.5 and hard.log_prior is not None
    soft = FrameDecoder.from_stats({"class_names": CLASS_NAMES, "rule": "soft", "frames_with_class": [100] * 7, "frames_with_duplicate": [0, 5, 50, 0, 0, 0, 0],
                                    "class_counts": [1] * 7})
    assert soft.duplicate_penalty[LND] == pytest.approx(np.log(51 / 102)) and soft.log_prior is None
    assert FrameDecoder.from_stats({"class_names": CLASS_NAMES, "rule": "soft", "frames_with_class": [1] * 7, "frames_with_duplicate": [0] * 7,
                                    "class_counts": [1] * 7}, prior_shift=-0.25).prior_shift == -0.25


def test_decode_frames_handles_each_frame_separately():
    decoder = hard_decoder()
    beliefs = np.array([[0.9, 0.1, 0, 0, 0, 0, 0], [0.8, 0.2, 0, 0, 0, 0, 0], [0.9, 0.1, 0, 0, 0, 0, 0]])
    out = decoder.decode_frames(beliefs, np.array([7, 7, 9]))
    assert out.tolist() == [0, 1, 0]


def test_the_endovis_class_set_has_ultrasound_probe_and_no_grasper():
    assert "Ultrasound Probe" in ENDOVIS18_CLASS_NAMES and "Laparoscopic Grasper" not in ENDOVIS18_CLASS_NAMES and len(ENDOVIS18_CLASS_NAMES) == 7


def test_frame_statistics_counts_instances_frames_and_repeats():
    counts, has, dup = frame_statistics([[0, 0, 2], [2, 2, 2], [1]], 3)
    assert counts == [2, 1, 4] and has == [1, 1, 2] and dup == [1, 0, 1]


def test_coco_frames_groups_by_image_and_skips_empty_boxes(tmp_path):
    doc = {"categories": [{"id": 5, "name": "a"}, {"id": 9, "name": "b"}], "images": [{"id": 1}, {"id": 2}],
           "annotations": [{"image_id": 1, "category_id": 9, "bbox": [0, 0, 10, 10]}, {"image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10]},
                           {"image_id": 2, "category_id": 5, "bbox": [0, 0, 0, 10]}, {"image_id": 2, "category_id": 5, "bbox": [0, 0, 5, 5]}]}
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(doc))
    assert sorted(map(sorted, coco_frames(path))) == [[0], [0, 1]]


def test_merge_adds_counts_and_recomputes_the_soft_penalties():
    a = build_stats([[0, 0], [0]], "grasp", "soft")
    b = build_stats([[2, 2], [2, 2]], "grasp", "soft")
    m = merge(a, b)
    assert m["class_counts"][0] == 3 and m["class_counts"][2] == 4 and m["frames_with_duplicate"][2] == 2
    assert m["duplicate_penalty"][2] == pytest.approx(np.log(3 / 4))


class RectSegmenter:
    def segment(self, image, boxes_xywh):
        out = []
        for x, y, w, h in boxes_xywh:
            m = np.zeros(image.shape[:2], bool)
            m[int(y):int(y + h), int(x):int(x + w)] = True
            out.append(m)
        return out


def test_the_pipeline_never_repeats_a_forbidden_class_when_a_decoder_is_set(synthetic_classifier, frame):
    boxes = [(10, 10, 60, 50), (100, 20, 70, 60), (180, 120, 80, 70)]
    forbid_all = FrameDecoder(CLASS_NAMES, np.full(7, FORBIDDEN))
    results = InstrumentPipeline(RectSegmenter(), synthetic_classifier, decoder=forbid_all).process_frame(frame, boxes)
    assert len({r.class_index for r in results}) == len(results) == 3


def test_the_pipeline_without_a_decoder_is_unchanged(synthetic_classifier, frame):
    boxes = [(10, 10, 60, 50), (100, 20, 70, 60)]
    plain = InstrumentPipeline(RectSegmenter(), synthetic_classifier).process_frame(frame, boxes)
    free = FrameDecoder(CLASS_NAMES, np.zeros(7))  # no penalty and no prior: the decode equals the argmax
    decoded = InstrumentPipeline(RectSegmenter(), synthetic_classifier, decoder=free).process_frame(frame, boxes)
    assert [r.class_name for r in plain] == [r.class_name for r in decoded] and not any(r.decoded for r in decoded)


def test_a_decoder_for_another_class_set_is_rejected(synthetic_classifier):
    wrong = FrameDecoder(ENDOVIS18_CLASS_NAMES, np.zeros(7))
    with pytest.raises(AssertionError):
        InstrumentPipeline(RectSegmenter(), synthetic_classifier, decoder=wrong)
