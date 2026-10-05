import numpy as np
import pytest

from grasp_classifier.classes import CLASS_NAMES
from grasp_classifier.evidential import EvidentialPrediction
from grasp_pipeline.pipeline import InstanceResult, InstrumentPipeline, semantic_map
from grasp_pipeline.scoring import aggregate, frame_class_ious, paint
from grasp_pipeline.segment import flip_boxes, xywh_to_xyxy


class RectSegmenter:
    """Stands in for SAM: each box becomes a filled rectangle."""

    def segment(self, image, boxes_xywh):
        out = []
        for x, y, w, h in boxes_xywh:
            m = np.zeros(image.shape[:2], bool)
            m[int(y):int(y + h), int(x):int(x + w)] = True
            out.append(m)
        return out


class FakeTracker:
    def __init__(self, answer="Clip Applier"):
        self.calls = 0
        self.answer = answer

    def track(self, frames, center_idx, mask, box):
        self.calls += 1
        belief = {name: 0.0 for name in CLASS_NAMES}
        belief[self.answer] = 1.0
        return EvidentialPrediction(self.answer, 0.0, belief, False, np.zeros((4, 7)), tracked=True)


BOXES = [(20, 20, 80, 60), (150, 100, 90, 70)]


def test_every_instance_gets_a_class_and_an_uncertainty(synthetic_classifier, frame):  # noqa: F811
    results = InstrumentPipeline(RectSegmenter(), synthetic_classifier).process_frame(frame, BOXES)
    assert len(results) == 2
    assert all(r.class_name in CLASS_NAMES and r.epistemic > 0 and not r.tracked for r in results)


def test_flagged_instances_are_tracked_and_unflagged_are_not(synthetic_classifier, frame):  # noqa: F811
    tracker = FakeTracker()
    always = InstrumentPipeline(RectSegmenter(), synthetic_classifier, tracker, epistemic_gate=0.0)
    results = always.process_frame(frame, BOXES, context_frames=[frame] * 3, center_idx=1)
    assert tracker.calls == 2 and all(r.tracked and r.class_name == "Clip Applier" for r in results)
    never = FakeTracker()
    InstrumentPipeline(RectSegmenter(), synthetic_classifier, never, epistemic_gate=1.0).process_frame(frame, BOXES, context_frames=[frame] * 3, center_idx=1)
    assert never.calls == 0


def test_tracking_is_skipped_without_context_frames(synthetic_classifier, frame):  # noqa: F811
    tracker = FakeTracker()
    InstrumentPipeline(RectSegmenter(), synthetic_classifier, tracker, epistemic_gate=0.0).process_frame(frame, BOXES)
    assert tracker.calls == 0


def test_the_single_pass_class_is_kept_next_to_the_corrected_one(synthetic_classifier, frame):  # noqa: F811
    results = InstrumentPipeline(RectSegmenter(), synthetic_classifier, FakeTracker(), epistemic_gate=0.0).process_frame(
        frame, BOXES, context_frames=[frame] * 3, center_idx=1)
    assert all(r.single_pass_class in CLASS_NAMES for r in results)


def test_empty_masks_are_dropped(synthetic_classifier, frame):  # noqa: F811
    empty = [np.zeros(frame.shape[:2], bool)]
    assert InstrumentPipeline(RectSegmenter(), synthetic_classifier).process_frame(frame, [BOXES[0]], masks=empty) == []


def _result(mask, cls, score):
    return InstanceResult((0, 0, 1, 1), mask, cls, CLASS_NAMES[cls], 0.0, {}, score, False, CLASS_NAMES[cls])


def test_a_higher_score_wins_an_overlap_in_the_semantic_map():
    a, b = np.zeros((10, 10), bool), np.zeros((10, 10), bool)
    a[:, :6], b[:, 4:] = True, True
    sem = semantic_map((10, 10), [_result(a, 0, 0.9), _result(b, 3, 0.4)])
    assert sem[5, 5] == 1 and sem[5, 8] == 4 and sem[5, 0] == 1


def test_a_perfect_prediction_scores_one():
    gt = np.zeros((20, 20), np.int8)
    gt[2:8, 2:8], gt[10:18, 10:18] = 1, 4
    assert frame_class_ious(gt.copy(), gt)[0] == {1: 1.0, 4: 1.0}
    result = aggregate([frame_class_ious(gt.copy(), gt)])
    assert result["mIoU"] == result["IoU"] == 1.0


def test_a_wrong_class_scores_zero_for_both_classes():
    gt = np.zeros((20, 20), np.int8)
    gt[2:8, 2:8] = 1
    pred = np.zeros_like(gt)
    pred[2:8, 2:8] = 2
    ious, gt_classes = frame_class_ious(pred, gt)
    assert ious == {1: 0.0, 2: 0.0} and gt_classes == {1}
    assert aggregate([(ious, gt_classes)])["mIoU"] == 0.0


def test_box_conversion_and_flip():
    xyxy = xywh_to_xyxy([(10, 20, 30, 40)])
    assert xyxy.tolist() == [[10, 20, 40, 60]]
    assert flip_boxes(xyxy, 100).tolist() == [[60, 20, 90, 60]]


def test_paint_orders_by_score():
    m1, m2 = np.zeros((4, 4), bool), np.zeros((4, 4), bool)
    m1[:], m2[:2] = True, True
    sem = paint((4, 4), [(m2, 2, 0.9), (m1, 1, 0.1)])
    assert sem[0, 0] == 2 and sem[3, 3] == 1
