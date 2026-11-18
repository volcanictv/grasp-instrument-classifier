import numpy as np
import torch
import yaml

from grasp_classifier.classes import ENDOVIS18_CLASS_NAMES
from grasp_classifier.evidential import EvidentialEnsembleClassifier
from grasp_classifier.models import BUILDERS
from grasp_pipeline.gt_inputs import build_inputs, decode_rle


def test_gt_inputs_writes_boxes_and_masks_per_case():
    from pycocotools import mask as mask_utils

    m = np.zeros((20, 30), np.uint8)
    m[5:10, 4:12] = 1
    rle = mask_utils.encode(np.asfortranarray(m))
    rle["counts"] = rle["counts"].decode("ascii")
    doc = {"images": [{"id": 1, "file_name": "CASE001/00001.jpg"}, {"id": 2, "file_name": "CASE002/00007.jpg"}],
           "annotations": [{"image_id": 1, "category_id": 1, "bbox": [4, 5, 8, 5], "segmentation": rle},
                           {"image_id": 1, "category_id": 2, "bbox": [0, 0, 0, 3], "segmentation": rle},
                           {"image_id": 2, "category_id": 1, "bbox": [4.4, 5.2, 7.6, 5], "segmentation": rle}]}
    out = build_inputs(doc)
    assert sorted(out) == ["CASE001", "CASE002"]
    boxes, masks = out["CASE001"]["00001.jpg"]
    assert boxes == [[4, 5, 8, 5]] and len(masks) == 1 and (masks[0] == m.astype(bool)).all()  # the empty-box annotation is skipped
    assert out["CASE002"]["00007.jpg"][0] == [[4, 5, 8, 5]]
    assert (decode_rle(rle) == m.astype(bool)).all()


def test_the_endovis_class_set_names_the_members_outputs(tmp_path):
    torch.manual_seed(0)
    members = []
    for label, size, letterbox in [("resnet50_320", 320, True), ("resnet50_224", 224, True), ("baseline", 224, False), ("letterbox_crop", 224, True)]:
        torch.save(BUILDERS["mobilenet_v3_small"](num_classes=7).state_dict(), tmp_path / f"{label}.pt")
        members.append({"checkpoint": f"{label}.pt", "model": "mobilenet_v3_small", "image_size": size, "letterbox": letterbox, "label": label})
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"class_set": "endovis18", "weight_resnet50_320": 0.4, "members": members}))
    clf = EvidentialEnsembleClassifier(cfg, device="cpu")
    assert clf.class_names == ENDOVIS18_CLASS_NAMES
    pred = clf.prediction_from_logits(np.random.default_rng(0).normal(size=(4, 7)))
    assert pred.class_name in ENDOVIS18_CLASS_NAMES and set(pred.belief) == set(ENDOVIS18_CLASS_NAMES)
