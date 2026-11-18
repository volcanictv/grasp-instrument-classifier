import json

from grasp_pipeline.reproduce_readout import ROOT, compute


def test_the_saved_beliefs_give_the_stored_numbers():
    expected = json.loads((ROOT / "reproduce" / "expected.json").read_text())
    assert compute() == expected


def test_headline_readout_numbers_of_the_paper():
    # numbers the research pipeline printed independently (scripts/frame_decode_eval.py and the EndoVis scoring), instance level, in percent
    result = compute()
    grasp = result["grasp"]
    assert [grasp["revisit, argmax"][s]["accuracy"] for s in ("42", "43", "44")] == [96.12, 96.40, 96.43]
    shifted = grasp["revisit + decode + prior shift -0.5 (paper row)"]
    assert [shifted[s]["accuracy"] for s in ("42", "43", "44")] == [96.82, 96.89, 96.96] and shifted["mean"]["accuracy"] == 96.89
    endovis = result["endovis18"]
    assert [endovis["revisit, argmax"][s]["accuracy"] for s in ("42", "43", "44")] == [89.52, 89.10, 87.23]
    decoded = endovis["revisit + decode (soft penalties; paper row)"]
    assert [decoded[s]["accuracy"] for s in ("42", "43", "44")] == [89.59, 89.04, 88.34]
