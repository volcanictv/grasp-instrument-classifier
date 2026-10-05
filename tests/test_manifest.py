"""The weights manifest is consistent with the config that reads it."""
import json
from pathlib import Path

import yaml

from grasp_pipeline import weights

ROOT = Path(__file__).resolve().parent.parent


def test_manifest_entries_are_well_formed():
    entries = weights.load_manifest()
    assert entries
    seen = set()
    for e in entries:
        assert {"path", "bytes", "sha256", "role", "url"} <= set(e)
        assert len(e["sha256"]) == 64 and e["bytes"] > 0
        assert e["path"] not in seen
        seen.add(e["path"])


def test_config_checkpoints_are_in_manifest():
    paths = {e["path"] for e in weights.load_manifest()}
    cfg = yaml.safe_load((ROOT / "evidential_config.yaml").read_text())
    assert len(cfg["members"]) == 4
    for m in cfg["members"]:
        assert m["checkpoint"] in paths, m["checkpoint"]


def test_all_three_seeds_have_four_members():
    paths = [e["path"] for e in weights.load_manifest()]
    for seed in (42, 43, 44):
        assert len([p for p in paths if f"evidential_armN_seed{seed}/" in p]) == 4
