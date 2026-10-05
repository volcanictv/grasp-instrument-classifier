"""Locates and, when a download URL is published, fetches the pipeline's weights, verifying every file against weights_manifest.json (size and sha256).

    python -m grasp_pipeline.weights            lists every file, whether it is present and whether its checksum matches
    python -m grasp_pipeline.weights --fetch    downloads what is missing and has a published URL

The base SAM2.1-large checkpoint has a public URL. The GraSP fine-tuned weights have one once they are published; until then the manifest's
`url` is null and the files must be copied into place. SAM3's base weights are not in the manifest: they come from Hugging Face (facebook/sam3,
a gated model that needs `huggingface-cli login` and an accepted licence).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PACKAGE_ROOT / "weights_manifest.json"


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    return json.loads(Path(path).read_text())["files"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status(entry: dict, root: Path = PACKAGE_ROOT) -> str:
    path = root / entry["path"]
    if not path.exists():
        return "missing"
    if path.stat().st_size != entry["bytes"]:
        return "wrong size"
    return "ok" if sha256(path) == entry["sha256"] else "checksum mismatch"


def fetch(entry: dict, root: Path = PACKAGE_ROOT) -> Path:
    path = root / entry["path"]
    if path.exists() and status(entry, root) == "ok":
        return path
    if not entry.get("url"):
        raise FileNotFoundError(f"{entry['path']} is not present and has no published URL yet; copy it into {root}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(entry["url"]) as response, open(tmp, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    tmp.replace(path)
    if status(entry, root) != "ok":
        path.unlink()
        raise OSError(f"{entry['path']} failed its checksum after download")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to entries whose path contains one of these strings")
    args = ap.parse_args()
    bad = 0
    for entry in load_manifest():
        if args.only and not any(s in entry["path"] for s in args.only):
            continue
        state = status(entry)
        if args.fetch and state != "ok" and entry.get("url"):
            fetch(entry)
            state = status(entry)
        print(f"{state:<18}{entry['bytes'] / 1e6:>9.1f} MB  {entry['path']}  ({entry['role']})")
        bad += state != "ok"
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
