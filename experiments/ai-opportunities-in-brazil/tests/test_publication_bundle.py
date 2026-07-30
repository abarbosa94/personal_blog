from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT / "publication"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publication_manifest_matches_tracked_files() -> None:
    manifest = json.loads(
        (PUBLICATION / "source-manifest.json").read_text(encoding="utf-8")
    )
    names = []
    for item in manifest["files"]:
        path = PUBLICATION / item["published_file"]
        names.append(item["published_file"])
        assert path.is_file()
        assert path.stat().st_size == item["bytes"]
        assert sha256(path) == item["sha256"]
    assert len(names) == len(set(names))


def test_sha256s_matches_manifest() -> None:
    manifest = json.loads(
        (PUBLICATION / "source-manifest.json").read_text(encoding="utf-8")
    )
    expected = {
        item["published_file"]: item["sha256"] for item in manifest["files"]
    }
    observed = {}
    for line in (PUBLICATION / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        checksum, name = line.split("  ", maxsplit=1)
        observed[name] = checksum
    assert observed == expected
