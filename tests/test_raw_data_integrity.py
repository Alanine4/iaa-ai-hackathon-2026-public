"""C007: raw competition files must stay byte-identical to the versions downloaded from Kaggle."""
import hashlib
import os

import pytest

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "iaa-ai-assisted-hackathon-2026-part-1")
MANIFEST = os.path.join(RAW, "MANIFEST.sha256")


def read_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return dict(line.split()[::-1] for line in f if line.strip())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_manifest_lists_every_parquet_file():
    on_disk = {f for f in os.listdir(RAW) if f.endswith(".parquet")}
    assert set(read_manifest()) == on_disk


@pytest.mark.parametrize("name", sorted(f for f in os.listdir(RAW) if f.endswith(".parquet")))
def test_raw_file_matches_manifest_hash(name):
    assert sha256(os.path.join(RAW, name)) == read_manifest()[name]
