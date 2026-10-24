"""C007: raw competition files must stay byte-identical to the versions downloaded from Kaggle.
C034: the Part 2 emerging folder (13 files) is checked against its own manifest as well."""
import hashlib
import os

import pytest

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FOLDERS = {"part-1": os.path.join(DATA, "iaa-ai-assisted-hackathon-2026-part-1"),
           "emerging": os.path.join(DATA, "iaa-ai-assisted-hackathon-2026-part-2", "emerging")}


def read_manifest(raw):
    with open(os.path.join(raw, "MANIFEST.sha256"), encoding="utf-8") as f:
        return dict(line.split()[::-1] for line in f if line.strip())


def parquet_files(raw):
    return sorted(f for f in os.listdir(raw) if f.endswith(".parquet"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.parametrize("folder", sorted(FOLDERS))
def test_manifest_lists_every_parquet_file(folder):
    raw = FOLDERS[folder]
    assert set(read_manifest(raw)) == set(parquet_files(raw))


def test_emerging_manifest_has_13_files():
    assert len(read_manifest(FOLDERS["emerging"])) == 13


@pytest.mark.parametrize("folder, name", [(k, f) for k in sorted(FOLDERS) for f in parquet_files(FOLDERS[k])])
def test_raw_file_matches_manifest_hash(folder, name):
    raw = FOLDERS[folder]
    assert sha256(os.path.join(raw, name)) == read_manifest(raw)[name]
