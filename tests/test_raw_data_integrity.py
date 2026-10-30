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
    return sorted(f for f in os.listdir(raw) if f.endswith(".parquet")) if os.path.isdir(raw) else []


def raw_data_present(raw):
    return bool(parquet_files(raw))


def require_raw_data(folder):
    # C061: the public copy of the repository has no raw data; skip these tests there instead of failing them
    raw = FOLDERS[folder]
    if not raw_data_present(raw):
        where = os.path.relpath(raw, DATA).replace(os.sep, "/")
        pytest.skip(f"raw data not present: download it from Kaggle into data/{where}/ (see README.md, Data)")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.parametrize("folder", sorted(FOLDERS))
def test_manifest_lists_every_parquet_file(folder):
    require_raw_data(folder)
    raw = FOLDERS[folder]
    assert set(read_manifest(raw)) == set(parquet_files(raw))


def test_emerging_manifest_has_13_files():
    require_raw_data("emerging")
    assert len(read_manifest(FOLDERS["emerging"])) == 13


# C061: a folder without parquet files gets one placeholder case (name None) so that it skips with a clear message
HASH_CASES = [(k, f) for k in sorted(FOLDERS) for f in (parquet_files(FOLDERS[k]) or [None])]


@pytest.mark.parametrize("folder, name", HASH_CASES)
def test_raw_file_matches_manifest_hash(folder, name):
    require_raw_data(folder)
    raw = FOLDERS[folder]
    assert sha256(os.path.join(raw, name)) == read_manifest(raw)[name]


def test_raw_data_absent_is_detected(tmp_path):
    # C061: a folder that holds only its manifest (the public copy of the repository) counts as "no raw data"
    (tmp_path / "MANIFEST.sha256").write_text("0" * 64 + "  a.parquet\n", encoding="utf-8")
    assert not raw_data_present(str(tmp_path))
    assert not raw_data_present(str(tmp_path / "missing"))
    (tmp_path / "a.parquet").write_bytes(b"")
    assert raw_data_present(str(tmp_path))
