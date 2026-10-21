"""C007: write SHA256 checksums of the raw Kaggle files and mark them read-only.

Run once after downloading a new raw dataset (e.g. Part 2):
    .venv\\Scripts\\python.exe scripts\\make_manifest.py data\\<dataset-folder>
"""
import hashlib
import os
import stat
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(folder):
    names = sorted(f for f in os.listdir(folder) if f.endswith(".parquet"))
    with open(os.path.join(folder, "MANIFEST.sha256"), "w", encoding="utf-8", newline="\n") as out:
        for name in names:
            path = os.path.join(folder, name)
            out.write(f"{sha256(path)}  {name}\n")
            os.chmod(path, stat.S_IREAD)
    print(f"{len(names)} files hashed and set read-only in {folder}")


if __name__ == "__main__":
    main(sys.argv[1])
