"""C054: the Part 1 rule (70% team recipe + 30% OBLEMOS, private +0.0003 on Part 1) on the emerging folds, per line.

Prints the six C044 views of the P2-S13 line recipe with a fixed share of an OBLEMOS candidate (each fold's components
as in blend_runs) and the gains over the recipe. No search: the 30% share comes from the Part 1 late-submission test.
C056 `--nb-segment`: the share is applied on the new-business rows only (renewal rows keep the recipe), as the C047
segment blends; the six views still score the whole line.
Usage: .venv\Scripts\python.exe -I scripts\mix30.py <line> [--cands oblemos:ob_gam] [--shares 0.2,0.3] [--nb-segment]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("AIIAA_MARKET", "emerging")
import blend_runs as br  # noqa: E402
from blend_peril import P2S13, runs_of  # noqa: E402

KEYS = ("time", "group_mean", "time_testmix", "group_testmix", "time_nb", "group_nb")


def main(line, cands, shares, nb_segment=False):
    recipe = P2S13[line]
    frame, names = br.load_runs(line, runs_of(list(recipe) + cands))
    folds, share = br.scheme_folds(frame, names), br.nb_share_2025(line)
    base = br.blend_scores(folds, recipe, share)
    print(f"{line} recipe: " + "  ".join(f"{k} {base[k]:.5f}" for k in KEYS))
    for cand in cands:
        for a in shares:
            if nb_segment:
                for scheme in folds:
                    for f in folds[scheme]:
                        rec = sum(v * f["preds"][c] for c, v in recipe.items())
                        nb = f["new_business"].astype(bool)
                        f["preds"]["_mix"] = np.where(nb, (1 - a) * rec + a * f["preds"][cand], rec)
                s = br.blend_scores(folds, {"_mix": 1.0}, share)
            else:
                w = {c: (1 - a) * v for c, v in recipe.items()}
                w[cand] = w.get(cand, 0) + a
                s = br.blend_scores(folds, w, share)
            gains = {k: s[k] - base[k] for k in KEYS}
            print(f"{line} {cand} {a:.0%}: " + "  ".join(f"{k} {g:+.5f}" for k, g in gains.items())
                  + f"   all six >= 0: {all(g >= 0 for g in gains.values())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("line")
    ap.add_argument("--cands", default="oblemos:ob_glm,oblemos:ob_gam")
    ap.add_argument("--shares", default="0.2,0.3")
    ap.add_argument("--nb-segment", action="store_true")
    a = ap.parse_args()
    main(a.line, a.cands.split(","), [float(s) for s in a.shares.split(",")], a.nb_segment)
