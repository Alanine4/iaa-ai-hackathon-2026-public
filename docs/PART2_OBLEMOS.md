# Part 2 work by OBLEMOS (branch `part2-oblemos`)

This branch is built on `main` at P2-S13 (0.15577), and it is merged with branch `oblemos-part1` (the Part 1 pipeline, in
`oblemos_part1/`). Change ids C048-C055 are in `docs/CHANGELOG.md`, and submissions in `docs/SUBMISSIONS.md`.

## Result

**P2-S14 = 0.15781 public** (+0.00204 vs P2-S13). It is P2-S13 with the buildings line replaced by
**0.4 × OBLEMOS Part 1 GAM + 0.6 × pooled_w1 CatBoost**, kept at the P2-S13 buildings total. Motor and contents are
unchanged. File: `submissions/part2_candidate_P2-S14_p2_s13_oblemos_buildings.zip`.

The buildings blend gains on all six C044 views against the P2-S13 recipe:

| View | Gain |
|---|---|
| Time fold, raw | +0.0068 |
| Grouped, raw | +0.0018 |
| Time fold, test mix | +0.0052 |
| Grouped, test mix | +0.0058 |
| Time fold, NB | +0.0048 |
| Grouped, NB | +0.0068 |

It passes the testmix gate and the C053 strict rule.

## Reproduce P2-S14

From the repository root, after the pooled runs of P2-S13 exist (`output/emerging/pooled_w1/`, buildings cat and lgbm):

```
.venv\Scripts\python.exe -I -u scripts\oblemos_emerging.py buildings
.venv\Scripts\python.exe -I -u scripts\blend_peril.py --lines buildings --name p2_s14 --extra oblemos:ob_glm,oblemos:ob_gam --out xrun_oblemos
```

The first command runs the OBLEMOS GLM/GAM on the buildings line, in about 2 minutes. The second runs the gated blend
search and writes `work/p2_s14.zip`, which is identical to the submitted file (checked 2026-10-29).

## What was tried

| Id | Experiment | Outcome |
|---|---|---|
| C048-C049 | Transfer model (developed-only bases re-levelled; emerging corrections) | Contents probe 0.15483 on P2-S12 (noise); corrections all lost |
| C050 | Test fix for paths with a space | 297 → 317 tests pass |
| C051 | Property level probes (buildings NB x0.85, contents NB x1.15); emerging age correction | 0.15504 / 0.15513, lost; age rejected on the NB views |
| C052-C053 | Per-peril frequency x severity GLM | Contents 0.15560 and motor GLM swap 0.15563, both within noise; buildings weight 0. Added the strict rule: no loss on the time-fold test-mix and NB views |
| **C054** | **OBLEMOS Part 1 GLM/GAM on the emerging market (`scripts/oblemos_emerging.py`)** | **Buildings adopted: P2-S14 0.15781**; contents and motor not adopted |
| C055 | OBLEMOS GLM/GAM with developed rows at weight 0.3 / 0.1; buildings mix variants | Worse; adopted mix kept |

Lesson: on the public board, only candidates that gained on all six views improved the score.
