"""C018: per-line hyper-parameters chosen by time-based CV (`scripts/tune_cv.py`, `docs/cv_trees.md`).

A tuned setting replaces the default only if it beats it by at least 0.002 in mean level-calibrated D2
over the 2023 and 2024 folds; otherwise the model defaults in `models.py` are kept (empty dict).
"""
import copy

TUNED = {
    "motor": {
        # CV d2_cal 0.0832 vs default 0.0757
        "lgbm": dict(num_leaves=15, min_child_samples=6000, learning_rate=0.03, colsample_bytree=0.5,
                     subsample=0.7, reg_lambda=10.0),
        # CV d2_cal 0.0824 vs default 0.0793
        "xgb": dict(max_depth=2, min_child_weight=1000, learning_rate=0.03, colsample_bytree=0.5,
                    subsample=0.9, reg_lambda=0.0),
    },
    "buildings": {
        # trees: best tuned setting within 0.0001 of the defaults, defaults kept
        # CV d2_cal 0.1017 vs default (8 knots) 0.0990 (`docs/cv_linear.md`)
        "gam": dict(n_knots=5),
    },
    "contents": {
        # CV d2_cal 0.0911 vs default 0.0888; LightGBM gain 0.0012 is below the threshold
        "xgb": dict(max_depth=2, min_child_weight=50, learning_rate=0.03, colsample_bytree=0.8,
                    subsample=0.7, reg_lambda=0.0),
    },
    # C036: emerging market (Part 2); model defaults until tuned on emerging data.
    # C046: emerging-only search (`tune_cv.py trees <line> --market emerging`, `docs/cv_emerging_trees.md`); a config
    # replaces the default only if it beats it on the raw time fold, the raw grouped mean AND the test-mix grouped mean
    # (d2_cal); among those the best raw grouped mean. Used by emerging-only fits only (pooled fits keep defaults).
    "emerging-motor": {
        # time 0.07234 vs 0.07199, grouped 0.06547 vs 0.06370, test-mix grouped 0.05057 vs 0.04304,
        # NB grouped 0.07102 vs 0.06830; xgb and cat: no config beat the defaults on all three
        "lgbm": dict(num_leaves=7, min_child_samples=1000, learning_rate=0.05, colsample_bytree=0.8, subsample=0.8),
    },
    # C046: no lgbm / xgb / cat config beat the defaults on all three (every one loses the 2023 -> 2024 time fold)
    "emerging-buildings": {},
    "emerging-contents": {
        # time 0.07115 vs 0.07114 (a tie in practice), grouped 0.07448 vs 0.07299, test-mix grouped 0.11605 vs
        # 0.11364, NB grouped 0.13660 vs 0.13682
        "xgb": dict(max_depth=3, min_child_weight=10),
        # time 0.08696 vs 0.08146, grouped 0.08815 vs 0.07820, test-mix grouped 0.13014 vs 0.12500,
        # NB grouped 0.14100 vs 0.13778; lgbm: no config beat the defaults on all three
        "cat": dict(depth=4, l2_leaf_reg=10),
    },
}


def for_model(line, name):
    return copy.deepcopy(TUNED[line].get(name, {}))
