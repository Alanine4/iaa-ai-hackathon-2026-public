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
    "buildings": {},  # best tuned setting within 0.0001 of the defaults
    "contents": {
        # CV d2_cal 0.0911 vs default 0.0888; LightGBM gain 0.0012 is below the threshold
        "xgb": dict(max_depth=2, min_child_weight=50, learning_rate=0.03, colsample_bytree=0.8,
                    subsample=0.7, reg_lambda=0.0),
    },
}


def for_model(line, name):
    return copy.deepcopy(TUNED[line].get(name, {}))
