"""Scoring and trend helpers."""
import numpy as np
from sklearn.metrics import d2_tweedie_score

POWER = 1.5  # competition metric: Tweedie D2 with power 1.5


def tweedie_d2(y, pred, weight=None):
    """Tweedie D2 against the mean-of-y baseline (sklearn definition)."""
    return d2_tweedie_score(y, pred, sample_weight=weight, power=POWER)


def _simplex(n, steps):
    if n == 1:
        yield (steps,)
        return
    for k in range(steps + 1):
        for rest in _simplex(n - 1, steps - k):
            yield (k,) + rest


def blend_weights(folds, step=0.05):
    """C021: non-negative blend weights summing to 1, chosen on a grid to maximise mean level-calibrated D2.

    folds: list of {"actual": array, "preds": {model: array}}; every fold has the same models.
    Level calibration (rescaling the blend to the fold's actual total) keeps the choice about risk ranking;
    the level is set separately by balancing, the large-loss load and the trend share.
    """
    names = list(folds[0]["preds"])
    steps = round(1 / step)
    best, best_w = -np.inf, None
    for combo in _simplex(len(names), steps):
        w = np.array(combo) / steps
        score = 0.0
        for f in folds:
            pred = sum(wi * f["preds"][n] for wi, n in zip(w, names))
            score += tweedie_d2(f["actual"], pred * f["actual"].sum() / pred.sum())
        if score > best:
            best, best_w = score, w
    return {n: round(float(v), 10) for n, v in zip(names, best_w)}


def rescale_trend(pred, full_factor, from_share, to_share):
    """C020: re-express a prediction made with full_factor ** from_share as full_factor ** to_share."""
    return np.asarray(pred) * full_factor ** (to_share - from_share)


def trend_factor(years, pure_premium, exposure, target_year):
    """C009: exposure-weighted log-linear trend, extrapolated from the weighted mean training year.

    Models pooled over several years predict roughly the weighted-average year's level; this factor
    moves that level to target_year.
    """
    years = np.asarray(years, dtype=float)
    w = np.asarray(exposure, dtype=float)
    slope = np.polyfit(years, np.log(pure_premium), 1, w=np.sqrt(w))[0]
    mean_year = np.average(years, weights=w)
    return float(np.exp(slope * (target_year - mean_year)))
