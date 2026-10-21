"""Scoring and trend helpers."""
import numpy as np
from sklearn.metrics import d2_tweedie_score

POWER = 1.5  # competition metric: Tweedie D2 with power 1.5


def tweedie_d2(y, pred, weight=None):
    """Tweedie D2 against the mean-of-y baseline (sklearn definition)."""
    return d2_tweedie_score(y, pred, sample_weight=weight, power=POWER)


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
