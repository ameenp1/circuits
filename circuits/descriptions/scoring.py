"""Scoring utilities for explanation evaluation."""

import numpy as np
from sklearn import linear_model


def compute_correlation_and_rsquared(
    all_true: list[float], all_pred: list[float]
) -> tuple[float | None, float | None]:
    """Compute Pearson correlation and linear regression R-squared."""
    if not all_true or not all_pred or len(all_true) != len(all_pred):
        return None, None

    true_arr = np.array(all_true)
    pred_arr = np.array(all_pred)

    # Pearson correlation
    if np.std(true_arr) == 0 or np.std(pred_arr) == 0:
        score = None
    else:
        score = float(np.corrcoef(true_arr, pred_arr)[0, 1])
        if np.isnan(score):
            score = None

    # R-squared via linear regression
    try:
        reg = linear_model.LinearRegression()
        X = pred_arr.reshape(-1, 1)
        reg.fit(X, true_arr)
        rsquared = float(reg.score(X, true_arr))
        if np.isnan(rsquared):
            rsquared = None
    except Exception:
        rsquared = None

    return score, rsquared
