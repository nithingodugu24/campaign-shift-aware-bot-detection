from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import log_loss


def probabilities(p):
    p = np.asarray(p, dtype=float)
    p = np.column_stack([1 - p, p]) if p.ndim == 1 else p
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or (p < 0).any() or (p.sum(1) <= 0).any():
        raise ValueError("Expected finite nonnegative binary probabilities")
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(p_validation, y_validation):
    logits = logit(np.clip(probabilities(p_validation)[:, 1], 1e-7, 1 - 1e-7))
    result = minimize_scalar(lambda t: log_loss(y_validation, expit(logits / np.exp(t)), labels=[0, 1]), bounds=(-3, 3), method="bounded")
    return float(np.exp(result.x))


def temperature_scale(p, temperature):
    return probabilities(expit(logit(np.clip(probabilities(p)[:, 1], 1e-7, 1 - 1e-7)) / temperature))


def score_sets(p, q):
    return 1 - probabilities(p) <= q + 1e-12


def conformal_quantile(scores, alpha):
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration data must be nonempty")
    rank = math.ceil((n + 1) * (1 - alpha) - 1e-12)
    return 1.0 if rank > n else float(np.sort(scores)[max(0, rank - 1)])


@dataclass
class SetCalibrator:
    method: str
    q: float
    alpha: float
    units: int
    feasible: bool = True
    corrected_calibration_risk: float | None = None
    temperature: float = 1.
    notes: str = "Empirical evaluation only; exchangeability under campaign shift is unverified"

    def predict(self, p):
        p = temperature_scale(p, self.temperature)
        if self.method in {"confidence", "temperature"}:
            accepted = p.max(axis=1) >= self.q
            sets = np.ones_like(p, dtype=bool)
            sets[accepted] = False
            sets[np.flatnonzero(accepted), p[accepted].argmax(axis=1)] = True
            return sets
        return score_sets(p, self.q)

    def to_dict(self):
        return asdict(self)


def confidence_threshold(p, y, alpha):
    p = probabilities(p)
    # Threshold grid and minimum validation acceptance are predeclared.
    for threshold in np.linspace(.5, 1., 101):
        accepted = p.max(axis=1) >= threshold
        if accepted.sum() >= max(5, int(.05 * len(y))) and (p[accepted].argmax(1) != np.asarray(y)[accepted]).mean() <= alpha:
            return float(threshold)
    return 1.000001


def fit_calibrator(method, p_cal, y_cal, groups, alpha=.1, p_val=None, y_val=None):
    p_cal = probabilities(p_cal)
    y_cal = np.asarray(y_cal, dtype=int)
    groups = np.asarray(groups)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if method in {"confidence", "temperature"}:
        if p_val is None or y_val is None:
            raise ValueError("Confidence rules require separate validation predictions")
        temp = fit_temperature(p_val, y_val) if method == "temperature" else 1.
        q = confidence_threshold(temperature_scale(p_val, temp), y_val, alpha)
        return SetCalibrator(method, q, alpha, len(y_val), temperature=temp,
                             notes="Validation-selected descriptive abstention rule; no selective-risk guarantee")
    scores = 1 - p_cal[np.arange(len(y_cal)), y_cal]
    unique = np.unique(groups)
    if method == "account_conformal":
        return SetCalibrator(method, conformal_quantile(scores, alpha), alpha, len(scores))
    if method == "group_max_conformal":
        maxima = [scores[groups == g].max() for g in unique]
        return SetCalibrator(method, conformal_quantile(maxima, alpha), alpha, len(unique),
                             notes="Split conformal on maximum account score per group; targets entire-group coverage under exchangeable groups, not CRC mean loss")
    if method != "campaign_crc":
        raise ValueError(method)
    # Fixed increasing grid; bounded monotone group loss, B=1.
    grid = np.linspace(0, 1, 1001)
    losses = np.asarray([(scores[groups == g, None] > grid[None, :] + 1e-12).mean(axis=0) for g in unique])
    corrected = (losses.sum(axis=0) + 1.) / (len(unique) + 1.)
    eligible = np.flatnonzero(corrected <= alpha + 1e-12)
    if not len(eligible):
        return SetCalibrator(method, 1., alpha, len(unique), False, float(corrected[-1]),
                             notes=f"Infeasible finite-sample criterion: floor={1/(len(unique)+1):.6f}; full-set fallback is explicitly abstention, not achieved CRC")
    j = eligible[0]
    return SetCalibrator(method, float(grid[j]), alpha, len(unique), True, float(corrected[j]))
