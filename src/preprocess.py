from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_audit import PROFILE


def account_features(dataset, rhythm=True):
    a = dataset.accounts
    profile = a[PROFILE].to_numpy(dtype=float, copy=True)
    # Stable monotone transforms; no fitted population statistics here.
    for j, name in enumerate(PROFILE):
        if name in {"followers", "following", "statuses", "listed", "description_length"}:
            profile[:, j] = np.log1p(np.maximum(0, profile[:, j]))
    x = np.column_stack([profile, np.isnan(profile).astype(float)])
    names = PROFILE + [f"{n}_missing" for n in PROFILE]
    if not rhythm:
        return x, names
    lookup = {k: frame.drop_duplicates("event_id") for k, frame in dataset.events.groupby("account_id", sort=False)}
    features = []
    for account in a.account_id:
        events = lookup.get(account)
        r = np.zeros(168 + 9, dtype=float)
        if events is None or len(events) == 0:
            r[-2:] = 1
        else:
            times = events.timestamp.sort_values()
            bins = times.dt.dayofweek.to_numpy() * 24 + times.dt.hour.to_numpy()
            r[:168] = np.bincount(bins, minlength=168) / len(times)
            seconds = times.dt.as_unit("ns").astype("int64").to_numpy() / 1e9
            delta = np.diff(seconds)
            r[168] = np.log1p(len(times))
            r[169] = np.log1p(seconds[-1] - seconds[0])
            if len(delta):
                r[170] = np.log1p(np.median(delta))
                r[171] = np.std(delta) / (np.mean(delta) + 1e-6)
                r[172] = np.mean(delta <= 60)
                r[173] = -np.sum(r[:168][r[:168] > 0] * np.log(r[:168][r[:168] > 0]))
            else:
                r[-2] = 1
            if len(times) >= 8 and seconds[-1] - seconds[0] >= 86400:
                # Fixed 168-bin weekly spectrum avoids allocating years of hourly zeros.
                power = np.abs(np.fft.rfft(r[:168] - r[:168].mean()))**2
                r[174] = power[1:].max() / (power[1:].sum() + 1e-12)
            else:
                r[-1] = 1
        features.append(r)
    names += [f"weekly_hour_{i}" for i in range(168)] + ["log_event_count", "log_span_seconds", "log_median_gap", "gap_cv", "burst_fraction", "weekly_entropy", "spectral_concentration", "gap_missing", "spectrum_missing"]
    return np.column_stack([x, np.asarray(features)]), names


def fit_preprocessor(x_train):
    pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    pipeline.fit(x_train)
    return pipeline
