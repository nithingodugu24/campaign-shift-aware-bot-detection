from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PARTITIONS = ["train", "validation", "calibration", "test"]


class UnsupportedProtocol(ValueError):
    pass


def make_split(dataset, protocol, seed, fractions=(.5, .15, .2, .15)):
    a = dataset.accounts
    if len(a) < 16:
        raise UnsupportedProtocol("Need at least 16 accounts for four separate partitions")
    assignment = pd.Series(index=a.index, dtype=str)
    if protocol == "P0":
        remaining = a.index.to_numpy()
        for i, part in enumerate(PARTITIONS[:-1]):
            try:
                take, remaining = train_test_split(remaining, train_size=fractions[i] / sum(fractions[i:]),
                                                   random_state=seed + i, stratify=a.loc[remaining, "label"])
            except ValueError as exc:
                raise UnsupportedProtocol(f"Insufficient class support for stratified P0: {exc}") from exc
            assignment.loc[take] = part
        assignment.loc[remaining] = "test"
    elif protocol in {"P1", "P2"}:
        if dataset.metadata["group_type"] == "unavailable":
            raise UnsupportedProtocol("Independent groups are unavailable")
        groups = a.group_id.unique().to_numpy(copy=True) if hasattr(a.group_id.unique(), "to_numpy") else a.group_id.unique().copy()
        if len(groups) < 8:
            raise UnsupportedProtocol("Need at least eight independent declared groups for this four-way protocol")
        # Selection uses only source group/class support, never model test outcomes.
        rng = np.random.default_rng(seed + (10000 if protocol == "P2" else 0))
        found = False
        for _ in range(500):
            rng.shuffle(groups)
            sizes = np.maximum(1, np.floor(np.array(fractions) * len(groups)).astype(int))
            sizes[0] += len(groups) - sizes.sum()
            offset = 0
            for part, size in zip(PARTITIONS, sizes):
                assignment.loc[a.group_id.isin(groups[offset:offset + size])] = part
                offset += size
            if all(a.loc[assignment == p, "label"].nunique() == 2 for p in PARTITIONS):
                found = True
                break
        if not found:
            raise UnsupportedProtocol("No four-way group-disjoint split with both classes; collection labels confound groups")
    elif protocol == "P3":
        if not dataset.metadata.get("chronology_supported", False):
            raise UnsupportedProtocol("Historical profile availability and valid prediction times are not established")
        if a.prediction_at.isna().any() or a.profile_observed_at.isna().any():
            raise UnsupportedProtocol("P3 requires per-account prediction_at and profile_observed_at")
        if (a.profile_observed_at > a.prediction_at).any():
            raise UnsupportedProtocol("A profile snapshot occurs after its prediction cutoff")
        # Equal prediction times are never split across chronological partitions.
        times = np.sort(a.prediction_at.unique())
        if len(times) < 8:
            raise UnsupportedProtocol("Too few distinct prediction times")
        boundaries = [times[min(len(times) - 1, int(f * len(times)))] for f in np.cumsum(fractions)[:-1]]
        assignment[:] = "test"
        assignment.loc[a.prediction_at < boundaries[2]] = "calibration"
        assignment.loc[a.prediction_at < boundaries[1]] = "validation"
        assignment.loc[a.prediction_at < boundaries[0]] = "train"
    else:
        raise UnsupportedProtocol(f"Unknown split protocol {protocol}")
    manifest = a[["account_id", "group_id", "label", "prediction_at"]].copy()
    manifest["partition"] = assignment
    validate_split(dataset, manifest, protocol)
    return manifest


def validate_split(dataset, manifest, protocol):
    if manifest.account_id.duplicated().any() or set(manifest.account_id) != set(dataset.accounts.account_id):
        raise AssertionError("Every eligible account must occur exactly once")
    if manifest.partition.isna().any() or set(manifest.partition) != set(PARTITIONS):
        raise AssertionError("Missing or invalid split partition")
    for p in PARTITIONS:
        if manifest.loc[manifest.partition == p, "label"].nunique() < 2:
            raise UnsupportedProtocol(f"{p} has fewer than two classes; classifier or comparison not estimable")
    if protocol in {"P1", "P2"} and manifest.groupby("group_id").partition.nunique().max() != 1:
        raise AssertionError("Group leakage")
    if protocol == "P3":
        for left, right in zip(PARTITIONS, PARTITIONS[1:]):
            if manifest.loc[manifest.partition == left, "prediction_at"].max() >= manifest.loc[manifest.partition == right, "prediction_at"].min():
                raise AssertionError("Chronological overlap")


def split_summary(manifest):
    return manifest.groupby("partition").agg(accounts=("account_id", "size"), groups=("group_id", "nunique"), bot_fraction=("label", "mean"))
