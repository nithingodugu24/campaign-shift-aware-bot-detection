from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .utils import dump_json, sha256

PROFILE = ["followers", "following", "statuses", "listed", "verified", "default_profile", "description_length"]
EVENT_COLUMNS = ["event_id", "account_id", "timestamp", "object_id", "action"]


@dataclass
class Dataset:
    accounts: pd.DataFrame
    events: pd.DataFrame
    edges: pd.DataFrame
    metadata: dict
    source_universe: set | None = None

    @property
    def name(self):
        return self.metadata["name"]

    def subset(self, ids):
        ids = set(map(str, ids))
        return Dataset(self.accounts[self.accounts.account_id.isin(ids)].copy().reset_index(drop=True),
                       self.events[self.events.account_id.isin(ids)].copy(),
                       self.edges[self.edges.source.isin(ids) & self.edges.target.isin(ids)].copy(),
                       self.metadata.copy(), self.source_universe)


def load_dataset(folder, audit_dir=None, require_labels=True):
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("group_type") not in {"verified_campaign", "proxy_collection", "synthetic", "unavailable"}:
        raise ValueError("metadata.group_type must declare verified_campaign, proxy_collection, synthetic or unavailable")
    a = pd.read_csv(folder / "accounts.csv", dtype={"account_id": str, "group_id": str})
    if not require_labels and "label" not in a:
        a["label"] = -1
    if not {"account_id", "label"}.issubset(a):
        raise ValueError("accounts.csv requires account_id and label")
    if a.account_id.isna().any() or a.account_id.duplicated().any():
        raise ValueError("Account IDs must be present and unique; resolve duplicates before training")
    a["label"] = pd.to_numeric(a.label, errors="raise")
    if not a.label.isin([0, 1] if require_labels else [-1, 0, 1]).all():
        raise ValueError("Only explicitly mapped human=0 and bot=1 labels are accepted")
    if "group_id" not in a:
        a["group_id"] = "unavailable"
    if a.group_id.isna().any():
        raise ValueError("Group IDs must be complete, or declare unavailable")
    for col in PROFILE:
        if col not in a:
            a[col] = np.nan
        a[col] = pd.to_numeric(a[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for col in ["prediction_at", "profile_observed_at"]:
        a[col] = pd.to_datetime(a[col], utc=True, errors="coerce", format="mixed") if col in a else pd.Series(pd.NaT, index=a.index, dtype="datetime64[ns, UTC]")
    epath = folder / "events.csv"
    e = pd.read_csv(epath, dtype={"account_id": str, "object_id": str, "event_id": str}) if epath.exists() else pd.DataFrame(columns=EVENT_COLUMNS)
    if not set(EVENT_COLUMNS).issubset(e):
        raise ValueError("events.csv requires " + ", ".join(EVENT_COLUMNS))
    before = len(e)
    e["timestamp"] = pd.to_datetime(e.timestamp, utc=True, errors="coerce", format="mixed")
    bad = e.timestamp.isna() | e.account_id.isna() | e.object_id.isna() | ~e.account_id.isin(a.account_id)
    invalid = int(bad.sum())
    e = e.loc[~bad].drop_duplicates(["event_id", "account_id", "object_id", "action"]).copy()
    duplicates = before - invalid - len(e)
    cutoffs = a.set_index("account_id").prediction_at
    cutoff = e.account_id.map(cutoffs)
    future = cutoff.notna() & (e.timestamp > cutoff)
    future_count = int(future.sum())
    e = e.loc[~future].reset_index(drop=True)
    edges_path = folder / "edges.csv"
    edges = pd.read_csv(edges_path, dtype=str) if edges_path.exists() else pd.DataFrame(columns=["source", "target", "relation"])
    if not {"source", "target", "relation"}.issubset(edges):
        raise ValueError("edges.csv requires source,target,relation")
    edges = edges[edges.source.isin(a.account_id) & edges.target.isin(a.account_id)].drop_duplicates().copy()
    if not edges.relation.isin(["follows"]).all():
        raise ValueError("Only the documented directed follows relation is currently supported")
    audit = dict(name=metadata["name"], data_kind=metadata["data_kind"], accounts=len(a),
                 humans=int((a.label == 0).sum()), bots=int((a.label == 1).sum()),
                 events=len(e), unique_account_events=int(e.groupby("account_id").event_id.nunique().sum()), edges=len(edges), groups=int(a.group_id.nunique()), group_type=metadata["group_type"],
                 excluded_invalid_events=invalid, duplicate_events=duplicates, excluded_future_events=future_count,
                 timestamp_start=str(e.timestamp.min()), timestamp_end=str(e.timestamp.max()),
                 input_sha256={p.name: sha256(p) for p in folder.glob("*") if p.is_file()})
    if audit_dir:
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        dump_json(audit_dir / "audit.json", audit)
        a[PROFILE].isna().mean().rename("missing_fraction").to_csv(audit_dir / "missingness.csv")
        a.groupby(["group_id", "label"]).size().rename("accounts").to_csv(audit_dir / "group_class_counts.csv")
        a.to_csv(audit_dir / "accounts.csv", index=False)
        e.to_csv(audit_dir / "events.csv", index=False)
        edges.to_csv(audit_dir / "edges.csv", index=False)
    universe_path = folder / metadata.get("source_universe_ids", "source_universe_ids.csv")
    universe = set(pd.read_csv(universe_path, dtype=str).account_id) if universe_path.exists() else set(a.account_id)
    return Dataset(a.reset_index(drop=True), e, edges, metadata, universe)
