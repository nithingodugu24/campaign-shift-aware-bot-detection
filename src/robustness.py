import numpy as np
import pandas as pd


def perturb_events(dataset, kind, value, seed):
    data = dataset.subset(dataset.accounts.account_id)
    rng = np.random.default_rng(seed)
    # A physical event may reference several objects. Perturb it once, consistently.
    keys = data.events.account_id.astype(str) + ":" + data.events.event_id.astype(str)
    unique = keys.unique()
    if kind == "remove":
        retained = set(unique[rng.random(len(unique)) >= value])
        data.events = data.events.loc[keys.isin(retained)].copy()
    elif kind == "jitter":
        offsets = pd.Series(rng.uniform(-value, value, len(unique)), index=unique)
        data.events["timestamp"] = data.events.timestamp + pd.to_timedelta(keys.map(offsets).to_numpy(), unit="s")
        cutoff = data.events.account_id.map(data.accounts.set_index("account_id").prediction_at)
        mask = cutoff.notna() & (data.events.timestamp > cutoff)
        data.events.loc[mask, "timestamp"] = cutoff[mask]
    elif kind != "clean":
        raise ValueError(kind)
    return data
