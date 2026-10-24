"""A declared multi-corpus collection holdout; never called verified campaign holdout."""
from pathlib import Path
import pandas as pd

from src.data_audit import load_dataset
from src.utils import dump_json
from scripts.prepare_public_data import stable_order


def main():
    root = Path("data/processed")
    accounts, events, edges, universes, sources = [], [], [], set(), []
    for name in ["cresci2017", "cresci2015", "gilani-2017"]:
        data = load_dataset(root / name)
        a = data.accounts.copy()
        if name == "gilani-2017":
            # Predeclared class-stratified profile-only sample; no tuning on model outcomes.
            a["_order"] = a.account_id.map(lambda uid: stable_order(uid, 2026))
            a = a.sort_values("_order").groupby("label", group_keys=False).head(120).drop(columns="_order")
            a["group_id"] = "gilani2017_collection"
        else:
            a["group_id"] = name + ":" + a.group_id
        accounts.append(a)
        events.append(data.events)
        edges.append(data.edges)
        universes.update(data.source_universe)
        sources.append(data.metadata)
    a = pd.concat(accounts, ignore_index=True)
    duplicated = set(a.loc[a.account_id.duplicated(keep=False), "account_id"])
    a = a[~a.account_id.isin(duplicated)].reset_index(drop=True)
    valid = set(a.account_id)
    e = pd.concat(events, ignore_index=True)
    e = e[e.account_id.isin(valid)].drop_duplicates(["event_id", "account_id", "object_id", "action"])
    ed = pd.concat(edges, ignore_index=True)
    ed = ed[ed.source.isin(valid) & ed.target.isin(valid)].drop_duplicates()
    out = root / "collection_pool"
    out.mkdir(exist_ok=True)
    a.to_csv(out / "accounts.csv", index=False)
    e.to_csv(out / "events.csv", index=False)
    ed.to_csv(out / "edges.csv", index=False)
    pd.DataFrame({"account_id": sorted(universes)}).to_csv(out / "source_universe_ids.csv", index=False)
    dump_json(out / "metadata.json", dict(name="multi_corpus_collection_pilot", data_kind="real", group_type="proxy_collection", chronology_supported=False,
        source=sources, license="Source research-use restrictions retained in each processed source directory",
        source_universe_ids="source_universe_ids.csv", group_definition="Original Cresci collection categories plus one Gilani collection, prefixed by dataset",
        limitations="Proxy collection shift, not verified campaign shift. Cresci 2015/2017 collections overlap historically; all duplicated sampled account IDs removed, but category dependence remains. Gilani has no temporal events, so event availability is confounded with collection. Few independent human collection units imply weak statistical power. No chronological profile history."))
    dump_json(out / "preparation_audit.json", dict(excluded_duplicate_sampled_ids=len(duplicated), accounts=len(a), events=len(e), groups=a.group_id.nunique()))
    print(a.groupby("group_id").label.agg(["count", "mean"]).to_string())


if __name__ == "__main__":
    main()
