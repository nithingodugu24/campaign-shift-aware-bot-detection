from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pandas as pd

from scripts.prepare_public_data import download, BASE
from src.data_audit import EVENT_COLUMNS
from src.utils import dump_json, sha256


def prepare(name, root=Path("data")):
    url = f"{BASE}/{name}/{name}.tar.gz"
    archive = root / "raw" / f"{name}.tar.gz"
    download(archive, url)
    out = root / "processed" / name
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        labels = pd.read_csv(tar.extractfile(name + ".tsv"), sep="\t", header=None, names=["account_id", "raw_label"], dtype=str)
        lookup = dict(zip(labels.account_id, labels.raw_label))
        snapshots = json.load(tar.extractfile(name + "_tweets.json"))
    rows = []
    for record in snapshots:
        user = record["user"]
        uid = str(user.get("id_str", user["id"]))
        label = lookup.get(uid)
        if label not in {"human", "bot"}:
            continue
        rows.append(dict(account_id=uid, label=int(label == "bot"), group_id="unavailable", followers=user.get("followers_count"),
            following=user.get("friends_count"), statuses=user.get("statuses_count"), listed=user.get("listed_count"),
            verified=int(user["verified"]) if "verified" in user else None,
            default_profile=int(user["default_profile"]) if "default_profile" in user else None,
            description_length=len(user["description"]) if user.get("description") else None, prediction_at="", profile_observed_at=""))
    accounts = pd.DataFrame(rows).drop_duplicates("account_id")
    accounts.to_csv(out / "accounts.csv", index=False)
    pd.DataFrame(columns=EVENT_COLUMNS).to_csv(out / "events.csv", index=False)
    pd.DataFrame(columns=["source", "target", "relation"]).to_csv(out / "edges.csv", index=False)
    dump_json(out / "metadata.json", dict(name=name + "_snapshots", data_kind="real", group_type="unavailable", chronology_supported=False,
        source=url, license="Bot Repository academic research use; see https://botometer.osome.iu.edu/bot-repository/datasets.html",
        citation="Gilani et al. Of bots and humans (on Twitter). ASONAM 2017." if name == "gilani-2017" else "Mazza et al. RTbust. WebSci 2019.",
        archive_sha256=sha256(archive), label_mapping={"human": 0, "bot": 1},
        limitations="Only user snapshots and a record timestamp are supplied. No tweet IDs, objects or event histories; temporal and graph transfer is unsupported. No campaign annotations."))
    dump_json(out / "preparation_audit.json", dict(label_records=len(labels), user_snapshots=len(snapshots), matched_accounts=len(accounts), no_profile_accounts=len(labels) - len(accounts)))
    print(name, len(accounts), accounts.label.value_counts().to_dict())


if __name__ == "__main__":
    for name in ["gilani-2017", "cresci-rtbust-2019"]:
        prepare(name)
