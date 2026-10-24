"""Download original research archives and build auditable bounded pilot subsets.

Run from the project root: python -m scripts.prepare_public_data
No tweet text, names or screen names are copied to processed outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import tarfile
from pathlib import Path
import zipfile
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from src.data_audit import PROFILE, EVENT_COLUMNS
from src.utils import dump_json, sha256

BASE = "https://botometer.osome.iu.edu/bot-repository/datasets"
FILES = {"cresci2017": ("cresci-2017.csv.zip", f"{BASE}/cresci-2017/cresci-2017.csv.zip"),
         "cresci2015": ("cresci-2015.csv.tar.gz", f"{BASE}/cresci-2015/cresci-2015.csv.tar.gz")}


def download(path, url):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        append = offset and response.status_code == 206
        with part.open("ab" if append else "wb") as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
    part.rename(path)


def stable_order(uid, seed):
    return hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest()


def find_member(archive, basename):
    return next((name for name in archive.namelist() if Path(name).name == basename and not name.startswith("__MACOSX")), None)


def convert_inner(inner, cohort, label, account_limit, event_limit, seed):
    users_name = find_member(inner, "users.csv")
    if not users_name:
        return [], [], [], dict(cohort=cohort, reason="users.csv unavailable")
    with inner.open(users_name) as stream:
        raw = pd.read_csv(stream, dtype=str, encoding="utf-8", encoding_errors="replace", keep_default_na=False)
    if raw.id.duplicated().any():
        if raw.groupby("id").size().max() > 1:
            raw = raw.drop_duplicates("id", keep="first")
    chosen = raw.assign(_order=raw.id.map(lambda uid: stable_order(uid, seed))).sort_values("_order").head(account_limit)
    accounts = []
    mapping = {"followers": "followers_count", "following": "friends_count", "statuses": "statuses_count", "listed": "listed_count", "verified": "verified", "default_profile": "default_profile"}
    for _, row in chosen.iterrows():
        record = dict(account_id=str(row.id), label=label, group_id=cohort, prediction_at="", profile_observed_at="")
        for dest, source in mapping.items():
            value = row.get(source, "")
            record[dest] = pd.to_numeric(value, errors="coerce")
        description = row.get("description", "")
        record["description_length"] = len(description) if description and description != "NULL" else np.nan
        accounts.append(record)
    ids = set(chosen.id)
    events, edges, count = [], [], defaultdict(int)
    tweets_name = find_member(inner, "tweets.csv")
    scanned = invalid_times = 0
    if tweets_name:
        with inner.open(tweets_name) as stream:
            # Only these fields enter the pipeline. The original tweet timestamp includes its UTC offset.
            reader = pd.read_csv(stream, dtype=str, chunksize=50000, encoding="utf-8", encoding_errors="replace", keep_default_na=False,
                                 usecols=lambda c: c in {"id", "user_id", "created_at", "text", "retweeted_status_id", "in_reply_to_status_id"})
            for chunk in reader:
                scanned += len(chunk)
                chunk = chunk[chunk.user_id.isin(ids)]
                for _, row in chunk.iterrows():
                    uid = str(row.user_id)
                    if count[uid] >= event_limit:
                        continue
                    stamp = pd.to_datetime(row.get("created_at", ""), utc=True, errors="coerce", format="mixed")
                    if pd.isna(stamp):
                        invalid_times += 1
                        continue
                    tweet_id = str(row.id)
                    count[uid] += 1
                    objects = []
                    for key in ["retweeted_status_id", "in_reply_to_status_id"]:
                        value = str(row.get(key, ""))
                        if value not in {"", "0", "NULL", "nan", "None"}:
                            objects.append(("post:" + value, "retweet" if key == "retweeted_status_id" else "reply"))
                    text = str(row.get("text", ""))
                    for hashtag in re.findall(r"(?<!\w)#[\w]+", text):
                        objects.append(("hashtag:" + hashtag.casefold(), "hashtag"))
                    for url in re.findall(r"https?://[^\s<>\"]+", text):
                        objects.append(("url:" + url.rstrip(".,;!?)"), "url"))
                    if not objects:
                        objects = [("post:" + tweet_id, "post")]
                    # Hash observable objects; never use labels, cohort names, or group IDs as objects.
                    for obj, action in sorted(set(objects)):
                        obj_hash = hashlib.sha256(obj.encode("utf-8")).hexdigest()[:24]
                        events.append(dict(event_id=tweet_id, account_id=uid, timestamp=stamp, object_id=obj_hash, action=action))
                if all(count[uid] >= event_limit for uid in ids):
                    break
    for basename in ["friends.csv", "followers.csv"]:
        name = find_member(inner, basename)
        if name:
            with inner.open(name) as stream:
                for chunk in pd.read_csv(stream, dtype=str, chunksize=100000):
                    sub = chunk[chunk.source_id.isin(ids) & chunk.target_id.isin(ids)]
                    edges.extend(dict(source=str(r.source_id), target=str(r.target_id), relation="follows") for _, r in sub.iterrows())
    audit = dict(cohort=cohort, label=label, source_accounts=len(raw), selected_accounts=len(accounts), unique_observed_tweets=sum(count.values()),
                 object_event_records=len(events), raw_tweets_scanned=scanned, invalid_selected_timestamps=invalid_times,
                 accounts_without_events=len(ids - set(count)), unique_source_ids=raw.id.tolist())
    return accounts, events, edges, audit


def prepare(name, root, account_limit=120, event_limit=40, seed=2026):
    filename, url = FILES[name]
    archive = root / "raw" / filename
    download(archive, url)
    out = root / "processed" / name
    out.mkdir(parents=True, exist_ok=True)
    all_accounts, all_events, all_edges, audits = [], [], [], []
    if name == "cresci2017":
        outer = zipfile.ZipFile(archive)
        members = [n for n in outer.namelist() if n.endswith(".csv.zip") and "crowdflower" not in n]
        read = outer.read
    else:
        outer = tarfile.open(archive)
        members = [m.name for m in outer.getmembers() if m.name.endswith(".csv.zip")]
        read = lambda n: outer.extractfile(n).read()
    disclaimer = read("READ.ME").decode(errors="replace")
    (out / "DATA_LICENSE.txt").write_text(disclaimer, encoding="utf-8")
    try:
        for member in sorted(members):
            cohort = Path(member).name.removesuffix(".csv.zip")
            label = 0 if cohort in {"genuine_accounts", "E13", "TFP"} else 1
            print(f"Preparing {name}/{cohort}", flush=True)
            with zipfile.ZipFile(io.BytesIO(read(member))) as inner:
                accounts, events, edges, audit = convert_inner(inner, cohort, label, account_limit, event_limit, seed)
            all_accounts.extend(accounts)
            all_events.extend(events)
            all_edges.extend(edges)
            audits.append(audit)
    finally:
        outer.close()
    accounts = pd.DataFrame(all_accounts)
    # Cross-cohort duplicate IDs are excluded entirely rather than assigned arbitrarily.
    duplicate_ids = accounts.loc[accounts.account_id.duplicated(keep=False), "account_id"].unique()
    accounts = accounts[~accounts.account_id.isin(duplicate_ids)]
    valid = set(accounts.account_id)
    events = pd.DataFrame(all_events, columns=EVENT_COLUMNS)
    events = events[events.account_id.isin(valid)].drop_duplicates(["event_id", "account_id", "object_id", "action"])
    edges = pd.DataFrame(all_edges, columns=["source", "target", "relation"]).drop_duplicates()
    accounts.to_csv(out / "accounts.csv", index=False)
    events.to_csv(out / "events.csv", index=False)
    edges.to_csv(out / "edges.csv", index=False)
    source_ids = sorted({uid for a in audits for uid in a.pop("unique_source_ids", [])})
    pd.DataFrame({"account_id": source_ids}).to_csv(out / "source_universe_ids.csv", index=False)
    dump_json(out / "preparation_audit.json", dict(cohorts=audits, excluded_cross_cohort_duplicate_ids=list(duplicate_ids), archive_sha256=sha256(archive)))
    dump_json(out / "metadata.json", dict(name=name + "_pilot", data_kind="real", group_type="proxy_collection", chronology_supported=False,
        source=url, citation="Cresci et al. The paradigm-shift of social spambots. WWW Companion 2017; Cresci et al. Fame for sale. Decision Support Systems 2015.",
        license="Research purposes only; original DATA_LICENSE.txt retained", retrieved_utc=datetime.now(timezone.utc).isoformat(),
        archive_sha256=sha256(archive), sample_seed=seed, accounts_per_collection_cap=account_limit, tweets_per_account_cap=event_limit,
        group_definition="Original archive collection category, not independently verified campaigns", source_universe_ids="source_universe_ids.csv",
        collection_family="Cresci overlapping research collections",
        limitations="Bounded pilot, not full benchmark. First available tweets per selected account in archive order; snapshot timing unverified, so chronological evaluation unsupported. Cresci-2015 and Cresci-2017 share underlying accounts/collections; transfer is a sensitivity analysis, not independent external validation."))
    print(name, "accounts", len(accounts), "object events", len(events), "edges", len(edges), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--accounts-per-collection", type=int, default=120)
    p.add_argument("--tweets-per-account", type=int, default=40)
    p.add_argument("--datasets", nargs="+", default=list(FILES), choices=list(FILES))
    args = p.parse_args()
    for name in args.datasets:
        prepare(name, Path(args.root), args.accounts_per_collection, args.tweets_per_account)


if __name__ == "__main__":
    main()
