from pathlib import Path
import time
import numpy as np
import pandas as pd

from src.build_graph import build_graph
from src.data_audit import load_dataset
from src.run_experiments import load_config
from scripts.prepare_public_data import stable_order


def run(data_path, output):
    output = Path(output)
    dataset = load_dataset(data_path)
    protocol = "P1" if (output / "splits/P1_s11/manifest.csv").exists() else "P0"
    manifest = pd.read_csv(output / "splits" / f"{protocol}_s11" / "manifest.csv", dtype={"account_id": str})
    ids = manifest.loc[manifest.partition == "train", "account_id"].tolist()
    ids = sorted(ids, key=lambda uid: stable_order(uid, 2026))[:128]
    train = dataset.subset(ids)
    config = load_config("configs/pilot.yaml")["graph"]
    bundles, times = {}, {}
    for name, cap, permutations in [("pilot", 64, 8), ("uncapped_training_reference", max(1, len(train.events)), 32)]:
        cfg = dict(config, object_event_cap=cap, null_permutations=permutations)
        start = time.perf_counter()
        bundles[name] = build_graph(train, cfg, seed=11)
        times[name] = time.perf_counter() - start
    def edge_map(bundle, col):
        return {(r.source, r.target): getattr(r, col) for r in bundle.edges.itertuples()}
    pilot, reference = bundles.values()
    p, r = edge_map(pilot, "raw"), edge_map(reference, "raw")
    keys = sorted(set(p) | set(r))
    x = np.array([p.get(k, 0) for k in keys])
    y = np.array([r.get(k, 0) for k in keys])
    retained = lambda b: {(row.source, row.target) for row in b.edges.itertuples() if row.retained}
    pp, rr = retained(pilot), retained(reference)
    summary = dict(accounts=len(train.accounts), object_event_records=len(train.events), selection="fixed hash order within training partition only; max 128 accounts",
        protocol=protocol, raw_pair_union=len(keys), raw_mean_absolute_difference=float(np.abs(x-y).mean()) if len(keys) else 0,
        raw_correlation=float(np.corrcoef(x, y)[0, 1]) if len(keys) > 1 and x.std() > 0 and y.std() > 0 else np.nan,
        retained_edge_jaccard=len(pp & rr)/len(pp | rr) if pp | rr else np.nan,
        pilot_retained_edges=len(pp), reference_retained_edges=len(rr), pilot_seconds=times["pilot"], reference_seconds=times["uncapped_training_reference"],
        interpretation="Exploratory sensitivity to cap and Monte Carlo budget combined; not a p-value, test-set tuning or proof of approximation accuracy at full scale")
    pd.DataFrame([summary]).to_csv(output / "training_graph_sensitivity.csv", index=False)
    print(output, summary, flush=True)


if __name__ == "__main__":
    for data, output in [("data/processed/cresci2017", "results/public"), ("data/processed/collection_pool", "results/collection"), ("data/processed/synthetic_source", "results/synthetic")]:
        run(data, output)
