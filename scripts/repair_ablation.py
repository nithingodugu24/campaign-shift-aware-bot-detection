"""Re-evaluate one ablation and rebuild graph diagnostics after a graph implementation fix."""
from __future__ import annotations
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_audit import load_dataset
from src.run_experiments import ExperimentRunner, load_config, PARTITIONS
from src.robustness import perturb_events


def repair(data_path, output):
    output = Path(output)
    data = load_dataset(data_path)
    original_config = load_config("configs/collection.yaml" if output.name == "collection" else "configs/pilot.yaml")
    runner = ExperimentRunner(data, original_config, output)
    for attr, name in [("metrics", "metrics.csv"), ("groups", "group_metrics.csv"), ("runtimes", "runtime.csv"), ("graphs", "graph_statistics.csv"), ("curves", "selective_curves.csv"), ("status", "experiment_status.csv"), ("calibration", "calibration.csv")]:
        path = output / name
        if path.exists():
            frame = pd.read_csv(path)
            if "model" in frame:
                frame = frame[frame.model != "A5_corrected"]
            if attr in {"runtimes", "graphs"}:
                # Graph stages are rebuilt rather than keeping stale zero-entry statistics.
                if attr == "runtimes":
                    frame = frame[~frame.component.isin(["graph_and_null", "graph_raw_seconds", "null_seconds"])]
                else:
                    frame = frame.iloc[:0]
            setattr(runner, attr, frame.to_dict("records"))
    runner.config = copy.deepcopy(original_config)
    runner.config["models"] = ["A5_corrected"]
    existing_splits = sorted((output / "splits").glob("*/manifest.csv"))
    for path in existing_splits:
        protocol, seed_str = path.parent.name.split("_s")
        runner.run_split(protocol, int(seed_str))
    # Reconstruct all graph-only robustness/scalability workloads for correct diagnostics.
    preferred = "P1" if (output / "splits/P1_s11/manifest.csv").exists() else "P0"
    for seed in original_config["seeds"]:
        manifest = pd.read_csv(output / "splits" / f"{preferred}_s{seed}/manifest.csv", dtype={"account_id": str})
        clean = data.subset(manifest.loc[manifest.partition == "test", "account_id"])
        conditions = [("clean", 0)] + [("remove", v) for v in original_config["robustness"]["remove_fractions"]] + [("jitter", v) for v in original_config["robustness"]["jitter_seconds"]]
        for kind, value in conditions:
            changed = perturb_events(clean, kind, value, seed)
            runner.bundle(changed, runner.context(preferred, seed, condition=f"{kind}_{value}"))
    runner.config = original_config
    runner.save()
    # Whole-subset scalability graph records will be regenerated in a fresh dedicated pass.
    print(output, "A5 and graph diagnostics repaired", flush=True)


if __name__ == "__main__":
    for data_path, output in [("data/processed/cresci2017", "results/public"), ("data/processed/collection_pool", "results/collection"), ("data/processed/synthetic_source", "results/synthetic")]:
        repair(data_path, output)
