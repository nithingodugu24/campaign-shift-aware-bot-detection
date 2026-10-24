"""Re-run P2 and merge it with an existing complete suite without repeating other tests."""
import argparse
from pathlib import Path
import pandas as pd

from src.data_audit import load_dataset
from src.run_experiments import ExperimentRunner, load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/pilot.yaml")
    args = p.parse_args()
    output = Path(args.output)
    runner = ExperimentRunner(load_dataset(args.data), load_config(args.config), output)
    for attr, name in [("metrics", "metrics.csv"), ("groups", "group_metrics.csv"), ("runtimes", "runtime.csv"), ("graphs", "graph_statistics.csv"), ("curves", "selective_curves.csv"), ("status", "experiment_status.csv"), ("calibration", "calibration.csv")]:
        path = output / name
        if path.exists():
            df = pd.read_csv(path)
            if "protocol" in df:
                df = df[df.protocol != "P2"]
            setattr(runner, attr, df.to_dict("records"))
    for seed in runner.config["seeds"]:
        runner.run_split("P2", seed)
    runner.save()


if __name__ == "__main__":
    main()
