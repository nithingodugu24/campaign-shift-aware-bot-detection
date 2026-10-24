"""Re-evaluate frozen checkpoints and measure scalability without retraining detection comparisons."""
from __future__ import annotations
import argparse
from dataclasses import fields
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from src.calibrate import SetCalibrator
from src.data_audit import load_dataset
from src.run_experiments import ExperimentRunner, load_config, PARTITIONS
from src.train_baselines import RelationalNet, FittedModel
from src.utils import dump_json


def recheck(data_path, output, config, robustness=False, scalability=False):
    output = Path(output)
    data = load_dataset(data_path, output / "audit")
    runner = ExperimentRunner(data, load_config(config), output)
    for attr, name in [("metrics", "metrics.csv"), ("groups", "group_metrics.csv"), ("runtimes", "runtime.csv"), ("graphs", "graph_statistics.csv"), ("curves", "selective_curves.csv"), ("status", "experiment_status.csv"), ("calibration", "calibration.csv")]:
        path = output / name
        if path.exists():
            df = pd.read_csv(path)
            if robustness and "experiment" in df:
                df = df[df.experiment != "E6"]
            if robustness and "condition" in df and attr in {"runtimes", "graphs"}:
                df = df[df.condition.isna()]
            if scalability and "protocol" in df:
                df = df[df.protocol != "scalability"]
            if scalability and attr == "status":
                df = df[df.experiment != "E7"]
            setattr(runner, attr, df.to_dict("records"))
    if robustness:
        preferred = "P1" if (output / "splits/P1_s11/manifest.csv").exists() else "P0"
        calibrations = pd.DataFrame(runner.calibration)
        for seed in runner.config["seeds"]:
            split_id = f"{preferred}_s{seed}"
            manifest = pd.read_csv(output / "splits" / split_id / "manifest.csv", dtype={"account_id": str})
            parts = {part: data.subset(manifest.loc[manifest.partition == part, "account_id"]) for part in PARTITIONS}
            for model in ["RF", "SVM", "GraphSAGE", "A2_rhythm", "Proposed"]:
                ckpt = output / "checkpoints" / split_id / model
                if not ckpt.exists():
                    continue
                with (ckpt / "preprocessor.pkl").open("rb") as f:
                    pre = pickle.load(f)
                settings = json.loads((ckpt / "settings.json").read_text())
                if (ckpt / "weights.pt").exists():
                    network = RelationalNet(settings["input_dim"], runner.config["training"]["hidden_dim"], 2 if model == "Proposed" else 1, runner.config["training"]["dropout"], sage=model == "GraphSAGE")
                    network.load_state_dict(torch.load(ckpt / "weights.pt", weights_only=True))
                    fitted = FittedModel(network, True, 0., settings, [])
                else:
                    with (ckpt / "estimator.pkl").open("rb") as f:
                        fitted = FittedModel(pickle.load(f), False, 0., settings, [])
                row = calibrations[(calibrations.split_id == split_id) & (calibrations.model == model) & (calibrations.method == "campaign_crc")].iloc[0].to_dict()
                cal = SetCalibrator(**{field.name: row[field.name] for field in fields(SetCalibrator) if field.name in row and pd.notna(row[field.name])})
                p = pd.read_csv(ckpt / "test_predictions.csv")[["p_human", "p_bot"]].to_numpy()
                runner.saved[(preferred, seed, model)] = (fitted, pre, cal, parts, None, {"test": p})
        runner.robustness()
    if scalability:
        runner.scalability()
        dump_json(output / "scalability_measurement_notes.json", dict(mode="One experiment worker at a time for this dedicated E7 pass", scope="Desktop background activity uncontrolled; whole-process RSS includes libraries and previous stages", warmup="No workloads excluded; import time outside stage measurements", training_timing="Main E1-E5 model timings were collected while other experiment workers could run; use dedicated E7 component timings for scalability comparisons"))
    runner.save()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/pilot.yaml")
    p.add_argument("--robustness", action="store_true")
    p.add_argument("--scalability", action="store_true")
    args = p.parse_args()
    recheck(args.data, args.output, args.config, args.robustness, args.scalability)
