from __future__ import annotations

import argparse
from dataclasses import fields
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .build_graph import build_graph
from .calibrate import SetCalibrator
from .data_audit import load_dataset
from .run_experiments import model_inputs
from .train_baselines import RelationalNet, FittedModel
from .utils import seed_everything


def predict(run, split, model, data, output):
    run = Path(run)
    cfg = json.loads((run / "configuration.json").read_text())
    seed_everything(0, cfg["training"]["threads"])
    checkpoint = run / "checkpoints" / split / model
    settings = json.loads((checkpoint / "settings.json").read_text())
    with (checkpoint / "preprocessor.pkl").open("rb") as f:
        preprocessor = pickle.load(f)
    dataset = load_dataset(data, require_labels=False)
    bundle = build_graph(dataset, cfg["graph"], seed=int(split.rsplit("s", 1)[1]))
    x, matrices, names = model_inputs(dataset, bundle, model)
    if names != settings["feature_names"]:
        raise ValueError("Inference feature schema differs from the fitted checkpoint")
    if (checkpoint / "weights.pt").exists():
        state = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
        network = RelationalNet(len(names), cfg["training"]["hidden_dim"], len(matrices), cfg["training"]["dropout"], sage=model == "GraphSAGE")
        network.load_state_dict(state)
        fitted = FittedModel(network, True, 0., settings, [])
    else:
        with (checkpoint / "estimator.pkl").open("rb") as f:
            fitted = FittedModel(pickle.load(f), False, 0., settings, [])
    table = pd.read_csv(run / "calibration.csv")
    record = table[(table.split_id == split) & (table.model == model) & (table.method == "campaign_crc")].iloc[0].to_dict()
    kwargs = {f.name: record[f.name] for f in fields(SetCalibrator) if f.name in record and pd.notna(record[f.name])}
    calibrator = SetCalibrator(**kwargs)
    p = fitted.predict(preprocessor.transform(x).astype(np.float32), matrices)
    sets = calibrator.predict(p)
    result = dataset.accounts[["account_id"]].copy()
    result["p_human"], result["p_bot"] = p.T
    result["set_human"], result["set_bot"] = sets.T
    label = np.where(sets.sum(1) == 1, sets.argmax(1), -1)
    result["decision"] = pd.Series(label).map({-1: "needs_review", 0: "human", 1: "bot"}).to_numpy()
    result["calibration_feasible"] = calibrator.feasible
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} account predictions to {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict with a saved model; pickle checkpoints must be trusted")
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="P0_s11")
    parser.add_argument("--model", default="Proposed")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="output/predictions.csv")
    args = parser.parse_args()
    predict(args.run, args.split, args.model, args.data, args.output)


if __name__ == "__main__":
    main()
