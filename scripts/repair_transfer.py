from pathlib import Path
from dataclasses import fields
import json
import pickle
import pandas as pd
import torch

from src.calibrate import SetCalibrator
from src.data_audit import load_dataset
from src.run_experiments import ExperimentRunner, load_config
from src.train_baselines import RelationalNet, FittedModel


def main():
    for suite, data_path, target_path in [("public", "data/processed/cresci2017", "data/processed/gilani-2017"),
            ("collection", "data/processed/collection_pool", "data/processed/cresci-rtbust-2019"),
            ("synthetic", "data/processed/synthetic_source", "data/processed/synthetic_target")]:
        output = Path("results") / suite
        runner = ExperimentRunner(load_dataset(data_path), load_config("configs/pilot.yaml"), output)
        for attr, name in [("metrics", "metrics.csv"), ("groups", "group_metrics.csv"), ("runtimes", "runtime.csv"), ("graphs", "graph_statistics.csv"), ("curves", "selective_curves.csv"), ("status", "experiment_status.csv"), ("calibration", "calibration.csv")]:
            frame = pd.read_csv(output / name)
            if "protocol" in frame:
                if attr == "graphs":
                    frame = frame[frame.protocol != "P4"]
                elif attr in {"metrics", "groups"}:
                    frame = frame[~((frame.protocol == "P4") & (frame.model == "A5_corrected"))]
            setattr(runner, attr, frame.to_dict("records"))
        target = load_dataset(target_path)
        if suite == "synthetic":
            ckpt = output / "checkpoints/P0_s11/A5_corrected"
            with (ckpt / "preprocessor.pkl").open("rb") as f:
                pre = pickle.load(f)
            settings = json.loads((ckpt / "settings.json").read_text())
            network = RelationalNet(settings["input_dim"], 32, 1, .15)
            network.load_state_dict(torch.load(ckpt / "weights.pt", weights_only=True))
            fitted = FittedModel(network, True, 0, settings, [])
            table = pd.DataFrame(runner.calibration)
            row = table[(table.split_id == "P0_s11") & (table.model == "A5_corrected") & (table.method == "campaign_crc")].iloc[0].to_dict()
            cal = SetCalibrator(**{field.name: row[field.name] for field in fields(SetCalibrator) if field.name in row and pd.notna(row[field.name])})
            runner.saved[("P0", 11, "A5_corrected")] = (fitted, pre, cal, None, None, None)
            runner.config["models"] = ["A5_corrected"]
            runner.transfer(target)
        else:
            source_ids = runner.dataset.source_universe
            target = target.subset(target.accounts.loc[~target.accounts.account_id.isin(source_ids), "account_id"])
            runner.bundle(target, runner.context("P4", 11, target_dataset=target.name))
        runner.save()


if __name__ == "__main__":
    main()
