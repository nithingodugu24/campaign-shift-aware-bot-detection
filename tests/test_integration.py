import numpy as np
import pandas as pd

from src.data_audit import load_dataset
from src.predict import predict
from src.run_experiments import ExperimentRunner, load_config
from src.synthetic import generate


def test_checkpoint_roundtrip_and_end_to_end_outputs(tmp_path):
    data_dir = tmp_path / "input"
    generate(data_dir, groups=16, accounts_per_group=8)
    data = load_dataset(data_dir)
    cfg = load_config("configs/pilot.yaml")
    cfg.update(seeds=[11], protocols=["P1"], models=["Proposed"], bootstrap_repeats=5)
    cfg["training"].update(epochs=2, patience=2, learning_rates=[.01])
    cfg["graph"]["null_permutations"] = 2
    run_dir = tmp_path / "run"
    runner = ExperimentRunner(data, cfg, run_dir)
    runner.run_split("P1", 11)
    manifest = pd.read_csv(run_dir / "splits/P1_s11/manifest.csv", dtype={"account_id": str})
    test = data.subset(manifest.loc[manifest.partition == "test", "account_id"])
    infer = tmp_path / "inference"
    infer.mkdir()
    test.accounts.drop(columns="label").to_csv(infer / "accounts.csv", index=False)
    test.events.to_csv(infer / "events.csv", index=False)
    test.edges.to_csv(infer / "edges.csv", index=False)
    (infer / "metadata.json").write_text((data_dir / "metadata.json").read_text())
    predicted = predict(run_dir, "P1_s11", "Proposed", infer, tmp_path / "predictions.csv")
    original = pd.read_csv(run_dir / "checkpoints/P1_s11/Proposed/test_predictions.csv")
    assert np.allclose(original.p_bot, predicted.p_bot, atol=1e-7)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert {"E1", "E2", "E3", "E4"}.issubset(set(metrics.experiment))
    assert not set(test.accounts.account_id).intersection(set(manifest.loc[manifest.partition == "train", "account_id"]))
    assert set(predicted.decision).issubset({"human", "bot", "needs_review"})
