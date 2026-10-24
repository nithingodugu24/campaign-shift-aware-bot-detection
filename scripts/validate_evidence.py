from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.utils import dump_json, sha256


def validate():
    summaries = []
    for suite in ["public", "collection", "synthetic"]:
        folder = Path("results") / suite
        metrics = pd.read_csv(folder / "metrics.csv")
        main = metrics[metrics.experiment == "E1"]
        graph_records = pd.read_csv(folder / "graph_statistics.csv")
        checked_predictions = 0
        split_count = 0
        for path in (folder / "splits").glob("*/manifest.csv"):
            manifest = pd.read_csv(path, dtype={"account_id": str})
            assert not manifest.account_id.duplicated().any(), path
            assert set(manifest.partition) == {"train", "validation", "calibration", "test"}, path
            if path.parent.name.startswith(("P1", "P2")):
                assert manifest.groupby("group_id").partition.nunique().max() == 1, path
            if path.parent.name.startswith("P3"):
                stamps = pd.to_datetime(manifest.prediction_at, utc=True)
                parts = [stamps[manifest.partition == p] for p in ["train", "validation", "calibration", "test"]]
                assert all(a.max() < b.min() for a, b in zip(parts, parts[1:])), path
            expected = set(manifest.loc[manifest.partition == "test", "account_id"])
            for partition in ["train", "validation", "calibration", "test"]:
                edge_file = path.parent / f"{partition}_graph_edges.csv"
                edges = pd.read_csv(edge_file)
                protocol, seed_text = path.parent.name.split("_s")
                counts = graph_records[(graph_records.protocol == protocol) & (graph_records.seed == int(seed_text)) &
                    (graph_records.partition == partition) & (graph_records.graph_variant == "corrected")]
                assert len(counts) > 0 and (counts.edges == int((edges.corrected > 0).sum())).all(), edge_file
            for prediction in (folder / "checkpoints" / path.parent.name).glob("*/test_predictions.csv"):
                frame = pd.read_csv(prediction, dtype={"account_id": str})
                assert set(frame.account_id) == expected, prediction
                probs = frame[["p_human", "p_bot"]].to_numpy()
                assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all(), prediction
                assert np.allclose(probs.sum(1), 1, atol=1e-6), prediction
                sizes = frame[["set_human", "set_bot"]].astype(bool).sum(1).to_numpy()
                assert np.array_equal(frame.automatic_label.to_numpy() == -1, sizes != 1), prediction
                checked_predictions += len(frame)
            split_count += 1
        for seed in [11, 23, 37]:
            p1, p2 = [folder / "splits" / f"{p}_s{seed}/manifest.csv" for p in ["P1", "P2"]]
            if p1.exists() and p2.exists():
                left, right = pd.read_csv(p1), pd.read_csv(p2)
                assert not left.partition.equals(right.partition), f"Repeated holdout duplicates P1: {suite} {seed}"
        selective = metrics[metrics.singleton_rate == 0]
        assert selective.selective_accuracy.isna().all(), suite
        for (protocol, seed), frame in metrics[(metrics.experiment == "E6") & (metrics.model == "RF")].groupby(["protocol", "seed"]):
            clean = frame.loc[frame.condition == "clean_0", "macro_f1"].iloc[0]
            changed = frame[frame.condition.str.startswith(("remove_", "jitter_"))]
            assert np.allclose(changed.macro_f1, clean, atol=1e-12), f"RF changed under event-only perturbation: {suite}"
        graphs = pd.read_csv(folder / "graph_statistics.csv")
        assert (graphs.loc[graphs.graph_variant == "full", "max_degree"] <= 10).all(), suite
        calibration = pd.read_csv(folder / "calibration.csv")
        crc = calibration[calibration.method == "campaign_crc"]
        for row in crc.itertuples():
            if not row.feasible:
                assert row.q == 1 and row.corrected_calibration_risk > row.alpha - 1e-12
            else:
                assert row.corrected_calibration_risk <= row.alpha + 1e-12
        observed = set(metrics.experiment) | {"E5", "E7"}
        assert set(f"E{i}" for i in range(1, 9)).issubset(observed), suite
        assert len(pd.read_csv(folder / "selective_curves.csv")) > 0
        runtime = pd.read_csv(folder / "runtime.csv")
        assert (runtime.component == "E7_total").sum() == 3, suite
        assert (folder / "scalability_measurement_notes.json").exists(), suite
        assert (folder / "training_graph_sensitivity.csv").exists(), suite
        summaries.append(dict(study=suite, split_manifests=split_count, detection_evaluations=len(main), prediction_rows_checked=checked_predictions,
            metric_rows=len(metrics), csv_hash=sha256(folder / "metrics.csv"), result="passed"))
    dump_json("output/evidence_validation.json", dict(status="passed", studies=summaries,
        checks=["unique four-way account partitions", "P1/P2 group separation", "strict chronological cutoffs where supported", "distinct P2 holdouts",
                "prediction IDs match held-out accounts", "finite normalized probabilities", "set/abstention identity", "undefined accuracy at zero automatic rate",
                "RF invariance to event-only perturbations", "mutual top-k degree bound", "CRC finite-sample feasibility", "measured E1-E8 outputs", "dedicated E7 and graph sensitivity records"]))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    validate()
