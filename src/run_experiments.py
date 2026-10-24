from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
import pandas as pd
import torch
import yaml

from .build_graph import build_graph, coordination_features
from .calibrate import fit_calibrator, SetCalibrator
from .data_audit import load_dataset, Dataset
from .evaluate import detection_metrics, selective_metrics, group_metrics, cluster_bootstrap
from .preprocess import account_features, fit_preprocessor
from .robustness import perturb_events
from .splits import make_split, split_summary, UnsupportedProtocol, PARTITIONS
from .train_baselines import train_model
from .utils import dump_json, environment, measured, seed_everything

CALIBRATORS = ["confidence", "temperature", "account_conformal", "group_max_conformal", "campaign_crc"]
VARIANTS = {
    "RF": (False, "none"), "SVM": (False, "none"), "LR": (False, "none"),
    "GraphSAGE": (True, "static"), "GroupDRO": (True, "none"), "GroupReweight": (True, "none"),
    "A1_profile": (False, "none"), "A2_rhythm": (True, "none"), "A3_static": (True, "static"),
    "A4_temporal": (True, "temporal"), "A5_corrected": (True, "corrected"), "Proposed": (True, "full")}


def merge(base, extra):
    result = copy.deepcopy(base)
    for k, value in extra.items():
        result[k] = merge(result[k], value) if isinstance(value, dict) and isinstance(result.get(k), dict) else value
    return result


def load_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    if "inherits" in config:
        base = load_config(path.parent / config.pop("inherits"))
        config = merge(base, config)
    return config


def model_inputs(dataset, bundle, model):
    rhythm, graph_type = VARIANTS[model]
    x, names = account_features(dataset, rhythm=rhythm)
    matrix = bundle.matrices[graph_type]
    # Coordination statistics only belong to graph variants.
    if graph_type != "none":
        x = np.column_stack([x, coordination_features(matrix)])
        names += ["log_degree", "log_strength", "clustering"]
    matrices = [matrix]
    if model == "Proposed":
        matrices.append(bundle.matrices["follows"])
    return x, matrices, names


class ExperimentRunner:
    def __init__(self, dataset, config, output):
        seed_everything(config["seeds"][0], config["training"]["threads"])
        self.dataset, self.config, self.output = dataset, config, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.metrics, self.groups, self.runtimes, self.graphs, self.curves, self.status, self.calibration = [], [], [], [], [], [], []
        self.saved = {}
        dump_json(self.output / "configuration.json", config)
        dump_json(self.output / "environment.json", environment())
        dump_json(self.output / "dataset_metadata.json", dataset.metadata)
        dump_json(self.output / "preregistered_analysis.json", dict(primary_protocol=config["primary_protocol"], primary_metric=config["primary_metric"],
            family=["Proposed vs A2_rhythm", "Proposed vs A4_temporal", "Proposed vs GraphSAGE"],
            inference="exploratory until independent groups and collection assumptions are justified", minimum_practical_effect=.02,
            graph_scope="inductive partition-local; no cross-partition message passing", chronology="requires historical profile availability; account creation time is not an observation cutoff"))

    def context(self, protocol, seed, **extra):
        return dict(dataset=self.dataset.name, data_kind=self.dataset.metadata["data_kind"], group_type=self.dataset.metadata["group_type"], protocol=protocol,
                    seed=seed, split_id=f"{protocol}_s{seed}", **extra)

    def record_status(self, experiment, status, reason, **extra):
        self.status.append(dict(experiment=experiment, status=status, reason=reason, **extra))
        print(f"[{experiment}] {status}: {reason}", flush=True)

    def bundle(self, data, context):
        with measured(self.runtimes, "graph_and_null", accounts=len(data.accounts), events=len(data.events), **context):
            bundle = build_graph(data, self.config["graph"], context["seed"])
        for variant, stats in bundle.stats.items():
            self.graphs.append(dict(**context, graph_variant=variant, **stats, **bundle.timings))
        for component, seconds in bundle.timings.items():
            self.runtimes.append(dict(**context, component=component, seconds=seconds, accounts=len(data.accounts), events=len(data.events), memory_method="Included in graph_and_null RSS measurement"))
        return bundle

    def evaluate(self, data, p, context, calibrator=None, experiment="E1", condition="clean", bootstrap=False):
        y = data.accounts.label.to_numpy(dtype=int)
        g = data.accounts.group_id.to_numpy()
        sets = calibrator.predict(p) if calibrator else None
        row = dict(**context, experiment=experiment, condition=condition, **detection_metrics(y, p))
        if sets is not None:
            row.update(selective_metrics(y, sets, g), calibrator=calibrator.method, calibration_feasible=calibrator.feasible)
        if bootstrap:
            row.update(cluster_bootstrap(y, p, g, context["seed"], self.config["bootstrap_repeats"]))
        self.metrics.append(row)
        for group in group_metrics(y, p, sets, g):
            self.groups.append(dict(**context, experiment=experiment, condition=condition, calibrator=calibrator.method if calibrator else "none", **group))
        return sets

    def run_split(self, protocol, seed):
        context = self.context(protocol, seed)
        try:
            manifest = make_split(self.dataset, protocol, seed)
        except UnsupportedProtocol as exc:
            self.record_status("E1", "unsupported", str(exc), **context)
            return
        split_dir = self.output / "splits" / context["split_id"]
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(split_dir / "manifest.csv", index=False)
        split_summary(manifest).to_csv(split_dir / "summary.csv")
        parts, bundles = {}, {}
        for part in PARTITIONS:
            parts[part] = self.dataset.subset(manifest.loc[manifest.partition == part, "account_id"])
            bundles[part] = self.bundle(parts[part], dict(**context, partition=part))
            bundles[part].edges.to_csv(split_dir / f"{part}_graph_edges.csv", index=False)
        no_events = len(self.dataset.events) == 0
        for model in self.config["models"]:
            if no_events and model not in {"RF", "SVM", "LR", "A1_profile"}:
                self.record_status("E2", "unsupported", "No timestamped events for temporal/graph experiment", **context, model=model)
                continue
            model_context = dict(**context, model=model, variant=model, train_accounts=len(parts["train"].accounts), calibration_groups=parts["calibration"].accounts.group_id.nunique())
            print(f"Training {self.dataset.name} {protocol} seed={seed} {model}", flush=True)
            features, matrices, schema = {}, {}, None
            with measured(self.runtimes, "feature_extraction", **model_context):
                for part in PARTITIONS:
                    features[part], matrices[part], schema = model_inputs(parts[part], bundles[part], model)
                preprocessor = fit_preprocessor(features["train"])
                features = {part: preprocessor.transform(x).astype(np.float32) for part, x in features.items()}
            def inputs(part):
                return (features[part], parts[part].accounts.label.to_numpy(dtype=int), matrices[part], parts[part].accounts.group_id.to_numpy())
            with measured(self.runtimes, "model_selection_and_training", **model_context):
                fitted = train_model(model, inputs("train"), inputs("validation"), self.config["training"], seed)
            predictions = {}
            for part in ["validation", "calibration", "test"]:
                with measured(self.runtimes, "inference", partition=part, batch_size=len(parts[part].accounts), **model_context):
                    predictions[part] = fitted.predict(features[part], matrices[part])
            ckpt = self.output / "checkpoints" / context["split_id"] / model
            ckpt.mkdir(parents=True, exist_ok=True)
            with (ckpt / "preprocessor.pkl").open("wb") as f:
                pickle.dump(preprocessor, f)
            if fitted.neural:
                torch.save(fitted.estimator.state_dict(), ckpt / "weights.pt")
            else:
                with (ckpt / "estimator.pkl").open("wb") as f:
                    pickle.dump(fitted.estimator, f)
            dump_json(ckpt / "settings.json", dict(**fitted.settings, feature_names=schema, model=model, graph=self.config["graph"], input_dim=len(schema)))
            pd.DataFrame(fitted.history).to_csv(ckpt / "training_history.csv", index=False)
            cdata, vdata, test = parts["calibration"], parts["validation"], parts["test"]
            with measured(self.runtimes, "calibration", **model_context):
                crc = fit_calibrator("campaign_crc", predictions["calibration"], cdata.accounts.label, cdata.accounts.group_id, self.config["alpha"])
            self.calibration.append(dict(**model_context, **crc.to_dict()))
            sets = self.evaluate(test, predictions["test"], model_context, crc, bootstrap=True)
            frame = test.accounts[["account_id", "group_id", "label"]].copy()
            frame["p_human"], frame["p_bot"] = predictions["test"].T
            frame["set_human"], frame["set_bot"] = sets.T
            frame["automatic_label"] = np.where(sets.sum(1) == 1, sets.argmax(1), -1)
            frame.to_csv(ckpt / "test_predictions.csv", index=False)
            for experiment in ["E2", "E4"]:
                self.evaluate(test, predictions["test"], model_context, crc, experiment=experiment)
            if model == "Proposed":
                for method in CALIBRATORS:
                    cal = fit_calibrator(method, predictions["calibration"], cdata.accounts.label, cdata.accounts.group_id, self.config["alpha"], predictions["validation"], vdata.accounts.label)
                    self.calibration.append(dict(**model_context, **cal.to_dict()))
                    self.evaluate(test, predictions["test"], model_context, cal, experiment="E3")
                    for alpha in self.config["selective_alphas"]:
                        operating = fit_calibrator(method, predictions["calibration"], cdata.accounts.label, cdata.accounts.group_id, alpha, predictions["validation"], vdata.accounts.label)
                        self.curves.append(dict(**model_context, experiment="E5", method=method, operating_point=alpha, q=operating.q, feasible=operating.feasible,
                            **selective_metrics(test.accounts.label.to_numpy(dtype=int), operating.predict(predictions["test"]), test.accounts.group_id)))
                for threshold in self.config["selective_thresholds"]:
                    cal = SetCalibrator("confidence", threshold, self.config["alpha"], 0)
                    self.curves.append(dict(**model_context, experiment="E5", method="confidence_grid", operating_point=threshold, q=threshold, feasible=True,
                        **selective_metrics(test.accounts.label.to_numpy(dtype=int), cal.predict(predictions["test"]), test.accounts.group_id)))
            self.saved[(protocol, seed, model)] = (fitted, preprocessor, crc, parts, bundles, predictions)
        self.record_status("E1-E5", "executed", "Partition-local models, graph ablations and calibration saved", **context)
        self.save()

    def robustness(self):
        config = self.config["robustness"]
        preferred = "P1" if any(k[0] == "P1" for k in self.saved) else "P0"
        for seed in self.config["seeds"]:
            models = [m for m in ["RF", "SVM", "GraphSAGE", "A2_rhythm", "Proposed"] if (preferred, seed, m) in self.saved]
            if not models or not len(self.dataset.events):
                continue
            parts = self.saved[(preferred, seed, models[0])][3]
            clean = parts["test"]
            conditions = [("clean", 0)] + [("remove", v) for v in config["remove_fractions"]] + [("jitter", v) for v in config["jitter_seconds"]]
            for kind, value in conditions:
                data = perturb_events(clean, kind, value, seed)
                context = self.context(preferred, seed, condition=f"{kind}_{value}")
                bundle = self.bundle(data, context)
                for model in models:
                    fitted, preprocessor, cal, _, _, _ = self.saved[(preferred, seed, model)]
                    x, matrices, _ = model_inputs(data, bundle, model)
                    p = fitted.predict(preprocessor.transform(x).astype(np.float32), matrices)
                    self.evaluate(data, p, self.context(preferred, seed, model=model, variant=model), cal, experiment="E6", condition=f"{kind}_{value}")
            # Strata are fixed from unperturbed training activity or predeclared group sizes.
            train_counts = parts["train"].events.groupby("account_id").event_id.nunique().reindex(parts["train"].accounts.account_id, fill_value=0)
            low_cutoff = train_counts.quantile(config["low_activity_quantile"])
            test_counts = clean.events.groupby("account_id").event_id.nunique().reindex(clean.accounts.account_id, fill_value=0).to_numpy()
            sizes = clean.accounts.group_id.map(clean.accounts.groupby("group_id").size()).to_numpy()
            strata = [("low_activity", test_counts <= low_cutoff), ("group_size_le10", sizes <= 10), ("group_size_11to30", (sizes > 10) & (sizes <= 30)), ("group_size_gt30", sizes > 30)]
            if "benign_synchronization" in clean.accounts:
                strata.append(("synthetic_benign_synchronization", clean.accounts.benign_synchronization.to_numpy(dtype=bool)))
            else:
                self.record_status("E6", "unsupported_subtest", "No independently identified benign synchronized human groups", seed=seed)
            for model in models:
                _, _, cal, _, _, predictions = self.saved[(preferred, seed, model)]
                for name, mask in strata:
                    if mask.any():
                        self.evaluate(clean.subset(clean.accounts.account_id[mask]), predictions["test"][mask], self.context(preferred, seed, model=model, variant=model), cal, experiment="E6", condition=name)
        self.record_status("E6", "executed" if len(self.dataset.events) else "unsupported", "Frozen-model perturbation evaluation; labels unchanged" if len(self.dataset.events) else "No event records")
        self.save()

    def scalability(self):
        if not len(self.dataset.events):
            self.record_status("E7", "unsupported", "No timestamped events")
            return
        seed = self.config["seeds"][0]
        rng = np.random.default_rng(seed)
        order = rng.permutation(self.dataset.accounts.account_id)
        for requested in self.config["scalability_sizes"]:
            if requested > len(order):
                continue
            data = self.dataset.subset(order[:requested])
            context = self.context("scalability", seed, subset_size=requested)
            with measured(self.runtimes, "E7_total", **context):
                bundle = self.bundle(data, context)
                manifest = make_split(data, "P0", seed)
                # Graph measured at whole-subset scale; learning remains partition-local below.
                local = {}
                for part in PARTITIONS[:2]:
                    sub = data.subset(manifest.loc[manifest.partition == part, "account_id"])
                    b = self.bundle(sub, dict(**context, partition=part))
                    x, mats, _ = model_inputs(sub, b, "Proposed")
                    local[part] = (sub, x, mats)
                pre = fit_preprocessor(local["train"][1])
                def inp(part):
                    sub, x, mats = local[part]
                    return pre.transform(x).astype(np.float32), sub.accounts.label.to_numpy(dtype=int), mats, sub.accounts.group_id.to_numpy()
                with measured(self.runtimes, "E7_training", **context):
                    model = train_model("Proposed", inp("train"), inp("validation"), self.config["training"], seed)
                with measured(self.runtimes, "E7_inference", batch_size=len(data.accounts), **context):
                    x, mats, _ = model_inputs(data, bundle, "Proposed")
                    model.predict(pre.transform(x).astype(np.float32), mats)
        self.record_status("E7", "executed", "Measured CPU process RSS and time at increasing account counts; no warm-up workloads excluded")
        dump_json(self.output / "scalability_measurement_notes.json", dict(mode="Single runner; external process contention must be recorded by the operator", scope="Desktop background activity uncontrolled; whole-process RSS includes libraries and previous stages", warmup="No workloads excluded; import time outside stage measurements"))
        self.save()

    def transfer(self, target):
        source_ids = self.dataset.source_universe or set(self.dataset.accounts.account_id)
        overlap = target.accounts.account_id.isin(source_ids)
        target = target.subset(target.accounts.loc[~overlap, "account_id"])
        if not len(target.accounts):
            self.record_status("E8", "unsupported", "Target contains no accounts after source-ID overlap removal")
            return
        seed = self.config["seeds"][0]
        context = self.context("P4", seed, target_dataset=target.name)
        bundle = self.bundle(target, context)
        for model in self.config["models"]:
            key = ("P0", seed, model)
            if key not in self.saved:
                continue
            if not len(target.events) and VARIANTS[model][0]:
                self.record_status("E8", "unsupported_model", "Target has profile snapshots only; rhythm and coordination inputs unavailable", model=model)
                continue
            fitted, pre, cal, *_ = self.saved[key]
            x, mats, names = model_inputs(target, bundle, model)
            with measured(self.runtimes, "cross_dataset_inference", model=model, batch_size=len(target.accounts), **context):
                p = fitted.predict(pre.transform(x).astype(np.float32), mats)
            self.evaluate(target, p, dict(**context, model=model, variant=model), cal, experiment="E8", bootstrap=True)
            dest = self.output / "transfer" / model
            dest.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(dict(account_id=target.accounts.account_id, label=target.accounts.label, p_bot=p[:, 1])).to_csv(dest / "predictions.csv", index=False)
        dump_json(self.output / "transfer" / "compatibility.json", dict(source=self.dataset.metadata, target=target.metadata,
            removed_overlapping_accounts=int(overlap.sum()), feature_mapping="Same canonical numeric profile fields; same rhythm and shared-object definitions. Missing target fields use source-fitted imputer.",
            target_label_tuning=False, target_calibration=False, limitation="Dataset-specific annotation and collection differences confound transfer; inspect feature missingness audits"))
        self.record_status("E8", "executed", f"Direct transfer to {target.name}, {len(target.accounts)} accounts; removed {int(overlap.sum())} overlapping source IDs")
        self.save()

    def save(self):
        for filename, rows in [("metrics.csv", self.metrics), ("group_metrics.csv", self.groups), ("runtime.csv", self.runtimes), ("graph_statistics.csv", self.graphs), ("selective_curves.csv", self.curves), ("experiment_status.csv", self.status), ("calibration.csv", self.calibration)]:
            if rows:
                pd.DataFrame(rows).to_csv(self.output / filename, index=False)

    def run(self, target=None):
        try:
            for protocol in self.config["protocols"]:
                for seed in self.config["seeds"]:
                    self.run_split(protocol, seed)
            self.robustness()
            self.scalability()
            if target is not None:
                self.transfer(target)
            else:
                self.record_status("E8", "unsupported", "No compatible independent target supplied")
            for method, reason in [("BotRGCN", "Official multimodal encoders and full required relational inputs not reproduced"), ("RGT", "Official heterogeneous transformer not reproduced"), ("TGN/TGAT", "Shared-object observations are not assumed equivalent to verified temporal interaction edges"), ("BotDGT/BotSTIP", "Official dynamic/hypergraph preprocessing and checkpoints not reproduced"), ("IRM", "Not reproduced; GroupDRO and group reweighting supplied as explicitly named alternatives")]:
                self.record_status("baseline", "not_reproduced", reason, model=method)
        except Exception:
            self.record_status("pipeline", "failed", traceback.format_exc())
            raise
        finally:
            self.save()


def main():
    parser = argparse.ArgumentParser(description="Run audited E1-E8 bot detection experiments")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seeds"][0], config["training"]["threads"])
    data = load_dataset(args.data, Path(args.output) / "audit")
    target = load_dataset(args.target, Path(args.output) / "target_audit") if args.target else None
    ExperimentRunner(data, config, args.output).run(target)


if __name__ == "__main__":
    main()
