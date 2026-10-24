"""Export the existing evidence into one directory per experiment.

The result CSVs remain the source of truth. This command does not retrain models.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from src.utils import dump_json, sha256


EXPERIMENTS = {
    "E1": ("campaign_shift", "Random versus collection shift", "02_protocol_performance", ["main_detection.csv", "random_to_collection_change.csv"], ["splits.py", "run_experiments.py"]),
    "E2": ("temporal_coordination", "Contribution of temporal coordination", "03_feature_ablation", ["ablation.csv", "paired_statistics.csv"], ["preprocess.py", "train_baselines.py", "train_proposed.py"]),
    "E3": ("calibration", "Calibration and group reliability", "04_group_miscoverage", ["calibration.csv"], ["calibrate.py", "evaluate.py"]),
    "E4": ("null_model_ablation", "Null correction and graph construction", "05_graph_ablation", ["null_ablation.csv", "graph_statistics.csv"], ["build_graph.py", "train_proposed.py"]),
    "E5": ("selective_prediction", "Selective prediction and abstention", "06_selective_tradeoff", ["matched_coverage_descriptive.csv"], ["calibrate.py", "evaluate.py"]),
    "E6": ("robustness", "Missing events and timestamp robustness", "07_robustness", ["robustness.csv"], ["robustness.py", "run_experiments.py"]),
    "E7": ("scalability", "Computational scalability", "08_scalability", [], ["utils.py", "build_graph.py", "run_experiments.py"]),
    "E8": ("cross_dataset", "Cross dataset generalization", "09_transfer", ["transfer.csv"], ["run_experiments.py", "predict.py"]),
}
SUITES = ["public", "collection", "synthetic"]


def organize(root, selected=None):
    root = Path(root).resolve()
    base = root / "experiments"
    base.mkdir(exist_ok=True)
    index = ["# Experiments", "", "Each E1-E8 folder contains separate public, collection and synthetic evidence, figures and instructions. These are exports of the executed studies in `results/`; synthetic evidence remains explicitly separate.", ""]
    checked = []
    for eid, (slug, title, figure, tables, modules) in EXPERIMENTS.items():
        folder_name = f"{eid}_{slug}"
        index.append(f"- [{eid} {title}]({folder_name}/README.md)")
        if selected and eid not in selected:
            continue
        folder = base / folder_name
        folder.mkdir(exist_ok=True)
        lines = [f"# {eid} {title}", "", "## Contents", "", "Each study subfolder contains the saved evidence for this experiment. `public` uses the Cresci-2017 pilot, `collection` uses proxy collection holdouts, and `synthetic` is an explicitly generated validation fixture. Check each study's `status.csv` and source metadata for unsupported cases.", "", "## Implementation", ""]
        lines += [f"- [{module}](../../src/{module})" for module in modules]
        lines += ["", "The experiments share trained detectors and audited partitions. Checkpoints and split manifests remain under `results/<study>/` so every experiment refers to the same fitted models and held-out IDs.", "", "## Reproduction", "", "From the project root, use `run_all.ps1` to reproduce the underlying studies. To refresh only this folder from existing results:", "", "```powershell", f".venv\\Scripts\\python.exe -m scripts.organize_experiments --experiment {eid}", "```", "", "Refreshing exports does not retrain models. Full setup commands are in [the project README](../../README.md)."]
        if eid == "E5":
            lines += ["", "`selective_curves.csv` contains the E5 measurements. The nearest-coverage table is descriptive and does not select deployment thresholds from test labels."]
        if eid == "E7":
            lines += ["", "`runtime.csv` contains the dedicated scalability workloads and their component timings. Total and component times overlap and must not be added together. See `measurement_notes.json`."]
        if eid == "E8":
            lines += ["", "Real targets provide profile snapshots only. Temporal/graph transfer is unsupported there; compatible model predictions remain in the linked canonical transfer directories."]
        (folder / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (folder / "refresh_results.ps1").write_text(f"$ErrorActionPreference = 'Stop'\nSet-Location -LiteralPath (Join-Path $PSScriptRoot '..\\..')\n& '.\\.venv\\Scripts\\python.exe' -m scripts.organize_experiments --experiment {eid}\nif ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}\n", encoding="utf-8")
        for suite in SUITES:
            source = root / "results" / suite
            dest = folder / suite
            dest.mkdir(exist_ok=True)
            sources = {}
            exported = {}
            def export(filename, predicate):
                path = source / filename
                if not path.exists():
                    return
                frame = pd.read_csv(path)
                subset = frame.loc[predicate(frame)].copy()
                subset.to_csv(dest / filename, index=False)
                assert len(pd.read_csv(dest / filename)) == len(subset)
                sources[filename] = sha256(path)
                exported[filename] = len(subset)
            export("metrics.csv", lambda f: f.experiment == eid)
            export("group_metrics.csv", lambda f: f.experiment == eid)
            if eid == "E5":
                export("selective_curves.csv", lambda f: f.experiment == "E5")
            if eid in {"E3", "E5"}:
                export("calibration.csv", lambda f: f.model == "Proposed")
            if eid == "E4":
                export("graph_statistics.csv", lambda f: f.protocol.isin(["P0", "P1", "P2", "P3"]) & f.partition.notna())
                export("runtime.csv", lambda f: f.component.isin(["graph_raw_seconds", "null_seconds", "graph_and_null"]) & f.protocol.isin(["P0", "P1", "P2", "P3"]) & f.get("partition", pd.Series(index=f.index, dtype=str)).notna())
                shutil.copy2(source / "training_graph_sensitivity.csv", dest / "training_graph_sensitivity.csv")
            if eid == "E7":
                export("runtime.csv", lambda f: f.protocol == "scalability")
                export("graph_statistics.csv", lambda f: f.protocol == "scalability")
                shutil.copy2(source / "scalability_measurement_notes.json", dest / "measurement_notes.json")
            if eid == "E8":
                export("runtime.csv", lambda f: f.component == "cross_dataset_inference")
                shutil.copy2(source / "transfer/compatibility.json", dest / "compatibility.json")
            statuses = pd.read_csv(source / "experiment_status.csv")
            direct = statuses.experiment == eid
            shared = statuses.experiment.isin(["E1", "E1-E5"]) if eid in {"E1", "E2", "E3", "E4", "E5"} else pd.Series(False, index=statuses.index)
            baseline = statuses.experiment == "baseline"
            statuses = statuses.loc[direct | shared | baseline].copy()
            statuses["status_scope"] = statuses.experiment.map(lambda x: "baseline_availability" if x == "baseline" else "this_experiment" if x == eid else "shared_detection_protocol")
            statuses.to_csv(dest / "status.csv", index=False)
            for name in ["configuration.json", "dataset_metadata.json"]:
                shutil.copy2(source / name, dest / name)
            for name in tables:
                path = source / "tables" / name
                if path.exists():
                    (dest / "tables").mkdir(exist_ok=True)
                    shutil.copy2(path, dest / "tables" / name)
            for stem in [figure] + (["01_split_manifest"] if eid == "E1" else []):
                for ext in ["png", "svg"]:
                    path = source / "figures" / f"{stem}.{ext}"
                    if path.exists():
                        (dest / "figures").mkdir(exist_ok=True)
                        shutil.copy2(path, dest / "figures" / path.name)
            notes = [f"# {eid} {suite} evidence", "", f"Canonical study: [results/{suite}](../../../results/{suite}/)", "", "This folder exports the measurements already executed for this experiment. Shared model checkpoints, test predictions and split manifests remain in the canonical study. `status_scope` distinguishes direct experiment status from shared protocol and baseline availability.", "", "Exported record counts:", ""]
            notes += [f"- `{name}`: {count} rows" for name, count in exported.items()]
            if eid in {"E5", "E7"}:
                notes += ["", "An empty detection metrics CSV is expected: the main evidence for this experiment is in selective curves or runtime measurements."]
            (dest / "README.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
            record = dict(experiment=eid, study=suite, canonical_source=f"results/{suite}", source_sha256=sources, exported_row_counts=exported)
            dump_json(dest / "export_manifest.json", record)
            checked.append(record)
    (base / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"Exported and checked {len(checked)} experiment/study folders")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", nargs="+", choices=list(EXPERIMENTS))
    args = parser.parse_args()
    organize(Path.cwd(), args.experiment)
