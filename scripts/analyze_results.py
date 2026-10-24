from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import paired_comparison, holm
from src.utils import dump_json

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 180})
COLORS = ["#16627a", "#c46b29", "#538b58", "#9166a6", "#737373"]


def savefig(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / (name + ".png"), bbox_inches="tight")
    fig.savefig(folder / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def analyze(folder):
    folder = Path(folder)
    figures = folder / "figures"
    tables = folder / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    m = pd.read_csv(folder / "metrics.csv")
    g = pd.read_csv(folder / "group_metrics.csv")
    curves = pd.read_csv(folder / "selective_curves.csv")
    graphs = pd.read_csv(folder / "graph_statistics.csv")
    runtime = pd.read_csv(folder / "runtime.csv")
    metadata = json.loads((folder / "dataset_metadata.json").read_text())
    main = m[m.experiment == "E1"]
    metrics = ["macro_f1", "pr_auc", "roc_auc", "mcc", "fpr", "empirical_coverage", "singleton_rate", "selective_accuracy"]
    summary = main.groupby(["model", "protocol"])[metrics].agg(["mean", "std", "count"])
    summary.to_csv(tables / "main_detection.csv")
    preferred = "P1" if "P1" in main.protocol.unique() else "P0"
    for experiment, name in [("E2", "ablation"), ("E3", "calibration"), ("E4", "null_ablation"), ("E6", "robustness"), ("E8", "transfer")]:
        subset = m[m.experiment == experiment]
        if experiment not in {"E6", "E8"}:
            subset = subset[subset.protocol == preferred]
        keys = [k for k in ["model", "protocol", "calibrator" if experiment == "E3" else None, "condition"] if k and k in subset]
        subset.groupby(keys, dropna=False)[[c for c in metrics if c in subset]].agg(["mean", "std", "count"]).to_csv(tables / (name + ".csv"))
    graphs.groupby(["protocol", "graph_variant"])[["nodes", "edges", "density", "max_degree", "stable_null_fraction", "support_fraction"]].mean().to_csv(tables / "graph_statistics.csv")
    runtime.groupby("component").agg(calls=("seconds", "count"), total_seconds=("seconds", "sum"), mean_seconds=("seconds", "mean"), max_rss_mb=("peak_rss_mb", "max")).to_csv(tables / "runtime.csv")
    if {"P0", "P1"}.issubset(set(main.protocol)):
        drops = main.pivot(index=["model", "seed"], columns="protocol", values="macro_f1")
        drops["P0_minus_P1"] = drops.P0 - drops.P1
        drops.to_csv(tables / "random_to_collection_change.csv")
    # Three predefined paired comparisons. Average repeated observations per declared group;
    # do not treat seed x group rows as independent statistical units.
    pairs = []
    primary = g[(g.experiment == "E1") & (g.protocol == "P1")]
    if len(primary):
        grouped = primary.groupby(["model", "group_id"]).macro_f1.mean().unstack("model")
        for baseline in ["A2_rhythm", "A4_temporal", "GraphSAGE"]:
            if baseline in grouped and "Proposed" in grouped:
                pair = grouped[["Proposed", baseline]].dropna()
                pairs.append(dict(comparison=f"Proposed minus {baseline}", **paired_comparison(pair.Proposed, pair[baseline]),
                    interpretation="Exploratory group-averaged paired Wilcoxon; overlapping training sets and proxy-group dependence limit inference"))
    if pairs:
        stats = pd.DataFrame(pairs)
        stats["holm_p_value"] = np.nan
        valid = stats.p_value.notna()
        stats.loc[valid, "holm_p_value"] = holm(stats.loc[valid, "p_value"])
        stats.to_csv(tables / "paired_statistics.csv", index=False)
    # Figure 1: actual saved four-way partitions.
    split_file = folder / "splits" / f"{preferred}_s11" / "manifest.csv"
    if split_file.exists():
        manifest = pd.read_csv(split_file)
        counts = manifest.groupby(["partition", "label"]).size().unstack(fill_value=0).reindex(["train", "validation", "calibration", "test"])
        fig, ax = plt.subplots(figsize=(7, 3.5))
        counts.rename(columns={0: "Human", 1: "Bot"}).plot.bar(ax=ax, stacked=True, color=COLORS[:2], rot=0)
        ax.set(title=f"Saved {preferred} split  |  {metadata['data_kind']} data", ylabel="Accounts", xlabel="")
        savefig(fig, figures, "01_split_manifest")
    # Figure 2: identical metrics and eligible models across protocols.
    chosen = [x for x in ["RF", "SVM", "GraphSAGE", "Proposed"] if x in main.model.unique()]
    values = main[main.model.isin(chosen)].groupby(["protocol", "model"]).macro_f1.agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, model in enumerate(chosen):
        line = values.xs(model, level="model")
        ax.errorbar(line.index, line["mean"], yerr=line["std"].fillna(0), marker="o", capsize=3, label=model, color=COLORS[i])
    ax.set(title="E1  Detection across split protocols", ylabel="Macro-F1", ylim=(0, 1.04), xlabel="Protocol; bars show run standard deviation")
    ax.legend(ncol=2, fontsize=9)
    savefig(fig, figures, "02_protocol_performance")
    # Figure 3: controlled graph/feature variants.
    order = ["A1_profile", "A2_rhythm", "A3_static", "A4_temporal", "A5_corrected", "Proposed"]
    values = m[(m.experiment == "E2") & (m.protocol == preferred)].groupby("model").macro_f1.agg(["mean", "std"]).reindex(order).dropna(subset=["mean"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(values.index, values["mean"], xerr=values["std"].fillna(0), color=COLORS[0], capsize=3)
    ax.invert_yaxis()
    ax.set(title=f"E2  Feature and graph ablations on {preferred}", xlabel="Macro-F1; run mean and standard deviation", xlim=(0, 1.04))
    savefig(fig, figures, "03_feature_ablation")
    # Figure 4: each point is an actually observed group/run, explicitly repeated.
    calgroups = g[(g.experiment == "E3") & (g.protocol == preferred)]
    fig, ax = plt.subplots(figsize=(9, 4))
    labels, arrays = [], []
    for method, frame in calgroups.groupby("calibrator"):
        labels.append(method)
        arrays.append(frame.miscoverage.dropna().to_numpy())
    if arrays:
        ax.boxplot(arrays, tick_labels=labels, showmeans=True)
    ax.tick_params(axis="x", labelrotation=18)
    ax.set(title="E3  Observed group miscoverage across runs", ylabel="Fraction of true labels excluded", ylim=(-.03, 1.03))
    ax.axhline(.1, color=COLORS[1], linestyle="--", linewidth=1)
    savefig(fig, figures, "04_group_miscoverage")
    # Figure 5: raw/corrected/filtered graph sizes.
    graph_table = graphs[(graphs.protocol == preferred) & (graphs.get("partition", pd.Series(index=graphs.index, dtype=str)) == "train")]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    stats = graph_table.groupby("graph_variant").edges.mean().reindex(["none", "static", "temporal", "corrected", "full"]).fillna(0)
    ax.bar(stats.index, stats, color=COLORS[0])
    ax.set(title="E4  Mean training graph edge counts", ylabel="Undirected edges", xlabel="Graph variant")
    savefig(fig, figures, "05_graph_ablation")
    # Figure 6: threshold operating points averaged across seeds, no test-selected optimum.
    fig, ax = plt.subplots(figsize=(8, 4))
    selected = curves[curves.protocol == preferred]
    for i, (method, frame) in enumerate(selected.groupby("method")):
        points = frame.groupby("operating_point")[["singleton_rate", "selective_accuracy"]].mean().sort_values("singleton_rate")
        points = points.dropna()
        ax.plot(points.singleton_rate, points.selective_accuracy, "o-", label=method, color=COLORS[i % len(COLORS)])
    ax.set(title="E5  Automatic-label utility at predefined operating points", xlabel="Singleton rate", ylabel="Accuracy among singleton decisions", xlim=(-.03, 1.03), ylim=(0, 1.04))
    ax.legend(fontsize=8, loc="best")
    savefig(fig, figures, "06_selective_tradeoff")
    # Figure 7: independent missing-event and timestamp conditions.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    robust = m[(m.experiment == "E6") & m.model.isin(["RF", "GraphSAGE", "Proposed"])]
    for ax, kind, label in zip(axes, ["remove", "jitter"], ["Removed event fraction", "Uniform jitter bound in seconds"]):
        for i, (model, frame) in enumerate(robust.groupby("model")):
            clean = frame[frame.condition == "clean_0"].macro_f1.mean()
            sub = frame[frame.condition.str.startswith(kind + "_")].copy()
            sub["severity"] = sub.condition.str.split("_").str[-1].astype(float)
            mean = sub.groupby("severity").macro_f1.mean()
            ax.plot([0] + mean.index.tolist(), [clean] + mean.tolist(), "o-", label=model, color=COLORS[i])
        ax.set(xlabel=label, ylabel="Macro-F1", ylim=(0, 1.04))
    axes[0].legend(fontsize=8)
    fig.suptitle("E6  Frozen-model robustness")
    savefig(fig, figures, "07_robustness")
    # Figure 8: measured workload costs, no theoretical complexity fit.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    scale = runtime[runtime.component == "E7_total"].dropna(subset=["subset_size"])
    axes[0].plot(scale.subset_size, scale.seconds, "o-", color=COLORS[0])
    axes[1].plot(scale.subset_size, scale.peak_rss_mb, "o-", color=COLORS[1])
    axes[0].set(xlabel="Accounts in workload", ylabel="Total measured seconds")
    axes[1].set(xlabel="Accounts in workload", ylabel="Peak process RSS in MB")
    fig.suptitle("E7  CPU scalability pilot")
    savefig(fig, figures, "08_scalability")
    # Figure 9: only compatible actually executed target models.
    transfer = m[m.experiment == "E8"]
    if len(transfer):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        transfer.groupby("model").macro_f1.mean().plot.bar(ax=ax, color=COLORS[0], rot=0)
        ax.set(title="E8  Direct transfer without target-label tuning", ylabel="Target Macro-F1", xlabel="Compatible model", ylim=(0, 1.04))
        savefig(fig, figures, "09_transfer")
    # Matched operating points are matched by realized coverage for description only;
    # no threshold is selected for deployment on the test set.
    matched = []
    for (protocol, seed, method), frame in curves.groupby(["protocol", "seed", "method"]):
        for rate in [.25, .5, .75]:
            j = (frame.singleton_rate - rate).abs().idxmin()
            row = frame.loc[j].to_dict()
            row.update(requested_rate=rate, coverage_distance=abs(row["singleton_rate"] - rate),
                       comparison_note="Nearest measured test coverage for descriptive comparison; no target threshold tuning; inspect coverage_distance")
            matched.append(row)
    pd.DataFrame(matched).to_csv(tables / "matched_coverage_descriptive.csv", index=False)
    dump_json(folder / "analysis_summary.json", dict(dataset=metadata["name"], data_kind=metadata["data_kind"], primary_available_protocol=preferred,
        detection_runs=len(main), metric_records=len(m), group_metric_records=len(g),
        experiments_with_measured_outputs=sorted(set(m.experiment) | ({"E5"} if len(curves) else set()) | ({"E7"} if len(scale) else set())),
        primary_protocol_supported="P1" in main.protocol.unique(), transfer_accounts=int(transfer.n.max()) if len(transfer) else 0))
    print(folder, "figures", len(list(figures.glob("*.png"))), "metric records", len(m))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+", default=["results/public", "results/collection", "results/synthetic"])
    args = parser.parse_args()
    for folder in args.folders:
        analyze(folder)
