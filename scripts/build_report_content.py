from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from src.utils import dump_json


def fmt(value, digits=3):
    return "undefined" if pd.isna(value) else f"{value:.{digits}f}"


def table(frame, columns=None):
    if columns:
        frame = frame[columns]
    return dict(type="table", headers=[str(x) for x in frame.columns], rows=[[str(x) for x in row] for row in frame.itertuples(index=False, name=None)])


def detection(frame, core=False):
    frame = frame[frame.experiment == "E1"]
    if core:
        frame = frame[frame.model.isin(["RF", "SVM", "GraphSAGE", "Proposed"])]
    records = []
    for (protocol, model), sub in frame.groupby(["protocol", "model"]):
        records.append({"Protocol": protocol, "Model": model, "Macro F1 mean (SD)": f"{sub.macro_f1.mean():.3f} ({sub.macro_f1.std():.3f})",
                        "AP": fmt(sub.pr_auc.mean()), "MCC": fmt(sub.mcc.mean()), "FPR": fmt(sub.fpr.mean())})
    return pd.DataFrame(records)


def build():
    root = Path.cwd()
    output = root / "output"
    output.mkdir(exist_ok=True)
    blocks = []
    def h(title): blocks.append(dict(type="heading", text=title))
    def p(text): blocks.append(dict(type="paragraph", text=text))
    def image(path, caption):
        if (root / path).exists():
            blocks.append(dict(type="image", path=path, caption=caption))
    suites = {}
    for name in ["public", "collection", "synthetic"]:
        folder = root / "results" / name
        suites[name] = dict(metrics=pd.read_csv(folder / "metrics.csv"), groups=pd.read_csv(folder / "group_metrics.csv"),
            curves=pd.read_csv(folder / "selective_curves.csv"), graphs=pd.read_csv(folder / "graph_statistics.csv"),
            runtime=pd.read_csv(folder / "runtime.csv"), metadata=json.loads((folder / "dataset_metadata.json").read_text()),
            audit=json.loads((folder / "audit/audit.json").read_text()), analysis=json.loads((folder / "analysis_summary.json").read_text()))
    public, collection, synthetic = [suites[n] for n in ["public", "collection", "synthetic"]]
    h("Outcome and evidence scope")
    pr = public["metrics"].query("experiment == 'E1'")
    cm = collection["metrics"].query("experiment == 'E1'")
    rf = pr[pr.model == "RF"].macro_f1
    proposed = pr[pr.model == "Proposed"].macro_f1
    p("The software pipeline is implemented and executed as three bounded studies. On the real Cresci-2017 random-account pilot, RF achieved mean Macro-F1 " + f"{rf.mean():.3f} (SD {rf.std():.3f}), compared with {proposed.mean():.3f} (SD {proposed.std():.3f}) for the proposed relational detector. These runs do not establish superiority of the proposed detector.")
    for model in ["RF", "Proposed"]:
        values = cm[cm.model == model].groupby("protocol").macro_f1.mean()
        if {"P0", "P1"}.issubset(values.index):
            p(f"In the real multi-corpus collection pilot, {model} changed from {values.P0:.3f} under random accounts to {values.P1:.3f} under collection holdout, a P0-minus-P1 difference of {values.P0-values.P1:.3f}. Collection categories are proxy groups, with historical dependencies and differing event availability; these differences cannot be attributed uniquely to verified campaign shift.")
    total_runs = sum(v["analysis"]["detection_runs"] for v in suites.values())
    p(f"The saved evidence contains {total_runs} trained model/split/seed detection evaluations across the three studies, plus separate robustness, calibration and transfer evaluations. All E1-E8 have measured outputs in supported settings. Real chronological evaluation, verified-campaign validation, full temporal external transfer and several official advanced baseline reproductions remain unsupported. Synthetic results validate implementation behavior and are separate from real empirical evidence.")
    h("Dataset preparation and audit")
    rows = []
    for name, values in suites.items():
        a = values["audit"]
        rows.append({"Study": name, "Accounts": a["accounts"], "Human": a["humans"], "Bot": a["bots"], "Object records": a["events"], "Groups": a["groups"]})
    blocks.append(table(pd.DataFrame(rows)))
    p("Cresci archives were obtained from the academic Bot Repository. The bounded source pilot selects at most 120 accounts per original collection by fixed SHA-256 order using seed 2026, and at most 40 valid tweets per selected account in archive order. Shared objects are observed retweet/reply post IDs, hashtags or URLs. If none is present, the tweet's own ID preserves its activity record. Object IDs are hashed. Multiple objects can create several records for one physical tweet; rhythm features and robustness sampling count physical events consistently.")
    p("The collection pool combines the Cresci-2017 pilot, Cresci-2015 pilot and a fixed 120-per-class Gilani snapshot sample. Every sampled account ID appearing in multiple sources is removed from all pooled sources. Original collection names are prefixed by corpus and used only for splitting, calibration and training group objectives. They never enter the predictor features or shared-object identifiers. Original source ID universes are retained so external evaluation excludes even IDs outside the sampled source subset.")
    p("Profile snapshots include log-transformed followers, following, statuses, listed count and description length, plus available verified/default-profile values and missingness flags. Missingness is material: Cresci CSV blank flags remain missing rather than being reconstructed as false. Target JSON booleans are observed values. This cross-format difference is disclosed and can harm transfer. A source-fitted median imputer and scaler are reused unchanged on target data.")
    h("Model and graph specification")
    p("The rhythm vector contains a normalized 168-bin weekly activity histogram; log event count and span; median inter-event gap; gap coefficient of variation; short-gap fraction; weekly entropy; a spectral-concentration summary; and missingness indicators. Profile and rhythm extraction are deterministic and use only events at or before a declared cutoff when one is supported. Labels, account IDs, collection identifiers and prediction times are excluded from numerical predictor inputs.")
    p("The temporal kernel is k(delta) = 0.40 exp(-delta/10) + 0.35 exp(-delta/60) + 0.25 exp(-delta/600), with seconds as the unit and a 3600-second candidate window. Raw pair weights sum this kernel over shared-object event pairs. Each object is capped at 64 sampled records in the pilot. Eight null replicates permute timestamps among events within each object and UTC day, preserving per-account object/day counts and the object's timestamp distribution. The corrected edge is max(0, (raw - null_mean)/(null_std + 0.001)); null SD below 0.01 removes the edge.")
    p("The full graph additionally requires two shared objects, corrected weight at least 1.0 and mutual top-10 membership, so undirected degree is at most ten. Positive corrected weight is excess synchrony under this specific null, not a significance test or evidence of malicious behavior. Node graph statistics are log degree, log weighted strength and clustering. A separate synchronized-event-fraction feature is not implemented.")
    p("The proposed network has two hidden layers of width 32 with ReLU and dropout 0.15. It uses separate self and relation transforms with row-normalized weighted-mean messages; observed follow links form a separate directed relation. In the Cresci-2017 pilot, follow links are unavailable, so this relation is empty. The feature-only ablations use the same training loop with an empty adjacency. GraphSAGE uses a full-neighborhood mean transform with L2 normalization; it is a local implementation, not a benchmark reproduction from official code.")
    p("RF, RBF SVM and logistic regression use common profile features. Group-DRO updates training-group weights exponentially; equal-group reweighting averages group losses. The neural candidates use AdamW, weight decay 0.01, learning rates 0.003 and 0.01, up to 50 epochs and validation patience eight. RF uses 80 trees and depth candidates four and twelve; classical C candidates are four and twelve. There is no test-driven threshold or model selection.")
    h("Split protocols and leakage boundaries")
    p("Code follows the detailed guide: P0 stratified account random; P1 four-way group-disjoint; P2 distinct repeated group holdout using seed offset 10000; P3 chronological observation; P4 source-to-target transfer. The manuscript experiment section numbers chronology and transfer differently. Three run seeds are 11, 23 and 37. Intended fractions are 0.50/0.15/0.20/0.15, with integer allocation and group constraints documented in each manifest.")
    p("Every model uses the same saved IDs within each run. Imputers and scalers fit only training accounts. Validation selects settings; calibration fits the fixed detector's reliability layer; final test labels enter metrics only. Graphs and message passing are partition-local. P0 can share group identity across partitions, so its group calibration is purely descriptive. P1/P2 prevent group overlap but do not establish exchangeability. P3 requires historical profile availability and strictly separated prediction cutoffs; creation timestamps alone are rejected.")
    image("results/collection/figures/01_split_manifest.png", "Figure 1  Saved collection-disjoint split for seed 11")
    h("E1 Random versus collection holdout")
    p("The Cresci-2017 source sample has only one human collection. A four-way group-disjoint split with both classes is therefore impossible; P1/P2 are explicitly unsupported there. The multi-corpus study supplies enough declared collection units for a proxy group holdout. Neither study establishes verified-campaign generalization. The following scores are run means; parenthesized values are run SD, not campaign confidence intervals.")
    blocks.append(table(detection(public["metrics"])))
    p("Core model comparison on the collection pool follows. The supplementary tables include every model, per-group results and the paired P0-minus-P1 differences.")
    blocks.append(table(detection(collection["metrics"], core=True)))
    image("results/collection/figures/02_protocol_performance.png", "Figure 2  Collection-study performance across supported protocols")
    h("E2 Contribution of temporal coordination")
    p("All ablations share partitions and the configured model-selection budget. A1 uses profile features; A2 adds rhythm; A3 uses static shared-object graph statistics/messages; A4 uses raw temporal weights; A5 applies null correction without the full support/top-k filter; Proposed adds full filtering and any permitted follow relation. Inputs and parameter counts differ, so these are controlled engineering ablations rather than equal-capacity proofs.")
    rows = []
    for model in ["A1_profile", "A2_rhythm", "A3_static", "A4_temporal", "A5_corrected", "Proposed"]:
        sub = collection["metrics"].query("experiment == 'E2' and protocol == 'P1'")
        sub = sub[sub.model == model]
        rows.append({"Variant": model, "Macro F1": fmt(sub.macro_f1.mean()), "AP": fmt(sub.pr_auc.mean()), "MCC": fmt(sub.mcc.mean()), "FPR": fmt(sub.fpr.mean())})
    blocks.append(table(pd.DataFrame(rows)))
    image("results/collection/figures/03_feature_ablation.png", "Figure 3  Temporal and graph ablations on held-out collections")
    p("The real collection pilot does not show a consistent gain from the complete model. Synthetic improvements are not used to override these results. Missing events in the Gilani component, class-homogeneous collections, small group counts and null-model assumptions are material alternative explanations.")
    h("E3 Calibration and group reliability")
    p("Prediction sets contain class c when 1 - p(c) <= q. Account split conformal uses the ceil((n+1)(1-alpha)) order statistic with a full-set boundary when the rank exceeds n. The group-maximum baseline calibrates maximum account nonconformity within each calibration group. Campaign CRC averages group mean miscoverage and selects the smallest predefined q satisfying (sum of group losses + 1)/(K+1) <= alpha. Its grid is 0, 0.001, ..., 1 and alpha is 0.10.")
    p("The finite-sample CRC criterion cannot fall below 1/(K+1). Infeasible calibration is recorded as infeasible and emits full sets as an abstention fallback. Group-max and mean-loss CRC target different quantities. Formal guarantees are not claimed: group exchangeability, fixed-predictor sampling conditions and independence of proxy collection units have not been established. Temperature and confidence thresholds are fit on validation predictions and carry no selective-risk guarantee.")
    cal = collection["metrics"].query("experiment == 'E3' and protocol == 'P1'")
    rows = []
    for method, sub in cal.groupby("calibrator"):
        rows.append({"Method": method, "Set coverage": fmt(sub.empirical_coverage.mean()), "Mean group loss": fmt(sub.group_miscoverage_mean.mean()), "Set size": fmt(sub.mean_set_size.mean()), "Auto rate": fmt(sub.singleton_rate.mean()), "Auto accuracy": fmt(sub.selective_accuracy.mean())})
    blocks.append(table(pd.DataFrame(rows)))
    image("results/collection/figures/04_group_miscoverage.png", "Figure 4  Observed per-group miscoverage; repeated group/run observations are descriptive")
    h("E4 Null correction and graph construction")
    gt = collection["graphs"]
    gt = gt[(gt.protocol == "P1") & (gt.partition == "train")]
    rows = []
    for variant in ["none", "static", "temporal", "corrected", "full"]:
        sub = gt[gt.graph_variant == variant]
        rows.append({"Graph": variant, "Mean edges": fmt(sub.edges.mean(), 1), "Mean density": fmt(sub.density.mean(), 5), "Mean max degree": fmt(sub.max_degree.mean(), 1), "Stable null fraction": fmt(sub.stable_null_fraction.mean())})
    blocks.append(table(pd.DataFrame(rows)))
    image("results/collection/figures/05_graph_ablation.png", "Figure 5  Training graph edge counts before and after correction/filtering")
    sensitivity = []
    for name in suites:
        path = root / "results" / name / "training_graph_sensitivity.csv"
        if path.exists():
            sub = pd.read_csv(path).iloc[0]
            sensitivity.append({"Study": name, "Train accounts": int(sub.accounts), "Raw correlation": fmt(sub.raw_correlation), "Edge Jaccard": fmt(sub.retained_edge_jaccard), "Pilot edges": int(sub.pilot_retained_edges), "Reference edges": int(sub.reference_retained_edges)})
    if sensitivity:
        blocks.append(table(pd.DataFrame(sensitivity)))
    p("The training-only diagnostic compares the pilot cap/permutation budget with an uncapped 32-permutation reference on up to 128 fixed training accounts. It jointly varies sampling and Monte Carlo precision, so it is a sensitivity check rather than an isolated causal comparison. The full-scale approximation is not thereby proven accurate. Sparse or unstable corrected graphs are retained as measured outcomes.")
    h("E5 Selective prediction and review workload")
    p("Operating alpha values 0.05, 0.10, 0.20 and 0.30 and confidence thresholds 0.50, 0.60, 0.70, 0.80, 0.90, 0.95 and 0.99 were fixed before evaluation. Only singleton sets produce automatic labels; empty and two-label sets abstain. Selective accuracy is undefined when no labels are accepted. Set coverage includes non-singleton true-label inclusion and must not be read as automatic-decision accuracy.")
    image("results/collection/figures/06_selective_tradeoff.png", "Figure 6  Accuracy among automatic decisions versus singleton rate at predefined operating points")
    p("At alpha 0.10 the real Cresci-2017 campaign CRC returned q=1 and full two-class sets, yielding complete review workload. In the collection-disjoint study, few calibration units make the correction infeasible. The result tables report these outcomes alongside confidence and account-level baselines. Matched-coverage comparisons choose the nearest measured point descriptively, report the achieved distance, and do not select a deployment threshold using test labels.")
    h("E6 Robustness to incomplete and perturbed activity")
    p("Only frozen test inputs are perturbed; model weights, source preprocessing, calibration and labels stay fixed. Whole physical events are removed at fractions 0.25, 0.50 and 0.75. Timestamp noise is uniform within +/-30, +/-300 or +/-900 seconds, applied once per physical event across all of its object records and clipped to supported prediction cutoffs. Low activity uses the training 25th percentile of unique event counts. Group-size strata are <=10, 11-30 and >30 accounts in the evaluated partition.")
    robust = collection["metrics"].query("experiment == 'E6'")
    rows = []
    for condition, sub in robust[robust.model.isin(["RF", "Proposed"])].groupby("condition"):
        values = sub.groupby("model").macro_f1.mean()
        rows.append({"Condition": condition, "RF Macro F1": fmt(values.get("RF", np.nan)), "Proposed Macro F1": fmt(values.get("Proposed", np.nan))})
    blocks.append(table(pd.DataFrame(rows)))
    image("results/collection/figures/07_robustness.png", "Figure 7  Frozen-model event-removal and jitter results")
    p("No independently annotated benign synchronized real groups were found, so that real subtest is unsupported. A declared benign-synchronization mechanism is exercised only in the synthetic suite. Empty group-size strata are not fabricated or assigned scores. RF profile predictions should be unchanged by event-only perturbations; this invariant is checked during final validation.")
    h("E7 Computational scalability")
    runtime = collection["runtime"]
    scale = runtime[runtime.component == "E7_total"]
    rows = [{"Accounts": int(r.subset_size), "Total seconds": fmt(r.seconds, 2), "Peak RSS MB": fmt(r.peak_rss_mb, 1)} for r in scale.itertuples()]
    blocks.append(table(pd.DataFrame(rows)))
    image("results/collection/figures/08_scalability.png", "Figure 8  Dedicated CPU workload runtime and process memory")
    env = json.loads((root / "results/collection/environment.json").read_text())
    p(f"The execution environment is {env['platform']}, {env['logical_cpus']} logical CPUs, {env['ram_gb']:.1f} GB system RAM and CPU PyTorch {env['packages']['torch']}; configured PyTorch threads are two. Stage timers measure raw graph construction, null estimation, model selection/training, calibration and inference. RSS is sampled every 10 ms and includes native allocations and previously loaded libraries. No GPU memory is reported.")
    p("The final E7 pass runs one experiment worker at a time. Background desktop activity is uncontrolled. Original E1-E5 training timings may include contention from parallel experiment processes; use dedicated E7 timings for workload comparisons. Total E7 cost includes whole-subset graph construction, partition-local learning and batch inference, so it must not be added to its component times. The pilot supports workload-cost observations, not asymptotic complexity or production-throughput claims.")
    h("E8 External dataset transfer")
    p("The public source pilot transfers to Gilani-2017 snapshots. The collection source transfers to independently collected RTbust-2019 snapshots. Original source-universe ID overlap is removed before evaluating targets. Source-trained weights, source-fitted preprocessing and source calibration are frozen; no target-label model selection or calibration is performed. Target labels are used only for evaluation.")
    rows = []
    for name in ["public", "collection"]:
        tr = suites[name]["metrics"].query("experiment == 'E8'")
        for r in tr.itertuples():
            rows.append({"Source study": name, "Model": r.model, "Target n": int(r.n), "Macro F1": fmt(r.macro_f1), "AP": fmt(r.pr_auc), "FPR": fmt(r.fpr)})
    blocks.append(table(pd.DataFrame(rows)))
    image("results/public/figures/09_transfer.png", "Figure 9  Compatible profile-only transfer to Gilani")
    p("The target archives contain user snapshots rather than usable timestamped shared-object histories. Only RF, SVM, logistic regression and profile MLP transfer is valid. Rhythm/graph transfer is explicitly unsupported; target events are not invented. Annotation semantics, population, snapshot formats, missing flags and collection times differ. The low observed scores therefore indicate a transfer failure in this pilot, with several plausible causes beyond campaign shift alone. Cresci-2015-to-Cresci-2017 is not used as independent external evidence because their underlying collections overlap.")
    h("Synthetic validation study")
    p("The synthetic source has 48 designed groups of 12 accounts; the target has 24 groups of 12. Groups include designed profile shifts, shared-object synchronization, benign synchronized accounts, observation cutoffs and optional follow links. Target synchronization uses different jitter. The generator has explicit seeds and writes data_kind=synthetic into metadata. Its results exercise all protocol paths, including P3, and are not claims about platform behavior.")
    blocks.append(table(detection(synthetic["metrics"], core=True)))
    h("Statistical uncertainty and limitations")
    p("AP denotes average precision for bot=1. Macro-F1 uses both class labels with zero undefined class precision/recall; single-class AP/ROC and zero-denominator MCC/FPR are missing. ECE uses ten equal-width confidence bins. Per-run Macro-F1 intervals resample whole groups with a percentile bootstrap and are suppressed below five observed groups. Repeated seeds/holdouts are not extra independent campaigns. Small, class-homogeneous and historically related collection units restrict inferential strength.")
    statistics_path = root / "results/collection/tables/paired_statistics.csv"
    if statistics_path.exists():
        stats = pd.read_csv(statistics_path)
        rows = [{"Comparison": r.comparison, "Groups": int(r.paired_units), "Mean difference": fmt(r.mean_difference), "p value": fmt(r.p_value), "Holm p": fmt(r.holm_p_value)} for r in stats.itertuples()]
        blocks.append(table(pd.DataFrame(rows)))
    p("The predefined primary comparison uses P1 Macro-F1 and a practical-effect reference of 0.02. The three paired exploratory comparisons are Proposed against A2 rhythm, A4 temporal and GraphSAGE. Repeated per-group observations are averaged before paired Wilcoxon analysis; Holm adjustment applies within that family. Proxy-group dependence, overlapping model training sets, few groups and exploratory data assembly prevent a strong confirmatory or causal interpretation. Complete CSVs preserve mixed and negative results.")
    h("Completion boundary and remaining research")
    p("Delivered work includes the implementation, executed supported E1-E8 pilots, raw/processed data separation, deterministic preparation, audit and split manifests, model checkpoints, prediction exports, all metrics, plots, runtime/memory measurements, automated checks and reproduction commands. The work should be presented as an implemented and evaluated research prototype.")
    p("A completed claim of verified campaign-shift robustness still needs independently annotated campaigns, sufficient independent calibration/test campaigns, chronological profile histories, compatible temporal external datasets and larger compute/data runs. Official BotRGCN, RGT, TGN/TGAT, BotDGT, BotSTIP and IRM reproductions are not included. The local relational model is not renamed as any of those methods. No missing score, theoretical guarantee, full-benchmark result or successful reproduction is invented.")
    h("Reproduction and verification")
    p("README.md gives exact setup, dataset preparation, experiment, analysis and inference commands. run_all.ps1 runs the studies sequentially and stops on failure. requirements.lock.txt records the tested packages. Configurations, environment files, SHA-256 input hashes, selected epochs/learning rates, training histories and saved partitions are retained. The standalone inference command reconstructs a checkpoint without target-label training and produces human, bot or needs_review with prediction-set flags.")
    test_log = root / "logs/tests.log"
    if test_log.exists():
        matches = re.findall(r"\d+ passed(?:[^\r\n]*)", test_log.read_text(errors="replace"))
        if matches:
            p("Latest automated test result: " + matches[-1] + ". Tests cover group and chronological separation, future events, training-only preprocessing, hand-calculated kernel values, graph reproducibility/degree bounds, finite-sample calibration, undefined selective accuracy, multi-object perturbation identity and checkpoint inference round trips.")
    p("The final validation manifest checks prediction probabilities, split uniqueness and group separation, P1/P2 holdout distinction, event-only RF invariance, calibration feasibility handling and the presence of measured E1-E8 outputs in the supported suites. PNG/SVG charts are regenerated directly from saved CSVs. Report figures omit account-identifying examples.")
    h("Sources")
    for text in [
        "User-supplied references: Campaign_Shift_Social_Bot_Detection_Detailed_Experimental_Guide.docx; Campaign_Shift_Aware_Social_Bot_Detection_With_Experiments.pdf; Campaign_Shift_Aware_Social_Bot_Detection_Third_Revision.pdf. These specify methodology, not completed empirical results.",
        "Bot Repository: https://botometer.osome.iu.edu/bot-repository/datasets.html . Original archive research-use notices are retained. Gilani and RTbust metadata: datasets/gilani-2017/info.json and datasets/cresci-rtbust-2019/info.json under that repository.",
        "Cresci et al. The paradigm-shift of social spambots. WWW Companion 2017. https://arxiv.org/abs/1701.03017 .",
        "Hamilton, Ying and Leskovec. Inductive Representation Learning on Large Graphs. NeurIPS 2017. https://proceedings.neurips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf .",
        "Angelopoulos, Bates, Fisch, Lei and Schuster. Conformal Risk Control. https://arxiv.org/abs/2208.02814 . Official implementation: https://github.com/aangelopoulos/conformal-risk .",
        "TwiBot-20 official access documentation: https://github.com/BunsenFeng/TwiBot-20 . Its public sample lacks labels and the full dataset requires author access; no access request was sent."
    ]: p(text)
    content = dict(title="Campaign Shift Aware Social Bot Detection", subtitle="Implementation and Executed Experimental Report", date="19 October 2026", blocks=blocks)
    dump_json(output / "report_content.json", content)
    markdown = ["# " + content["title"], "", content["subtitle"] + " | " + content["date"], ""]
    for b in blocks:
        if b["type"] == "heading": markdown += ["## " + b["text"], ""]
        elif b["type"] == "paragraph": markdown += [b["text"], ""]
        elif b["type"] == "image": markdown += [f"![{b['caption']}](../{b['path']})", ""]
        else:
            markdown += ["| " + " | ".join(b["headers"]) + " |", "| " + " | ".join(["---"] * len(b["headers"])) + " |"]
            markdown += ["| " + " | ".join(row) + " |" for row in b["rows"]]
            markdown += [""]
    (output / "Project_Report.md").write_text("\n".join(markdown), encoding="utf-8")
    print("Report content generated from", total_runs, "detection evaluations")


if __name__ == "__main__":
    build()
