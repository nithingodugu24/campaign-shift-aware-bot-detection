from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

from .calibrate import probabilities


def detection_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = probabilities(p)
    if not len(y):
        return {}
    pred = p.argmax(1)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    confidence, correct = p.max(1), (pred == y)
    ece = 0.
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    both = len(np.unique(y)) == 2
    mcc_defined = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0
    return dict(n=len(y), macro_f1=float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
                accuracy=float(accuracy_score(y, pred)), pr_auc=float(average_precision_score(y, p[:, 1])) if both else np.nan,
                roc_auc=float(roc_auc_score(y, p[:, 1])) if both else np.nan,
                mcc=float(matthews_corrcoef(y, pred)) if mcc_defined else np.nan,
                fpr=float(fp / (fp + tn)) if fp + tn else np.nan,
                brier_score=float(brier_score_loss(y, p[:, 1])), log_loss=float(log_loss(y, p, labels=[0, 1])), ece=float(ece),
                human_precision=float(prec[0]), human_recall=float(recall[0]), human_f1=float(f1[0]),
                bot_precision=float(prec[1]), bot_recall=float(recall[1]), bot_f1=float(f1[1]),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), single_class=not both)


def selective_metrics(y, sets, groups=None):
    y = np.asarray(y, dtype=int)
    sizes = sets.sum(1)
    accepted = sizes == 1
    covered = sets[np.arange(len(y)), y]
    unique_label = sets.argmax(1)
    accepted_benign = accepted & (y == 0)
    result = dict(empirical_coverage=float(covered.mean()), mean_set_size=float(sizes.mean()),
                  singleton_rate=float(accepted.mean()), abstention_rate=float(1 - accepted.mean()),
                  empty_set_rate=float((sizes == 0).mean()),
                  selective_accuracy=float((unique_label[accepted] == y[accepted]).mean()) if accepted.any() else np.nan,
                  selective_fpr=float((unique_label[accepted_benign] == 1).mean()) if accepted_benign.any() else np.nan)
    if accepted.any():
        result["selective_macro_f1"] = float(f1_score(y[accepted], unique_label[accepted], labels=[0, 1], average="macro", zero_division=0))
    else:
        result["selective_macro_f1"] = np.nan
    for cls in [0, 1]:
        result[f"class_{cls}_coverage"] = float(covered[y == cls].mean()) if (y == cls).any() else np.nan
    if groups is not None:
        groups = np.asarray(groups)
        losses = np.asarray([1 - covered[groups == g].mean() for g in np.unique(groups)])
        result.update(group_miscoverage_mean=float(losses.mean()), group_miscoverage_worst=float(losses.max()),
                      group_miscoverage_median=float(np.median(losses)), group_miscoverage_iqr=float(np.quantile(losses, .75) - np.quantile(losses, .25)), test_groups=len(losses))
    return result


def group_metrics(y, p, sets, groups):
    groups = np.asarray(groups)
    records = []
    for group in np.unique(groups):
        mask = groups == group
        row = dict(group_id=group, group_size=int(mask.sum()), **detection_metrics(np.asarray(y)[mask], np.asarray(p)[mask]))
        if sets is not None:
            row.update(selective_metrics(np.asarray(y)[mask], sets[mask]))
            row["miscoverage"] = 1 - row["empirical_coverage"]
        records.append(row)
    return records


def cluster_bootstrap(y, p, groups, seed=0, repeats=300):
    """Resample whole test groups. Seeds are never counted as independent groups."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 5:
        return dict(macro_f1_ci_low=np.nan, macro_f1_ci_high=np.nan, ci_groups=len(unique), ci_note="Fewer than five groups; descriptive results only")
    rng = np.random.default_rng(seed)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(repeats):
        take = np.concatenate([indices[g] for g in rng.choice(unique, len(unique), replace=True)])
        values.append(f1_score(np.asarray(y)[take], np.asarray(p)[take].argmax(1), labels=[0, 1], average="macro", zero_division=0))
    return dict(macro_f1_ci_low=float(np.quantile(values, .025)), macro_f1_ci_high=float(np.quantile(values, .975)), ci_groups=len(unique), ci_note="Within-run percentile group bootstrap; few groups imply unstable intervals")


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1., np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return result


def paired_comparison(left, right):
    delta = np.asarray(left) - np.asarray(right)
    delta = delta[np.isfinite(delta)]
    if len(delta) < 5:
        return dict(paired_units=len(delta), mean_difference=float(delta.mean()) if len(delta) else np.nan, p_value=np.nan)
    p = 1. if np.all(delta == 0) else float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
    return dict(paired_units=len(delta), mean_difference=float(delta.mean()), p_value=p)
