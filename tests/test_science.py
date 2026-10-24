import copy

import numpy as np
import pandas as pd
import pytest

from src.build_graph import build_graph, _scores
from src.calibrate import conformal_quantile, fit_calibrator, score_sets
from src.data_audit import load_dataset, Dataset
from src.evaluate import selective_metrics, detection_metrics, holm
from src.preprocess import account_features, fit_preprocessor
from src.run_experiments import load_config
from src.splits import make_split, validate_split, UnsupportedProtocol
from src.synthetic import generate
from src.robustness import perturb_events


@pytest.fixture
def fixture(tmp_path):
    generate(tmp_path, groups=16, accounts_per_group=8)
    return load_dataset(tmp_path)


def test_group_splits_are_disjoint_and_reproducible(fixture):
    one = make_split(fixture, "P1", 11)
    two = make_split(fixture, "P1", 11)
    pd.testing.assert_frame_equal(one, two)
    assert one.groupby("group_id").partition.nunique().max() == 1
    assert not one.account_id.duplicated().any()
    repeated = make_split(fixture, "P2", 11)
    assert not one.partition.equals(repeated.partition)
    one.loc[0, "partition"] = "test" if one.loc[0, "partition"] != "test" else "train"
    with pytest.raises(AssertionError, match="Group leakage"):
        validate_split(fixture, one, "P1")


def test_future_events_and_duplicate_ids_rejected(tmp_path):
    generate(tmp_path, groups=16, accounts_per_group=8)
    e = pd.read_csv(tmp_path / "events.csv")
    e.loc[0, "timestamp"] = "2099-01-01T00:00:00Z"
    e.to_csv(tmp_path / "events.csv", index=False)
    data = load_dataset(tmp_path, tmp_path / "audit")
    assert len(data.events) == len(e) - 1
    a = pd.read_csv(tmp_path / "accounts.csv")
    pd.concat([a, a.iloc[:1]]).to_csv(tmp_path / "accounts.csv", index=False)
    with pytest.raises(ValueError, match="unique"):
        load_dataset(tmp_path)


def test_chronology_requires_historical_availability(fixture):
    m = make_split(fixture, "P3", 11)
    validate_split(fixture, m, "P3")
    fixture.metadata["chronology_supported"] = False
    with pytest.raises(UnsupportedProtocol, match="Historical"):
        make_split(fixture, "P3", 11)


def test_training_only_imputer():
    train = np.array([[1., np.nan], [3., 4.]])
    pre = fit_preprocessor(train)
    center = pre.named_steps["standardscaler"].mean_.copy()
    pre.transform([[10000., 9999.]])
    assert np.array_equal(center, pre.named_steps["standardscaler"].mean_)
    assert center[0] == 2


def test_kernel_matches_hand_calculation():
    config = load_config("configs/pilot.yaml")["graph"]
    config.update(scales_seconds=[10.], scale_weights=[1.], window_seconds=30)
    objects = [("x", np.array([0, 1, 0]), np.array([0., 10., 100.]), np.array([0, 1, 2]))]
    scores, support, _ = _scores(objects, config)
    assert scores[(0, 1)] == pytest.approx(np.exp(-1))
    assert len(support[(0, 1)]) == 1


def test_graph_reproducibility_and_degree_bound(fixture):
    config = load_config("configs/pilot.yaml")["graph"]
    config.update(null_permutations=3, top_k=2)
    small = fixture.subset(fixture.accounts.account_id[:24])
    one = build_graph(small, config, 19)
    two = build_graph(small, config, 19)
    pd.testing.assert_frame_equal(one.edges, two.edges)
    assert np.diff(one.matrices["full"].indptr).max() <= 2
    assert np.isfinite(one.matrices["corrected"].data).all()
    assert (one.matrices["corrected"].data > 0).all()
    assert one.stats["corrected"]["edges"] == int((one.edges.corrected > 0).sum())
    assert (one.matrices["full"] != one.matrices["full"].T).nnz == 0
    assert set(one.edges.source).issubset(set(small.accounts.account_id))
    # A later node's feature history must not enter an earlier node's graph.
    pair = fixture.subset(fixture.accounts.account_id[:2])
    pair.accounts.loc[1, "prediction_at"] += pd.Timedelta(days=1)
    separated = build_graph(pair, config, 19)
    assert all(matrix.nnz == 0 for matrix in separated.matrices.values())


def test_conformal_finite_sample_rank_and_crc_infeasibility():
    assert conformal_quantile([.1, .2, .3], .25) == .3
    assert conformal_quantile([.1, .2, .3], .1) == 1
    p = np.array([[.9, .1], [.2, .8], [.7, .3], [.1, .9]])
    y = np.array([0, 1, 0, 1])
    cal = fit_calibrator("campaign_crc", p, y, ["a", "a", "b", "b"], .1)
    assert not cal.feasible and cal.q == 1
    assert cal.corrected_calibration_risk == pytest.approx(1/3)
    cal = fit_calibrator("campaign_crc", p, y, ["a", "a", "b", "b"], .4)
    assert cal.feasible
    assert cal.corrected_calibration_risk <= .4


def test_set_metrics_do_not_confuse_abstention_and_accuracy():
    y = np.array([0, 1])
    full = selective_metrics(y, np.ones((2, 2), dtype=bool))
    assert full["empirical_coverage"] == 1 and full["singleton_rate"] == 0
    assert np.isnan(full["selective_accuracy"])
    empty = selective_metrics(y, np.zeros((2, 2), dtype=bool))
    assert empty["empirical_coverage"] == 0 and empty["abstention_rate"] == 1
    p = np.array([[.8, .2], [.1, .9]])
    assert (score_sets(p, .4) <= score_sets(p, .9)).all()


def test_single_class_metrics_and_holm():
    result = detection_metrics([0, 0], [[.7, .3], [.8, .2]])
    assert np.isnan(result["roc_auc"]) and np.isnan(result["pr_auc"])
    assert np.isnan(result["mcc"])
    assert holm([.01, .03, .2]).tolist() == pytest.approx([.03, .06, .2])


def test_perturbations_preserve_multi_object_event_identity(fixture):
    e = fixture.events.iloc[:1].copy()
    e["object_id"] = "another_object"
    fixture.events = pd.concat([fixture.events, e], ignore_index=True)
    jitter = perturb_events(fixture, "jitter", 300, 12)
    times = jitter.events.groupby(["account_id", "event_id"]).timestamp.nunique()
    assert times.max() == 1
    removed = perturb_events(fixture, "remove", .5, 12)
    original_counts = fixture.events.groupby(["account_id", "event_id"]).size()
    counts = removed.events.groupby(["account_id", "event_id"]).size()
    assert (counts == original_counts.loc[counts.index]).all()


def test_unlabeled_input_without_optional_cutoffs(tmp_path):
    generate(tmp_path, groups=16, accounts_per_group=8)
    a = pd.read_csv(tmp_path / "accounts.csv")
    a = a.drop(columns=["label", "prediction_at", "profile_observed_at"])
    a.to_csv(tmp_path / "accounts.csv", index=False)
    data = load_dataset(tmp_path, require_labels=False)
    assert len(data.events) > 0 and (data.accounts.label == -1).all()
