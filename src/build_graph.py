from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class GraphBundle:
    matrices: dict
    edges: pd.DataFrame
    stats: dict
    timings: dict


def _scores(objects, config):
    scores = defaultdict(float)
    support = defaultdict(set)
    synchronized_events = set()
    window = config["window_seconds"]
    scales = np.asarray(config["scales_seconds"])
    weights = np.asarray(config["scale_weights"])
    observation_units = config.get("_observation_units")
    for oid, users, times, event_ids in objects:
        order = np.argsort(times, kind="stable")
        users, times, event_ids = users[order], times[order], event_ids[order]
        for i in range(len(times)):
            stop = np.searchsorted(times, times[i] + window, side="right")
            if stop <= i + 1:
                continue
            other = users[i + 1:stop]
            valid = other != users[i]
            if observation_units is not None:
                valid &= observation_units[other] == observation_units[users[i]]
            delta = times[i + 1:stop][valid] - times[i]
            values = np.exp(-delta[:, None] / scales) @ weights
            for v, value, event_id in zip(other[valid], values, event_ids[i + 1:stop][valid]):
                pair = (min(int(users[i]), int(v)), max(int(users[i]), int(v)))
                scores[pair] += float(value)
                support[pair].add(oid)
                synchronized_events.update([int(event_ids[i]), int(event_id)])
    return scores, support, synchronized_events


def _adjacency(n, pairs, values):
    if not pairs:
        return sparse.csr_matrix((n, n), dtype=np.float32)
    u, v = np.asarray(pairs).T
    matrix = sparse.csr_matrix((np.r_[values, values], (np.r_[u, v], np.r_[v, u])), shape=(n, n), dtype=np.float32)
    matrix.eliminate_zeros()
    return matrix


def build_graph(dataset, config, seed=0):
    """Partition-local graphs only. Candidate truncation is declared, never label-dependent."""
    started = time.perf_counter()
    a, e = dataset.accounts, dataset.events
    n = len(a)
    idx = dict(zip(a.account_id, range(n)))
    # Node features belong to each prediction cutoff. Cross-cutoff messages could
    # expose a later node's profile/history even when a particular edge event is old.
    observation_units = a.prediction_at.dt.as_unit("ns").astype("int64").to_numpy()
    score_config = dict(config, _observation_units=observation_units)
    rng = np.random.default_rng(seed)
    objects, static = [], defaultdict(float)
    skipped = capped = 0
    for oid, frame in e.groupby("object_id", sort=True):
        if frame.account_id.nunique() < 2:
            continue
        if len(frame) > config["object_event_cap"]:
            # Fixed random sample, shared by raw and all null replicates.
            frame = frame.iloc[np.sort(rng.choice(len(frame), config["object_event_cap"], replace=False))]
            capped += 1
        users = frame.account_id.map(idx).to_numpy(dtype=int)
        times = frame.timestamp.dt.as_unit("ns").astype("int64").to_numpy() / 1e9
        event_ids = frame.index.to_numpy(dtype=int)
        objects.append((oid, users, times, event_ids))
        unique = np.unique(users)
        for i, u in enumerate(unique):
            for v in unique[i + 1:]:
                if observation_units[u] == observation_units[v]:
                    static[(int(u), int(v))] += 1
    raw, support, sync = _scores(objects, score_config)
    raw_seconds = time.perf_counter() - started
    pairs = sorted(raw)
    null_started = time.perf_counter()
    mean = np.zeros(len(pairs))
    m2 = np.zeros(len(pairs))
    for b in range(config["null_permutations"]):
        shuffled = []
        for oid, users, times, event_ids in objects:
            randomized = times.copy()
            blocks = np.floor(times / config["null_block_seconds"]).astype(np.int64)
            # Preserve object-level timestamp distribution and each account's object/block event counts.
            for block in np.unique(blocks):
                positions = np.flatnonzero(blocks == block)
                randomized[positions] = rng.permutation(times[positions])
            shuffled.append((oid, users, randomized, event_ids))
        null, _, _ = _scores(shuffled, score_config)
        values = np.asarray([null.get(pair, 0.) for pair in pairs])
        delta = values - mean
        mean += delta / (b + 1)
        m2 += delta * (values - mean)
    sd = np.sqrt(m2 / max(1, config["null_permutations"] - 1))
    raw_values = np.asarray([raw[p] for p in pairs])
    z = np.maximum(0, (raw_values - mean) / (sd + config["epsilon"]))
    stable = sd >= config["minimum_null_std"]
    z[~stable] = 0
    supported = np.asarray([len(support[p]) >= config["minimum_shared_objects"] for p in pairs], dtype=bool)
    eligible = supported & (z >= config["minimum_weight"]) & (z > 0)
    neighbors = defaultdict(list)
    for j in np.flatnonzero(eligible):
        u, v = pairs[j]
        neighbors[u].append((z[j], v))
        neighbors[v].append((z[j], u))
    top = {u: {v for _, v in sorted(ns, key=lambda t: (-t[0], t[1]))[:config["top_k"]]} for u, ns in neighbors.items()}
    # Mutual top-k enforces the promised degree bound on the undirected graph.
    keep = np.asarray([eligible[j] and v in top.get(u, set()) and u in top.get(v, set()) for j, (u, v) in enumerate(pairs)], dtype=bool)
    matrices = {
        "none": sparse.csr_matrix((n, n), dtype=np.float32),
        "static": _adjacency(n, list(static), list(static.values())),
        "temporal": _adjacency(n, pairs, raw_values),
        "corrected": _adjacency(n, pairs, z),
        "full": _adjacency(n, [pairs[j] for j in np.flatnonzero(keep)], z[keep]),
    }
    follows = dataset.edges
    if len(follows):
        same_cutoff = observation_units[follows.source.map(idx).to_numpy()] == observation_units[follows.target.map(idx).to_numpy()]
        follows = follows.loc[same_cutoff]
    if len(follows):
        # Matrix row receives messages from followed accounts; directed relation is kept distinct.
        matrices["follows"] = sparse.csr_matrix((np.ones(len(follows)), (follows.source.map(idx), follows.target.map(idx))), shape=(n, n), dtype=np.float32)
    else:
        matrices["follows"] = sparse.csr_matrix((n, n), dtype=np.float32)
    edges = pd.DataFrame([dict(source=a.account_id.iloc[u], target=a.account_id.iloc[v], raw=raw_values[j], null_mean=mean[j], null_std=sd[j], corrected=z[j], shared_objects=len(support[(u, v)]), retained=bool(keep[j])) for j, (u, v) in enumerate(pairs)], columns=["source", "target", "raw", "null_mean", "null_std", "corrected", "shared_objects", "retained"])
    stats = {}
    for name, mat in matrices.items():
        graph = nx.from_scipy_sparse_array(mat)
        degrees = np.diff(mat.indptr)
        stats[name] = dict(nodes=n, edges=int(mat.nnz // (1 if name == "follows" else 2)),
                           density=float(mat.nnz / max(1, n * (n - 1))), mean_degree=float(degrees.mean()),
                           max_degree=int(degrees.max(initial=0)), components=nx.number_connected_components(graph),
                           mean_weight=float(mat.data.mean()) if mat.nnz else 0,
                           capped_objects=capped, candidate_pairs=len(pairs),
                           stable_null_fraction=float(stable.mean()) if len(stable) else 0,
                           support_fraction=float(supported.mean()) if len(supported) else 0)
    return GraphBundle(matrices, edges, stats, dict(graph_raw_seconds=raw_seconds, null_seconds=time.perf_counter() - null_started))


def coordination_features(matrix):
    graph = nx.from_scipy_sparse_array(matrix)
    degree = np.diff(matrix.indptr)
    strength = np.asarray(matrix.sum(axis=1)).ravel()
    clustering = nx.clustering(graph)
    return np.column_stack([np.log1p(degree), np.log1p(strength), [clustering[i] for i in range(matrix.shape[0])]])


def normalized(matrix):
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    return (sparse.diags(1 / np.maximum(sums, 1e-12)) @ matrix).astype(np.float32)
