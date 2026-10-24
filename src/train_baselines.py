from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from torch import nn

from .build_graph import normalized
from .utils import seed_everything


def tensor_graph(matrix):
    coo = normalized(matrix).tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(indices, torch.tensor(coo.data, dtype=torch.float32), coo.shape, check_invariants=True).coalesce()


class RelationalNet(nn.Module):
    """Two layers of separate self and relation-specific weighted-mean transforms."""
    def __init__(self, input_dim, hidden=32, relations=2, dropout=.15, sage=False):
        super().__init__()
        self.self_layers = nn.ModuleList([nn.Linear(input_dim, hidden), nn.Linear(hidden, hidden)])
        self.relation_layers = nn.ModuleList([nn.ModuleList([nn.Linear(d, hidden, bias=False) for _ in range(relations)]) for d in [input_dim, hidden]])
        self.output = nn.Linear(hidden, 2)
        self.dropout = dropout
        self.sage = sage

    def forward(self, x, graphs):
        for own, transformations in zip(self.self_layers, self.relation_layers):
            h = own(x)
            for transform, graph in zip(transformations, graphs):
                h = h + transform(torch.sparse.mm(graph, x))
            x = torch.relu(h)
            if self.sage:
                x = nn.functional.normalize(x, p=2, dim=1)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.output(x)


@dataclass
class FittedModel:
    estimator: object
    neural: bool
    train_seconds: float
    settings: dict
    history: list

    def predict(self, x, matrices):
        if not self.neural:
            return self.estimator.predict_proba(x)
        self.estimator.eval()
        with torch.no_grad():
            return torch.softmax(self.estimator(torch.tensor(x, dtype=torch.float32), [tensor_graph(m) for m in matrices]), dim=1).numpy()


def train_model(name, train, validation, config, seed):
    """Input tuples: x, y, permitted partition-local adjacency matrices, group IDs."""
    seed_everything(seed, config.get("threads", 2))
    x, y, graphs, groups = train
    xv, yv, vg, _ = validation
    start = time.perf_counter()
    if name in {"RF", "SVM", "LR"}:
        best = None
        for value in config["classical_grid"]:
            if name == "RF":
                model = RandomForestClassifier(n_estimators=config["rf_trees"], max_depth=value, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=config.get("threads", 2))
            elif name == "SVM":
                model = SVC(C=float(value or 1), kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
            else:
                model = LogisticRegression(C=float(value or 1), class_weight="balanced", max_iter=1000, random_state=seed)
            model.fit(x, y)
            score = f1_score(yv, model.predict(xv), labels=[0, 1], average="macro", zero_division=0)
            if best is None or score > best[0]:
                best = (score, model, value)
        return FittedModel(best[1], False, time.perf_counter() - start, dict(candidate=best[2], validation_macro_f1=float(best[0])), [])
    xt, yt, xvt = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long), torch.tensor(xv, dtype=torch.float32)
    gt, gvt = [tensor_graph(m) for m in graphs], [tensor_graph(m) for m in vg]
    unique, inverse = np.unique(groups, return_inverse=True)
    gi = torch.tensor(inverse, dtype=torch.long)
    counts = torch.bincount(gi).float()
    best_run = None
    history = []
    for lr in config["learning_rates"]:
        seed_everything(seed, config.get("threads", 2))
        model = RelationalNet(x.shape[1], config["hidden_dim"], len(graphs), config["dropout"], sage=name == "GraphSAGE")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
        best_score, best_state, patience, best_epoch = -1., None, 0, 0
        group_weights = torch.ones(len(unique)) / len(unique)
        for epoch in range(config["epochs"]):
            epoch_start = time.perf_counter()
            model.train()
            optimizer.zero_grad()
            loss_vector = nn.functional.cross_entropy(model(xt, gt), yt, reduction="none")
            if name in {"GroupDRO", "GroupReweight"}:
                risks = torch.zeros(len(unique)).scatter_add_(0, gi, loss_vector) / counts
                if name == "GroupDRO":
                    group_weights = group_weights * torch.exp(config["dro_eta"] * risks.detach())
                    group_weights /= group_weights.sum()
                    loss = (group_weights * risks).sum()
                else:
                    loss = risks.mean()
            else:
                # Training-only class balance weights.
                class_counts = np.bincount(y, minlength=2)
                cw = torch.tensor(len(y) / (2 * np.maximum(1, class_counts)), dtype=torch.float32)
                loss = (loss_vector * cw[yt]).mean()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                pred = model(xvt, gvt).argmax(1).numpy()
            score = f1_score(yv, pred, labels=[0, 1], average="macro", zero_division=0)
            history.append(dict(lr=lr, epoch=epoch + 1, loss=float(loss.detach()), validation_macro_f1=float(score), epoch_seconds=time.perf_counter() - epoch_start))
            if score > best_score + 1e-7:
                best_score, best_state, patience, best_epoch = score, copy.deepcopy(model.state_dict()), 0, epoch + 1
            else:
                patience += 1
            if patience >= config["patience"]:
                break
        model.load_state_dict(best_state)
        if best_run is None or best_score > best_run[0]:
            best_run = (best_score, model, lr, best_epoch)
    return FittedModel(best_run[1], True, time.perf_counter() - start,
                       dict(learning_rate=best_run[2], selected_epoch=best_run[3], validation_macro_f1=float(best_run[0]), parameters=sum(p.numel() for p in best_run[1].parameters())), history)
