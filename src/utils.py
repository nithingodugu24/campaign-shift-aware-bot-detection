from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
import threading
import time
from contextlib import contextmanager
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import psutil
import torch


def seed_everything(seed: int, threads: int = 2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "matplotlib", "networkx", "psutil"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    return dict(python=sys.version, platform=platform.platform(), cpu=platform.processor(),
                logical_cpus=psutil.cpu_count(), ram_gb=psutil.virtual_memory().total / 2**30,
                accelerator="CPU", torch_threads=torch.get_num_threads(), packages=packages)


@contextmanager
def measured(rows, component, **metadata):
    """Sample process RSS every 10 ms, including native NumPy/PyTorch allocations."""
    proc = psutil.Process()
    baseline = proc.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample():
        while not stop.wait(.01):
            peak[0] = max(peak[0], proc.memory_info().rss)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        peak[0] = max(peak[0], proc.memory_info().rss)
        stop.set()
        worker.join()
        rows.append(dict(component=component, seconds=elapsed,
                         peak_rss_mb=peak[0] / 2**20,
                         incremental_peak_rss_mb=max(0, peak[0] - baseline) / 2**20,
                         memory_method="process RSS sampled every 10ms; absolute peak includes prior stages",
                         **metadata))
