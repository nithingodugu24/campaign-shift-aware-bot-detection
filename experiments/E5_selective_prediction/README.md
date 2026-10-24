# E5 Selective prediction and abstention

## Contents

Each study subfolder contains the saved evidence for this experiment. `public` uses the Cresci-2017 pilot, `collection` uses proxy collection holdouts, and `synthetic` is an explicitly generated validation fixture. Check each study's `status.csv` and source metadata for unsupported cases.

## Implementation

- [calibrate.py](../../src/calibrate.py)
- [evaluate.py](../../src/evaluate.py)

The experiments share trained detectors and audited partitions. Checkpoints and split manifests remain under `results/<study>/` so every experiment refers to the same fitted models and held-out IDs.

## Reproduction

From the project root, use `run_all.ps1` to reproduce the underlying studies. To refresh only this folder from existing results:

```powershell
.venv\Scripts\python.exe -m scripts.organize_experiments --experiment E5
```

Refreshing exports does not retrain models. Full setup commands are in [the project README](../../README.md).

`selective_curves.csv` contains the E5 measurements. The nearest-coverage table is descriptive and does not select deployment thresholds from test labels.
