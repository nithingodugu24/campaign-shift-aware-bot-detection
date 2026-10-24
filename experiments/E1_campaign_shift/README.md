# E1 Random versus collection shift

## Contents

Each study subfolder contains the saved evidence for this experiment. `public` uses the Cresci-2017 pilot, `collection` uses proxy collection holdouts, and `synthetic` is an explicitly generated validation fixture. Check each study's `status.csv` and source metadata for unsupported cases.

## Implementation

- [splits.py](../../src/splits.py)
- [run_experiments.py](../../src/run_experiments.py)

The experiments share trained detectors and audited partitions. Checkpoints and split manifests remain under `results/<study>/` so every experiment refers to the same fitted models and held-out IDs.

## Reproduction

From the project root, use `run_all.ps1` to reproduce the underlying studies. To refresh only this folder from existing results:

```powershell
.venv\Scripts\python.exe -m scripts.organize_experiments --experiment E1
```

Refreshing exports does not retrain models. Full setup commands are in [the project README](../../README.md).
