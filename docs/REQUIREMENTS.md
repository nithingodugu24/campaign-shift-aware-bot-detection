# Requirements and implementation mapping

The supplied detailed experimental guide and two manuscripts define project scope. Document instructions are treated as research requirements subordinate to the user's request. The third revision resolves ambiguous statistical claims; none of its proposed target scores is copied into a results table.

## Protocol names

| Meaning | Code and detailed guide | Manuscript experiment section |
|---|---|---|
| Random accounts | P0 | P0 |
| Campaign or proxy collection holdout | P1 | P1 |
| Repeated group holdout | P2 | described as multi-campaign holdout |
| Chronological | P3 | P2 |
| Cross-dataset | P4 | P3 |

## Experiment traceability

| Requirement | Implementation | Evidence |
|---|---|---|
| Dataset audit | `src/data_audit.py`, preparation scripts | `audit/`, input hashes, licenses, preparation exclusions |
| E1 split sensitivity | `src/splits.py`, `src/run_experiments.py` | manifests, E1 metric rows, protocol plot, P0-minus-P1 tables |
| E2 temporal contribution | A1/A2/A3/A4/A5/Proposed with common optimization | ablation tables, parameter counts, E2 plot |
| E3 reliability | `src/calibrate.py` | calibrator parameters, feasibility flags, E3 and group rows |
| E4 null and filtering | `src/build_graph.py` | raw/corrected/filtered edge files, graph statistics, E4 metrics |
| E5 selective prediction | predefined alpha/threshold grids | selective curves, matched-coverage descriptive comparisons |
| E6 robustness | `src/robustness.py`, frozen test evaluation | severity rows, low-activity/group-size strata, benign-sync fixture |
| E7 scalability | measured pipeline stages and increasing subset sizes | time, 10 ms sampled process RSS, environment and notes |
| E8 transfer | fixed source checkpoints and preprocessing | target predictions, compatibility mapping, all-source-ID exclusions |
| Uncertainty | group bootstrap, paired tests, Holm | per-run intervals and `paired_statistics.csv` |
| Reproducibility | configs, seed control, checkpoints, tests | README, exact environment, manifests and validation log |

## Declared implementation choices and deviations

The manuscripts leave layer widths, optimization details, null permutation count, thresholds and exact calibration implementation open. They are fixed in YAML before final evaluation. The temporal kernel is a fixed mixture of exponentials, not a Hawkes model or an end-to-end learned point process. Corrected weights are standardized excess synchrony, not p-values. Popular objects are sampled to a fixed event cap. This approximation is disclosed and has a training-only sensitivity diagnostic; it is not a universal correction for external synchronization.

The null permutes timestamps among event records within the same object and UTC day. It preserves the object's observed time distribution and per-account counts inside that object/day block. It does not prove benign coordination has been removed. Near-zero null variance removes an edge; a support rule and mutual top-k bound prevent unbounded weights/degrees. Graph node summaries are log degree, log weighted strength and clustering; synchronized-event fraction is not added as a separate feature.

The full-neighborhood GraphSAGE baseline implements a mean aggregate with separate self/neighborhood linear transforms and L2 normalization. The proposed two-layer relational model uses weighted-mean coordination and an optional separate follow relation. This is an explicitly specified local implementation, not an official BotRGCN reproduction. Baselines that need unavailable multimodal inputs, verified temporal interactions or unimplemented official pipelines are reported as not reproduced.

No real input has sufficiently documented historical profile availability for P3. Public external targets have profile snapshots only, so RF/SVM/LR/profile MLP transfer is supported; graph/temporal transfer is not. Verified campaign-level validation remains outstanding. Collection proxies and synthetic groups are explicitly distinguished in every result file.

The actual public subsets and all measured negative/mixed outcomes are retained. There is no claim that the proposed method must win, attain 85% selective accuracy, or guarantee risk under arbitrary campaign shift.
