# EAAI Paper-to-Artifact Map

The latest paper output is the EAAI LaTeX build dated 18 July 2026. This file is
the human-readable counterpart of `configs/paper/eaai_expected.yaml`.

## Tables

| EAAI item | Kind | Canonical source | Reproduction path |
|---|---|---|---|
| Table 1 | Literature synthesis | Paper bibliography | Not code-generated |
| Table 2 | Backdoor IAT features | `configs/data/features.yaml` | Data preparation validates and emits `iatfeaturelist.npy` |
| Table 3 | Class distribution | `configs/data/cicids2017.yaml` | `scripts/prepare_cicids2017.py` → `data_report.json` |
| Table 4 | CI/CII experiences | `src/data/ci_builder.py` | Scenario-builder tests and primary runs |
| Table 5 | Hyperparameters | `configs/config.yaml` and experiment manifests | Resolved run configuration in each `run.json` |
| Table 6 | CI results | Primary CI runs | `table06_ci.csv` |
| Table 7 | CII checkpoint average | Primary CII runs | `table07_cii_checkpoint_average.csv` |
| Table 8 | CII final checkpoint | Per-experience results | `table08_cii_final.csv` |
| Table 9 | Forgetting/intransigence | Derived primary metrics | `table09_forgetting_intransigence.csv` |
| Table 10 | Label flipping | `eaai-attacks.yaml` | `table10_label_flip.csv` |
| Table 11 | Backdoor aggregate | `eaai-attacks.yaml` | `table11_backdoor.csv` |
| Table 12 | Historical backdoor ASR | Bundled raw logs | `table12_backdoor_asr_historical.csv` |
| Table 13 | Cross-architecture CI | Independent benchmark | `table13_cross_architecture_ci.csv` |
| Table 14 | Cross-architecture CII | Independent benchmark | `table14_cross_architecture_cii.csv` |

Tables 6-9 use the TabTransformer pipeline. Tables 13-14 use the separate
MLP/SGM-CNN/FT-Transformer benchmark under `benchmarks/cross_architecture/` and
are trend-comparable rather than identical to the primary pipeline.

## Figures

| EAAI item | Source |
|---|---|
| Figures 1-3 | Conceptual architecture/replay diagrams; not numerical outputs |
| Figure 4 | CICIDS2017 CI/CII scenario distribution generated from Table 4 data |
| Figures 5-6 | Conceptual label-flip and backdoor pipelines |
| Figure 7 | Table 6 CI values |
| Figure 8 | Table 7 CII values |
| Figure 9 | Table 11 backdoor aggregate values |
| Figure 10 | Tables 13-14 cross-architecture values |

The five numerical figures are regenerated as deterministic SVG files by
`analysis/plot_paper_results.py`. The renderer uses only the versioned table and
dataset contracts and therefore does not depend on a plotting-library version.

## Table 5 configuration note

The EAAI LaTeX table states “up to 100” epochs with early stopping patience 10.
The released primary runs and their bundled values used the effective 8-epoch
cap recorded in `configs/config.yaml`. Both facts are recorded in the paper
contract. Exact regeneration of the released values uses 8; a new run following
the literal paper cap should override it to 100 and be labelled as a rerun.

## Preprocessing provenance note

The EAAI prose states that invalid flows were dropped and the dataset
deduplicated. The executable released-data contract instead retains duplicates
and replaces infinite/NaN numeric values with zero, producing 1,138,612 rows.
`configs/data/cicids2017.yaml` records that historical policy explicitly. The
repository preserves it for exact result provenance; corrected preprocessing is
a distinct experiment and must not overwrite the reported bundle.

## Historical ASR limitation

The four text logs under `results/eaai-reported/asr-logs/` are the raw evidence
used for EAAI Table 12. Three reported estimates exceed 100%. They are flagged in
the machine-readable paper contract. The excess comes from test-set trigger
carryover and small-sample baseline variance; see `docs/PIPELINE.md`.

## Verification

```bash
uv run python analysis/build_paper_results.py
uv run python analysis/plot_paper_results.py
uv run python analysis/verify_eaai_mapping.py
```

The first command generates the result bundle. The second fails when a mapped
source, manifest or EAAI table entry is missing.
