# ReplayIDS

[![Python 3.12.4](https://img.shields.io/badge/Python-3.12.4-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Self-reproducible implementation and EAAI artifact package for:

> **Adaptive Intrusion Detection System using Transformer-Based Neural Networks
> and Continual Learning Approach with Adversarial Investigation**

ReplayIDS evaluates a TabTransformer intrusion detector on CICIDS2017 under
class-incremental (CI) and class-instance incremental (CII) streams. It includes
six continual-learning methods, two replay-buffer attacks, a joint-training
oracle, and the independent cross-architecture benchmark reported in the paper.

Repository: <https://github.com/um-csnet/ReplayIDS>

## What this repository reproduces

- Naive sequential fine-tuning
- Elastic Weight Consolidation (EWC)
- Learning without Forgetting (LwF)
- iCaRL
- ER-Stratified
- ER-Balanced with benign anchoring
- Label flipping of replay-buffer exemplars
- Timing-feature backdoor poisoning and historical ASR analysis
- MLP, SGM-CNN and FT-Transformer cross-architecture benchmark
- EAAI Tables 2-14 and numerical Figures 4 and 7-10

See [the complete EAAI paper map](docs/PAPER_MAP.md).

## Repository layout

```text
configs/
  data/                  CICIDS2017 schema and feature contracts
  experiments/           primary, attack and cross-architecture manifests
  paper/                 machine-readable EAAI result contract
docs/
  DATA.md                dataset acquisition and validation
  PIPELINE.md            end-to-end reproduction workflow
  PAPER_MAP.md           EAAI table/figure -> artifact mapping
scripts/
  prepare_cicids2017.py  deterministic raw-data builder
  run_experiments.py     portable manifest runner
  local/                 optional UM multi-GPU orchestration examples
analysis/
  result_io.py           log -> structured metrics
  export_run_summaries.py rerun artifacts -> paper-builder inputs
  build_paper_results.py canonical table bundle generator
  plot_paper_results.py  deterministic SVG figure generator
  verify_eaai_mapping.py mapping completeness check
results/
  eaai-reported/         compact released results and raw ASR logs
benchmarks/
  cross_architecture/    independent Tables 13-14 benchmark
src/                     primary ReplayIDS implementation
tests/                   unit and integration tests
main.py                  primary training entry point
```

Raw data, generated arrays, checkpoints and full rerun directories are ignored
because they are large. Compact paper CSVs and provenance artifacts are tracked.

## Requirements

- Linux recommended
- Python 3.12.4
- [`uv`](https://docs.astral.sh/uv/)
- CUDA-capable NVIDIA GPU recommended for the full matrix
- Approximately 8 GB free space for raw and prepared CICIDS2017 data

CPU execution is supported but the full experiment matrix is impractically slow.
Weights & Biases is optional and disabled in all official manifests.

## 1. Install the exact environment

```bash
git clone https://github.com/um-csnet/ReplayIDS.git
cd ReplayIDS
uv sync --frozen
```

The committed `uv.lock` is the dependency source of truth.

## 2. Download CICIDS2017

Download `GeneratedLabelledFlows.zip` from the
[University of New Brunswick CICIDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html).

Extract these files into one directory:

```text
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
```

Thursday and Friday are not used because their attack classes are outside the
paper's eight-class label space. See [the full data contract](docs/DATA.md).

## 3. Rebuild and verify the dataset

```bash
uv run python scripts/prepare_cicids2017.py \
  --raw-dir /path/to/GeneratedLabelledFlows

uv run python scripts/prepare_cicids2017.py --verify-only
```

The builder performs all previously manual steps:

- validates the three required files and schema;
- maps the eight classes;
- checks expected row and class counts;
- replaces infinities/NaNs consistently;
- creates the seeded stratified 60/20/20 split;
- generates categorical and IAT feature indices by column name;
- copies the six required `.npy` files into `dataset/CICIDS2017/`;
- writes hashes and distributions to `data_report.json`.

Expected shapes:

| Split | Shape |
|---|---:|
| Train | `(683167, 79)` |
| Validation | `(227722, 79)` |
| Test | `(227723, 79)` |

## 4. Inspect the official experiment matrix

Dry-run commands before using GPU time:

```bash
uv run python scripts/run_experiments.py \
  --manifest configs/experiments/eaai-primary.yaml \
  --dry-run

uv run python scripts/run_experiments.py \
  --manifest configs/experiments/eaai-attacks.yaml \
  --dry-run
```

The primary manifest expands to:

- 4 non-replay strategies × 2 scenarios = 8 runs;
- 2 replay variants × 3 budgets × 2 scenarios = 12 runs;
- 2 oracle runs.

## 5. Run the primary EAAI experiments

```bash
uv run python scripts/run_experiments.py \
  --manifest configs/experiments/eaai-primary.yaml
```

Run only one configuration while checking a machine:

```bash
uv run python scripts/run_experiments.py \
  --manifest configs/experiments/eaai-primary.yaml \
  --only er_balanced_s1_b10
```

Each run produces:

```text
results/runs/<run-id>/
  run.log
  run.json
```

`run.json` contains the resolved configuration, command, Git commit, host,
timestamps, return code and parsed per-experience/final metrics.

After the primary matrix completes, turn those run artifacts into the same
three-input schema used by the paper builder:

```bash
uv run python analysis/export_run_summaries.py
uv run python analysis/build_paper_results.py \
  --primary results/rerun-primary \
  --output results/rerun-generated
uv run python analysis/plot_paper_results.py \
  --tables-dir results/rerun-generated \
  --output-dir results/rerun-generated/figures
```

This keeps a rerun separate from the immutable EAAI-reported reference bundle.

## 6. Run the replay-buffer attacks

```bash
uv run python scripts/run_experiments.py \
  --manifest configs/experiments/eaai-attacks.yaml
```

The label-flip experiment varies the per-class buffer budget over 1%, 5% and
10%, while flipping 100% of admitted exemplars. The backdoor experiment uses a
10% replay buffer and varies the percentage of benign samples carrying the IAT
trigger over 1%, 5% and 10%.

The raw historical logs behind EAAI Table 12 are versioned under
`results/eaai-reported/asr-logs/`. Three historical ASR estimates exceed 100%
because the original evaluation reused mutated test objects and because very
small injected sets magnified baseline variance. They are retained and flagged
for provenance. Corrected reruns must be stored separately and must not overwrite
the paper-reported artifacts.

## 7. Reproduce the cross-architecture benchmark

Tables 13-14 use an independent pipeline:

```bash
cd benchmarks/cross_architecture
uv sync --frozen
uv run python src/data_prep.py
uv run python src/run.py \
  --protocols ci cii \
  --models mlp cnn ftt \
  --strategies naive replay ewc lwf icarl \
  --epochs 5 --gpu 0
```

This benchmark caps majority-class training samples and treats all features as
continuous. It tests whether the CI/CII pattern generalises across backbones; its
values are not intended to equal the primary TabTransformer results.

## 8. Generate and verify the EAAI result bundle

```bash
uv run python analysis/build_paper_results.py
uv run python analysis/plot_paper_results.py
uv run python analysis/verify_eaai_mapping.py
```

Generated outputs are written under `results/generated/`:

```text
table06_ci.csv
table07_cii_checkpoint_average.csv
table08_cii_final.csv
table09_forgetting_intransigence.csv
table10_label_flip.csv
table11_backdoor.csv
table12_backdoor_asr_historical.csv
table13_cross_architecture_ci.csv
table14_cross_architecture_cii.csv
RESULTS_REFERENCE.md
verification.json
figures/figure04_distribution.svg
figures/figure07_results.svg
figures/figure08_results.svg
figures/figure09_backdoor.svg
figures/figure10_cross_architecture.svg
```

The generated directory is intentionally ignored. The released inputs and EAAI
contract are versioned, so anyone can regenerate and compare it.

## EAAI result map

| Paper output | Repository artifact |
|---|---|
| Table 2 | `configs/data/features.yaml` |
| Table 3 | `dataset/CICIDS2017/data_report.json` |
| Table 4 | `src/data/ci_builder.py` |
| Table 5 | `configs/config.yaml` + manifests |
| Table 6 | `table06_ci.csv` |
| Table 7 | `table07_cii_checkpoint_average.csv` |
| Table 8 | `table08_cii_final.csv` |
| Table 9 | `table09_forgetting_intransigence.csv` |
| Table 10 | `table10_label_flip.csv` |
| Table 11 | `table11_backdoor.csv` |
| Table 12 | `table12_backdoor_asr_historical.csv` + raw logs |
| Tables 13-14 | Cross-architecture summary CSVs |
| Figures 7-10 | Generated from Tables 6-7, 11 and 13-14 |

See [PAPER_MAP.md](docs/PAPER_MAP.md) for the complete mapping, including
conceptual figures and known limitations.

## Direct CLI examples

```bash
# Naive CI
WANDB_MODE=disabled uv run python main.py --scenario 1 --seed 42

# EWC / LwF / iCaRL
WANDB_MODE=disabled uv run python main.py --ewc --scenario 1 --seed 42
WANDB_MODE=disabled uv run python main.py --lwf --scenario 1 --seed 42
WANDB_MODE=disabled uv run python main.py --icarl --scenario 1 --seed 42

# ER-Stratified and ER-Balanced
WANDB_MODE=disabled uv run python main.py --er --mem 10 --scenario 1 --seed 42
WANDB_MODE=disabled uv run python main.py --er --balanced True --mem 10 --scenario 1 --seed 42

# Label flipping: 10% buffer, every stored exemplar flipped
WANDB_MODE=disabled uv run python main.py --er --balanced True --mem 10 \
  --scenario 1 --lf --poison_rate 100 --seed 42

# Historical backdoor configuration
WANDB_MODE=disabled uv run python main.py --er --balanced True --mem 10 \
  --scenario 1 --mp --poison_rate 10 --seed 42
```

### Primary CLI reference

| Argument | Default | Meaning |
|---|---:|---|
| `--er` | off | Use experience replay; without a strategy flag the run is Naive |
| `--ewc` | off | Use Elastic Weight Consolidation |
| `--ewc-lambda` | `1000` | EWC quadratic penalty weight |
| `--lwf` | off | Use Learning without Forgetting |
| `--lwf-alpha` | `0.5` | LwF distillation-loss weight |
| `--lwf-T` | `2.0` | LwF distillation temperature |
| `--icarl` | off | Use iCaRL |
| `--icarl-memory` | `2000` | Fixed total iCaRL exemplar budget |
| `--mem` | `10` | ER per-class memory percentage |
| `--balanced` | `False` | Use benign-anchored balanced ER (`True` or `False`) |
| `--scenario` | `1` | `1` for CI; `2` for CII |
| `--dataset` | `CICIDS2017` | Dataset directory/config key |
| `--lf` | off | Apply label flipping to the replay buffer |
| `--mp` | off | Apply the timing-feature backdoor to the replay buffer |
| `--poison_rate` | `0` | Percentage of admitted buffer samples poisoned |
| `--seed` | `42` | Python, NumPy and PyTorch random seed |
| `--learning_rate` | config value | Override `configs/config.yaml` learning rate |
| `--oracle` | off | Train jointly on all experiences |

`--er`, `--ewc`, `--lwf` and `--icarl` are mutually exclusive. `--lf` and
`--mp` are also mutually exclusive and the official attack manifests combine
either attack with ER-Balanced. Prefer the manifests for paper replication;
direct flags are mainly useful for a single diagnostic run.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The test suite uses synthetic arrays and does not require CICIDS2017.

## Reproducibility expectations

- Seed 42 is the paper-replication seed.
- CUDA kernels, drivers and GPU models can produce small numerical differences.
- Compare within the tolerances documented in the paper contract rather than
  expecting byte-identical checkpoints.
- Table 5 describes a 100-epoch training cap with early stopping. The released
  runs that produced the bundled values record an effective 8-epoch cap in
  `configs/config.yaml`; this distinction is retained rather than silently
  changing the historical run configuration.
- The paper text describes dropping invalid flows and deduplication. The
  released 1,138,612-row contract retains duplicates and replaces invalid
  numeric values with zero. That executable historical policy is recorded in
  `configs/data/cicids2017.yaml`; corrected preprocessing must be reported as a
  separate experiment.
- The CICIDS2017 Heartbleed class has only 11 total samples; its per-class metric
  is inherently unstable.
- W&B is an optional dashboard, never the sole result store.

## Citation

Please cite the accompanying paper when using ReplayIDS. A formal bibliographic
entry will be added after publication.

## Licence

MIT License. See [LICENSE](LICENSE).
