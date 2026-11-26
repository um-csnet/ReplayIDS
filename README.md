# ReplayIDS

[![Python 3.12.4](https://img.shields.io/badge/Python-3.12.4-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Title

Code and reproducibility package for the paper:

> **Adaptive Intrusion Detection System using Transformer-Based Neural Networks
> and Continual Learning Approach with Adversarial Investigation**
>
> Azizi Ariffin, Afif Haris, Faiz Zaki, Hazim Hanif, Nor Badrul Anuar.
> Submitted to *PeerJ Computer Science*. Preprint:
> [arXiv:2608.04602](https://arxiv.org/abs/2608.04602)

Repository: <https://github.com/um-csnet/ReplayIDS>

## Description

Network intrusion detection systems (IDS) trained on a fixed snapshot of traffic
degrade after deployment because the threat distribution keeps shifting.
Fine-tuning on new attack data causes catastrophic forgetting, and retraining
from scratch on all historical data becomes infeasible.

ReplayIDS is the implementation behind the paper. It provides:

- an adaptive IDS that couples a **tabular transformer (TabTransformer)
  encoder** with a **benign-anchored class-balanced experience replay buffer**,
  in which benign traffic is replayed at every update;
- two continual-learning evaluation scenarios on CICIDS2017: the standard
  **class-incremental (CI)** scenario and the proposed **class-instance
  incremental (CII)** scenario, in which benign flows reappear in every
  experience alongside new attack classes;
- six continual-learning strategies plus a joint-training (oracle) upper bound;
- two **poisoning attacks on the replay buffer**: label flipping and a
  timing-feature (inter-arrival-time) backdoor;
- an independent cross-architecture benchmark (MLP, SGM-CNN, FT-Transformer);
- scripts that regenerate every numerical table and figure in the paper from
  released results or from a full rerun.

## Dataset Information

| Item | Details |
|---|---|
| Dataset | CICIDS2017 (Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset 2017) |
| Provider | Canadian Institute for Cybersecurity, University of New Brunswick |
| Source | <https://www.unb.ca/cic/datasets/ids-2017.html> |
| Download | `GeneratedLabelledFlows.zip` (CICFlowMeter-labelled flow CSVs) |
| Files used | `Monday-WorkingHours.pcap_ISCX.csv`, `Tuesday-WorkingHours.pcap_ISCX.csv`, `Wednesday-workingHours.pcap_ISCX.csv` |
| Records used | 1,138,612 labelled network flows |
| Features | 78 flow features (statistics on packet counts, sizes, flags, durations and inter-arrival times) plus 1 integer label |
| Classes | 8: Benign, DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, FTP-Patator, Heartbleed, SSH-Patator |
| Split | Seeded stratified 60/20/20 train/validation/test |
| Access | Publicly available from the provider; not redistributed in this repository |

Class distribution after preprocessing:

| ID | Class | Train | Validation | Test | Total |
|---:|---|---:|---:|---:|---:|
| 0 | Benign | 523,263 | 174,421 | 174,421 | 872,105 |
| 1 | DoS GoldenEye | 6,176 | 2,058 | 2,059 | 10,293 |
| 2 | DoS Hulk | 138,644 | 46,214 | 46,215 | 231,073 |
| 3 | DoS Slowhttptest | 3,299 | 1,100 | 1,100 | 5,499 |
| 4 | DoS slowloris | 3,477 | 1,160 | 1,159 | 5,796 |
| 5 | FTP-Patator | 4,763 | 1,587 | 1,588 | 7,938 |
| 6 | Heartbleed | 7 | 2 | 2 | 11 |
| 7 | SSH-Patator | 3,538 | 1,180 | 1,179 | 5,897 |
| | **Total** | 683,167 | 227,722 | 227,723 | 1,138,612 |

Thursday and Friday files are not used because their attack classes fall outside
this eight-class label space. The full data contract (schema, expected counts,
hashes) is in [docs/DATA.md](docs/DATA.md) and `configs/data/cicids2017.yaml`.

## Code Information

```text
configs/
  data/                  CICIDS2017 schema and feature contracts
  experiments/           primary, attack and cross-architecture manifests
  paper/                 machine-readable paper result contract
docs/
  DATA.md                dataset acquisition and validation
  PIPELINE.md            end-to-end reproduction workflow
  PAPER_MAP.md           paper table/figure -> artifact mapping
scripts/
  prepare_cicids2017.py  deterministic raw-data builder
  run_experiments.py     portable manifest runner
  local/                 optional multi-GPU orchestration examples
analysis/
  result_io.py           log -> structured metrics
  export_run_summaries.py rerun artifacts -> paper-builder inputs
  build_paper_results.py canonical table bundle generator
  plot_paper_results.py  deterministic SVG figure generator
  verify_eaai_mapping.py mapping completeness check
results/
  eaai-reported/         compact released results and raw ASR logs
benchmarks/
  cross_architecture/    independent cross-architecture benchmark
src/
  data/                  preprocessing and CI/CII experience builder
  strategies/            replay buffer, CL strategies, buffer attacks
  training/              training and evaluation loop
tests/                   unit and integration tests
main.py                  primary training entry point
```

Key modules:

| Module | Purpose |
|---|---|
| `src/data/preprocess.py` | Cleans and encodes CICIDS2017 flows |
| `src/data/ci_builder.py` | Builds the four experiences for the CI and CII scenarios |
| `src/strategies/replay.py` | ER-Stratified and benign-anchored ER-Balanced replay buffers |
| `src/strategies/cl_strategies.py` | Naive, EWC, LwF and iCaRL baselines |
| `src/strategies/attack.py` | Label-flipping and IAT-backdoor poisoning of the buffer |
| `src/training/train.py` | TabTransformer training, evaluation, forgetting and intransigence |

Raw data, generated arrays, checkpoints and full rerun directories are ignored
because they are large. Compact result CSVs and provenance artifacts are tracked.
Some paths keep the prefix `eaai` from an earlier version of the manuscript; they
refer to the same experiments reported in the paper.

## Requirements

- Linux recommended (CPU execution works but the full matrix is very slow)
- Python 3.12.4
- [`uv`](https://docs.astral.sh/uv/) package manager
- CUDA-capable NVIDIA GPU recommended (experiments ran on an NVIDIA RTX A5000, 24 GB)
- About 8 GB free disk space for raw and prepared CICIDS2017 data

Main Python dependencies (exact versions pinned in `uv.lock`):

- PyTorch 2.6+ (with torchvision and torchaudio)
- `tab-transformer-pytorch` (TabTransformer implementation)
- Avalanche (`avalanche-lib`) continual-learning library
- NumPy, pandas, scikit-learn, PyYAML, tqdm
- plotly, torchinfo, torchsummary
- pytest and ruff (testing and linting)
- Weights & Biases (optional; disabled in all official manifests)

## Usage Instructions

### 1. Install the exact environment

```bash
git clone https://github.com/um-csnet/ReplayIDS.git
cd ReplayIDS
uv sync --frozen
```

The committed `uv.lock` is the dependency source of truth.

### 2. Download CICIDS2017

Download `GeneratedLabelledFlows.zip` from the
[University of New Brunswick CICIDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html)
and extract the Monday, Tuesday and Wednesday CSV files listed under
[Dataset Information](#dataset-information) into one directory.

### 3. Rebuild and verify the dataset

```bash
uv run python scripts/prepare_cicids2017.py \
  --raw-dir /path/to/GeneratedLabelledFlows

uv run python scripts/prepare_cicids2017.py --verify-only
```

Expected shapes: train `(683167, 79)`, validation `(227722, 79)`,
test `(227723, 79)`.

### 4. Inspect the experiment matrix

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

### 5. Run the primary experiments

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

Each run writes `results/runs/<run-id>/run.log` and `run.json`. `run.json`
contains the resolved configuration, command, Git commit, host, timestamps,
return code and parsed per-experience/final metrics.

Turn the run artifacts into the paper tables and figures:

```bash
uv run python analysis/export_run_summaries.py
uv run python analysis/build_paper_results.py \
  --primary results/rerun-primary \
  --output results/rerun-generated
uv run python analysis/plot_paper_results.py \
  --tables-dir results/rerun-generated \
  --output-dir results/rerun-generated/figures
```

This keeps a rerun separate from the released reference results.

### 6. Run the replay-buffer attacks

```bash
uv run python scripts/run_experiments.py \
  --manifest configs/experiments/eaai-attacks.yaml
```

### 7. Reproduce the cross-architecture benchmark

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

### 8. Regenerate the paper results from the released data

No GPU or dataset is needed for this step:

```bash
uv run python analysis/build_paper_results.py
uv run python analysis/plot_paper_results.py
uv run python analysis/verify_eaai_mapping.py
```

Outputs are written to `results/generated/` (tables 6-14 as CSV, figures 4 and
7-10 as SVG, plus `RESULTS_REFERENCE.md` and `verification.json`).

### Direct CLI examples

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

# Backdoor configuration
WANDB_MODE=disabled uv run python main.py --er --balanced True --mem 10 \
  --scenario 1 --mp --poison_rate 10 --seed 42
```

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

`--er`, `--ewc`, `--lwf` and `--icarl` are mutually exclusive. `--lf` and `--mp`
are also mutually exclusive; the official attack manifests combine either attack
with ER-Balanced.

### Testing

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The test suite uses synthetic arrays and does not require CICIDS2017.

## Methodology

1. **Data processing.** The three CICIDS2017 CSVs are merged, the eight classes
   are mapped to integer labels, infinite and missing numeric values are
   replaced, categorical features keep their original values (with a per-column
   vocabulary passed to the embedding layers), and continuous features are scaled
   with statistics from the training partition only. A seeded stratified
   60/20/20 split produces the train, validation and test sets.
2. **Continual-learning scenarios.** The data is partitioned into four
   experiences of two new classes each. In **CI**, benign traffic appears only
   in the first experience. In **CII**, benign traffic is present in every
   experience alongside the new attack classes.
3. **Model.** A TabTransformer embeds categorical features, contextualises them
   with 6 transformer blocks (10 attention heads), concatenates them with the
   continuous features and classifies with an MLP head that expands as new
   classes arrive.
4. **Strategies.** Naive fine-tuning, EWC, LwF, iCaRL, ER-Stratified and
   benign-anchored ER-Balanced are compared, with per-class replay budgets of
   1%, 5% and 10%, against a joint-training oracle.
5. **Training.** AdamW (learning rate 5e-4, weight decay 1e-5), batch size 256,
   cross-entropy loss, ReduceLROnPlateau scheduling and early stopping; random
   seed 42.
6. **Metrics.** Accuracy and macro-F1 on all seen classes, plus forgetting and
   intransigence computed from the accuracy matrix across experiences.
7. **Adversarial investigation.** The replay buffer is poisoned by (a) label
   flipping of stored exemplars at buffer budgets of 1%, 5% and 10%, and (b) a
   backdoor that perturbs inter-arrival-time features of benign-labelled
   exemplars at poison rates of 1%, 5% and 10%, measured by attack success rate.

See [docs/PIPELINE.md](docs/PIPELINE.md) for the end-to-end workflow and
[docs/PAPER_MAP.md](docs/PAPER_MAP.md) for the mapping from every paper table and
figure to the code and result files.

### Reproducibility notes

- Seed 42 is the paper-replication seed. CUDA kernels, drivers and GPU models can
  produce small numerical differences; compare within the tolerances in
  `configs/paper/eaai_expected.yaml`.
- The released runs record an effective 8-epoch cap in `configs/config.yaml`;
  this historical configuration is retained as run.
- The released 1,138,612-row data contract retains duplicate rows and replaces
  invalid numeric values with zero; this policy is recorded in
  `configs/data/cicids2017.yaml`.
- The Heartbleed class has only 11 samples, so its per-class metric is unstable.
- Three historical backdoor ASR estimates exceed 100% because the original
  evaluation reused mutated test objects; they are retained and flagged in
  `results/eaai-reported/asr-logs/`.

## Citations

If you use this code, please cite the paper:

```bibtex
@misc{ariffin2026replayids,
  author       = {Ariffin, Azizi and Haris, Afif and Zaki, Faiz and Hanif, Hazim and Anuar, Nor Badrul},
  title        = {Adaptive Intrusion Detection System using Transformer-Based Neural Networks
                  and Continual Learning Approach with Adversarial Investigation},
  howpublished = {arXiv preprint arXiv:2608.04602},
  year         = {2026},
  doi          = {10.48550/arXiv.2608.04602}
}
```

If you use the CICIDS2017 dataset, please also cite its authors:

```bibtex
@inproceedings{sharafaldin2018cicids,
  author    = {Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A.},
  title     = {Toward Generating a New Intrusion Detection Dataset and Intrusion
               Traffic Characterization},
  booktitle = {Proceedings of the 4th International Conference on Information
               Systems Security and Privacy (ICISSP)},
  pages     = {108--116},
  year      = {2018}
}
```

## License & Contribution Guidelines

This code is released under the MIT License. See [LICENSE](LICENSE).
The CICIDS2017 dataset is subject to the terms of its provider, the Canadian
Institute for Cybersecurity.

Contributions are welcome:

1. Open an issue describing the bug or proposed change.
2. Fork the repository and create a feature branch.
3. Run `uv run pytest` and `uv run ruff check .` before submitting.
4. Open a pull request that references the issue.

Do not overwrite the released results in `results/eaai-reported/`; store new
runs in a separate directory.
