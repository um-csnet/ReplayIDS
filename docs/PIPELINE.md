# End-to-End EAAI Reproduction Pipeline

This pipeline reproduces the computational outputs in the EAAI LaTeX paper from
a fresh ReplayIDS clone. It is deliberately portable: private hosts, NAS paths,
GPU numbers and W&B credentials are not part of the scientific specification.

## Pipeline

```text
CICIDS2017 GeneratedLabelledFlows CSVs
  -> scripts/prepare_cicids2017.py
  -> dataset/CICIDS2017/*.npy + data_report.json
  -> scripts/run_experiments.py + configs/experiments/*.yaml
  -> results/runs/<run-id>/{run.log,run.json}
  -> analysis/export_run_summaries.py
  -> analysis/build_paper_results.py --primary <summary-directory>
  -> results/generated/table*.csv + RESULTS_REFERENCE.md
  -> analysis/plot_paper_results.py
  -> analysis/verify_eaai_mapping.py
```

## Commands

```bash
uv sync --frozen

uv run python scripts/prepare_cicids2017.py \
  --raw-dir /path/to/GeneratedLabelledFlows

uv run python scripts/prepare_cicids2017.py --verify-only

uv run python scripts/run_experiments.py \
  --manifest configs/experiments/eaai-primary.yaml \
  --dry-run

uv run python scripts/run_experiments.py \
  --manifest configs/experiments/eaai-primary.yaml

uv run python scripts/run_experiments.py \
  --manifest configs/experiments/eaai-attacks.yaml

uv run python analysis/export_run_summaries.py
uv run python analysis/build_paper_results.py \
  --primary results/rerun-primary \
  --output results/rerun-generated
uv run python analysis/plot_paper_results.py \
  --tables-dir results/rerun-generated \
  --output-dir results/rerun-generated/figures
uv run python analysis/verify_eaai_mapping.py
```

The full primary matrix is GPU-intensive. Use `--only <substring>` to reproduce
one run, for example `--only er_balanced_s1_b10`.

The no-argument result-builder command reconstructs the immutable published
reference bundle. To analyse a completed rerun, pass the directory emitted by
`export_run_summaries.py` via `--primary` and use a separate `--output` path.

## Reported and corrected attack results

`results/eaai-reported/` preserves the values and raw ASR logs behind the paper.
The historical ASR estimator has three values above 100% because a reused test
dataset retained triggers across experiences and because very small injected
sets amplify ordinary model variance. These values are kept for traceability,
not presented as mathematically valid literal rates.

Corrected attack evaluation must use isolated test copies and direct triggered
sample accounting. Corrected reruns belong under `results/corrected/` and must
not overwrite the EAAI-reported artifacts.
