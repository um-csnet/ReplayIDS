# IDS Continual-Learning Benchmark (MLP + CNN)

Baseline benchmark for the adaptive-IDS project: how do **MLP** and **SGM-CNN**
backbones behave under the paper's continual-learning protocols, across four CL
strategies? Complements the main paper (which uses a TabTransformer + Experience
Replay). Strategic goal per the handover: establish foundational sequential
pipelines before scaling to Federated Prompt-Based Continual Learning.

## Design

- **Dataset:** CIC-IDS-2017 (Tuesday + Wednesday CSVs in the repository's
  `raw/CICIDS2017/` directory, or supplied with `--raw-dir`).
- **Label space (canonical, = paper Table 3 / ReplayIDS):**
  `0 Benign · 1 DoS GoldenEye · 2 DoS Hulk · 3 DoS Slowhttptest · 4 DoS slowloris · 5 FTP-Patator · 6 Heartbleed · 7 SSH-Patator`
- **Protocols (paper Table 3):**
  - **CI** (class-incremental): E₀{0,1} E₁{2,3} E₂{4,5} E₃{6,7} — Benign only in E₀.
  - **CII** (class-instance incremental): E₀{0,1} E₁{0,2,3} E₂{0,4,5} E₃{0,6,7} — Benign anchored every experience with new instances.
- **Models:** MLP (256-128, ReLU) and SGM-CNN (78→9×9 reshape, 2×Conv2d+MaxPool+FC). Fixed 8-way head.
- **Strategies:** Naive (fine-tune), Replay (class-balanced buffer, 10%), EWC (separate diagonal Fisher, λ=1000), LwF (frozen teacher, KD on seen classes, α=0.5, T=2).
- **Preprocessing:** clean inf/NaN, StandardScaler fit on the union of training data. Class-weighted CE (capped ×20) keeps rare attacks learnable.
- **Training:** AdamW, lr 1e-3, wd 1e-4, batch 256, 5 epochs/experience, grad-clip 1.0, seed 42.

## Metrics (matching paper Table 5/6)

Per (protocol × model × strategy): overall accuracy, macro-F1 (combined test after
final experience), **average forgetting** (Acc, F1), **intransigence** (Oracle − overall).
A Joint/Oracle model per architecture gives the upper bound.

## Results (capped 50k/class train; test full)

**CI (class-incremental)** — final, all 3 backbones (`results/summary_ci.csv`):

| Model | Naive | Replay | EWC | LwF |
|---|---|---|---|---|
| MLP | 0.080 | **0.964** | 0.147 | 0.157 |
| CNN | 0.006 | **0.966** | 0.260 | 0.007 |
| FTTransformer | 0.008 | **0.948** | 0.049 | 0.018 |

(overall accuracy after E3.) Story: **Replay dominates (~0.95–0.97); Naive/EWC/LwF all collapse in CI** across every backbone — the expected class-incremental result. FTTransformer **LwF CI = 0.018** is the first real transformer LwF number.

**CII (class-instance incremental, benign anchored)** — final (`results/summary_cii.csv`):

| Model | Naive | Replay | EWC | LwF | iCaRL |
|---|---|---|---|---|---|
| MLP | 0.777 | **0.967** | 0.920 | 0.938 | 0.904 |
| CNN | 0.769 | **0.966** | 0.835 | 0.879 | 0.905 |
| FTTransformer | 0.758 | **0.979** | 0.923 | 0.837 | 0.927 |

Benign anchoring lifts everything: **LwF recovers to 0.84–0.94** (vs its CI collapse), EWC to
0.84–0.92, even Naive to ~0.77.

**iCaRL** (CI+CII × 3 backbones, `results/icarl/`) — added as a 5th strategy (herding
exemplars + BCE distillation + NME classification). **2nd-best in CI (0.71–0.82)**, well ahead
of EWC/LwF there; 0.90–0.93 in CII. All 5 strategies × 3 backbones × 2 protocols complete;
32 confusion matrices in `results/cm_*.png` + `results/icarl/cm_*.png`.

## Layout

```
src/
  data_prep.py   build CI + CII experience streams -> results/prepared_{ci,cii}.npz
  models.py      MLP, SGMCNN
  strategies.py  Naive / Replay / EWC / LwF as experience-stream trainers
  evaluate.py    combined-test metrics + confusion-matrix plots
  run.py         orchestrator (protocols × models × strategies + oracle)
results/
  summary_ci.csv, summary_cii.csv   headline tables
  metrics.json                      full results incl. forgetting matrices
  cm_<proto>_<model>_<strategy>.png  confusion matrices
  run.log
```

## Reproduce

```bash
uv sync --frozen
uv run python src/data_prep.py --raw-dir ../../raw/CICIDS2017
uv run python src/run.py --protocols ci cii --models mlp cnn ftt \
  --strategies naive replay ewc lwf icarl --epochs 5 --gpu 0
```

## Notes / caveats

- **Heartbleed = 11 samples total** (7 train / 4 test): its per-class numbers are
  noise; read macro-F1 and the confusion matrices with that in mind.
- **EWC/LwF underperform Replay in CI** — expected and literature-consistent:
  regularisation-based CL fails in class-incremental settings; replay works. This
  is the same stability argument the main paper makes. CII (benign anchoring) is
  where the reg methods and replay separate differently.
- `ewc_lambda` (λ=1000) is the one calibration knob; sweep {1e2,1e3,1e4} if CII
  forgetting stays high.
- Not the same as the paper's runs (different backbone: MLP/CNN vs TabTransformer),
  so numbers are comparable in *trend*, not identical in value.

## Code review notes (Codex gpt-5.5, 2026-07-08 — accepted as known simplifications)

Static review confirmed the metrics and data splits are correct (forgetting formula,
intransigence = oracle − overall, CI/CII partitioning, EWC diagonal Fisher, CII benign
anchoring, no scaler leakage). The following are deliberate simplifications, kept for
faithfulness to the reference repos and because they apply equally across strategies (so
the *relative* comparison holds):

- **Replay buffer is not globally class-balanced.** It appends a per-experience balanced
  slice each step rather than enforcing fixed per-class quotas, so under CII benign
  accumulates across experiences. (`strategies.py` Replay.)
- **LwF uses `(1-α)·CE + α·KD`** (α=0.5), matching the IoT-Continual-Learning reference
  (`(1-alpha)*ce + alpha*kd`). α is therefore a CE/KD trade-off, not a pure distillation
  weight. A standard alternative is `CE + λ·KD`.
- **Fixed 8-way head from E0** (no seen-class logit masking): CE over all 8 classes mildly
  suppresses not-yet-seen logits before their classes arrive. Equivalent effect across all
  strategies; a stricter setup would mask logits to seen classes during training.
- **Scaler fit on the union of all experiences' training data** (offline standardisation,
  as in the paper's `*_standardised.csv`), not a strict online-CL fit-as-you-go.
- Minor: `run.py --models` default is `mlp cnn`; pass `--models mlp cnn ftt` for all three.
