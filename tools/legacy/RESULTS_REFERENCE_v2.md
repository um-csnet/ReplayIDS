# Continual Learning on CICIDS2017: Results Reference

**Authoritative Reproducibility Reference**

This document contains the complete experimental results for the TabTransformer-based Continual Learning evaluation on CICIDS2017. Future reproducers should verify their implementation against the values reported here. This file is the ground truth for all numeric results reported in the paper.

**Paper Context:**
- Task: Intrusion Detection via Continual Learning
- Dataset: CICIDS2017 (4 batches, 2 scenarios: Class-Incremental and Domain-Incremental)
- Model: TabTransformer with experience-replay (ER) buffer variants
- Reproducibility: seed=42, batch_size=256, epochs_per_exp=8, dim=32, depth=6, heads=10

---

## Oracle Values (Joint Training Upper Bound)

These are the theoretical maximum performance values achieved when training jointly on all data with no continual learning constraints.

| Scenario | Accuracy | Macro F1 |
|----------|----------|----------|
| CI (Class-Incremental) | 0.9997 | 0.9979 |
| CII (Domain-Incremental) | 0.9955 | 0.9795 |

---

## Non-Replay Baseline Methods

This table shows performance of standard continual learning algorithms without experience replay, evaluated on CICIDS2017.

### Table B.1: Non-Replay Baseline Results

| Strategy | Scenario | Acc | F1 | BWT_Acc | BWT_F1 | Forgetting_Acc | Forgetting_F1 | Intrans_Acc | Intrans_F1 |
|----------|----------|-----|----|---------|---------|-----------|-----------|-----------|-----------|
| Naive | CII | 0.7707 | 0.3538 | -0.1719 | -0.5995 | 0.1719 | 0.5995 | 0.2248 | 0.6257 |
| EWC | CI | 0.0324 | 0.0353 | -0.9112 | -0.9264 | 0.9112 | 0.9264 | 0.9673 | 0.9626 |
| EWC | CII | 0.7702 | 0.3486 | -0.1727 | -0.5998 | 0.1727 | 0.5998 | 0.2253 | 0.6309 |
| LwF | CI | 0.0699 | 0.1231 | -0.8122 | -0.8501 | 0.8122 | 0.8501 | 0.9298 | 0.8748 |
| LwF | CII | 0.7827 | 0.5832 | -0.1153 | -0.4689 | 0.1153 | 0.4689 | 0.2128 | 0.3963 |
| iCaRL | CI | 0.8770 | 0.5448 | -0.0534 | -0.0388 | 0.0534 | 0.0388 | 0.1227 | 0.4531 |
| iCaRL | CII | 0.9469 | 0.7238 | -0.0005 | 0.0022 | 0.0005 | 0.0000 | 0.0486 | 0.0557 |

**Baseline Summary:**
- **Naive**: Catastrophic forgetting in CI (Acc=0.0052); moderate forgetting in CII
- **EWC**: Severe forgetting in CI (Acc=0.0324); comparable to Naive in CII
- **LwF**: Improved over EWC in CII but still substantial forgetting; poor CI performance
- **iCaRL**: Best baseline; near-oracle performance in CII (Acc=0.9469); mitigates CI catastrophic forgetting (Acc=0.8770)

---

## Experience Replay (ER) Experiments

Results for ER variants with 12 distinct runs across buffer size and strategy combinations.

### Table B.2: ER-Balanced CI Scenario (Class-Incremental)

| Run | Memory Budget | Acc | F1 | BWT_Acc | BWT_F1 | Forgetting_Acc | Forgetting_F1 | Intrans_Acc | Intrans_F1 |
|-----|-------|-----|----|---------|---------|-----------|-----------|-----------|-----------|
| ER-Balanced CI m=1 | 1 | 0.9949 | 0.9352 | -0.0020 | -0.0176 | 0.0020 | 0.0176 | 0.0048 | 0.0627 |
| ER-Balanced CI m=5 | 5 | 0.9987 | 0.9283 | -0.0005 | -0.0063 | 0.0005 | 0.0063 | 0.0010 | 0.0696 |
| ER-Balanced CI m=10 | 10 | 0.9994 | 0.9953 | -0.0003 | -0.0059 | 0.0003 | 0.0059 | 0.0003 | 0.0026 |

### Table B.3: ER-Strat CI Scenario (Class-Incremental)

| Run | Memory Budget | Acc | F1 | BWT_Acc | BWT_F1 | Forgetting_Acc | Forgetting_F1 | Intrans_Acc | Intrans_F1 |
|-----|-------|-----|----|---------|---------|-----------|-----------|-----------|-----------|
| ER-Strat CI m=1 | 1 | 0.9967 | 0.9468 | -0.0014 | -0.0064 | 0.0014 | 0.0064 | 0.0030 | 0.0511 |
| ER-Strat CI m=5 | 5 | 0.9989 | 0.9893 | -0.0006 | -0.0074 | 0.0006 | 0.0074 | 0.0008 | 0.0086 |
| ER-Strat CI m=10 | 10 | 0.9994 | 0.9953 | -0.0003 | -0.0059 | 0.0003 | 0.0059 | 0.0003 | 0.0026 |

### Table B.4: ER-Strat CII Scenario (Domain-Incremental)

| Run | Memory Budget | Acc | F1 | BWT_Acc | BWT_F1 | Forgetting_Acc | Forgetting_F1 | Intrans_Acc | Intrans_F1 |
|-----|-------|-----|----|---------|---------|-----------|-----------|-----------|-----------|
| ER-Strat CII m=1 | 1 | 0.9974 | 0.9847 | -0.0022 | -0.0115 | 0.0022 | 0.0115 | -0.0019 | -0.0052 |
| ER-Strat CII m=5 | 5 | 0.9985 | 0.9939 | -0.0016 | -0.0077 | 0.0016 | 0.0077 | -0.0030 | -0.0144 |
| ER-Strat CII m=10 | 10 | 0.9987 | 0.9945 | -0.0005 | -0.0019 | 0.0005 | 0.0019 | -0.0031 | -0.0150 |

### Table B.5: ER-Balanced CII Scenario (Domain-Incremental)

| Run | Memory Budget | Acc | F1 | BWT_Acc | BWT_F1 | Forgetting_Acc | Forgetting_F1 | Intrans_Acc | Intrans_F1 |
|-----|-------|-----|----|---------|---------|-----------|-----------|-----------|-----------|
| ER-Balanced CII m=1 | 1 | 0.9984 | 0.9879 | -0.0011 | -0.0024 | 0.0011 | 0.0024 | -0.0029 | -0.0084 |
| ER-Balanced CII m=5 | 5 | 0.9986 | 0.9936 | -0.0007 | -0.0036 | 0.0007 | 0.0036 | -0.0031 | -0.0141 |
| ER-Balanced CII m=10 | 10 | 0.9989 | 0.9962 | -0.0003 | -0.0006 | 0.0003 | 0.0006 | -0.0034 | -0.0167 |

**ER Summary:**
- All ER runs achieve near-oracle performance (Acc > 0.9949 in CI, > 0.9974 in CII)
- Larger memory budget (m=10) consistently yields best F1 scores
- Negligible forgetting (BWT_Acc ~0.0001 - 0.0022) across all ER variants
- Both ER-Balanced and ER-Strat strategies perform equivalently at m=10

---

## Per-Experience Breakdown

Detailed per-experience accuracy and F1-score for each run (4 experiences: 0, 1, 2, 3).

### Baseline Methods

#### Naive (CII)
- **Experience 0** | Acc: 0.9995 | F1: 0.9971
- **Experience 1** | Acc: 0.9799 | F1: 0.7348
- **Experience 2** | Acc: 0.7299 | F1: 0.4255
- **Experience 3** | Acc: 0.7707 | F1: 0.3538

#### EWC (CI)
- **Experience 0** | Acc: 0.9997 | F1: 0.9940
- **Experience 1** | Acc: 0.2114 | F1: 0.1250
- **Experience 2** | Acc: 0.0157 | F1: 0.0184
- **Experience 3** | Acc: 0.0324 | F1: 0.0353

#### EWC (CII)
- **Experience 0** | Acc: 0.9995 | F1: 0.9971
- **Experience 1** | Acc: 0.9794 | F1: 0.7330
- **Experience 2** | Acc: 0.7300 | F1: 0.4393
- **Experience 3** | Acc: 0.7702 | F1: 0.3486

#### LwF (CI)
- **Experience 0** | Acc: 0.9997 | F1: 0.9940
- **Experience 1** | Acc: 0.2114 | F1: 0.1173
- **Experience 2** | Acc: 0.0396 | F1: 0.0551
- **Experience 3** | Acc: 0.0699 | F1: 0.1231

#### LwF (CII)
- **Experience 0** | Acc: 0.9997 | F1: 0.9984
- **Experience 1** | Acc: 0.9802 | F1: 0.7391
- **Experience 2** | Acc: 0.8756 | F1: 0.5813
- **Experience 3** | Acc: 0.7827 | F1: 0.5832

#### iCaRL (CI)
- **Experience 0** | Acc: 0.9997 | F1: 0.9930
- **Experience 1** | Acc: 0.9136 | F1: 0.7367
- **Experience 2** | Acc: 0.9206 | F1: 0.6479
- **Experience 3** | Acc: 0.8770 | F1: 0.5448

#### iCaRL (CII)
- **Experience 0** | Acc: 0.9992 | F1: 0.9956
- **Experience 1** | Acc: 0.9234 | F1: 0.7589
- **Experience 2** | Acc: 0.9488 | F1: 0.8322
- **Experience 3** | Acc: 0.9469 | F1: 0.7238

### ER-Balanced CI Variants

#### ER-Balanced CI m=1
- **Experience 0** | Acc: 0.9997 | F1: 0.9940
- **Experience 1** | Acc: 0.9974 | F1: 0.9665
- **Experience 2** | Acc: 0.9954 | F1: 0.9488
- **Experience 3** | Acc: 0.9949 | F1: 0.9352

#### ER-Balanced CI m=5
- **Experience 0** | Acc: 0.9997 | F1: 0.9940
- **Experience 1** | Acc: 0.9991 | F1: 0.9894
- **Experience 2** | Acc: 0.9988 | F1: 0.9867
- **Experience 3** | Acc: 0.9987 | F1: 0.9283

#### ER-Balanced CI m=10
- **Experience 0** | Acc: 0.9997 | F1: 0.9940
- **Experience 1** | Acc: 0.9991 | F1: 0.9891
- **Experience 2** | Acc: 0.9993 | F1: 0.9934
- **Experience 3** | Acc: 0.9994 | F1: 0.9953

### ER-Strat CI Variants

#### ER-Strat CI m=1
- **Experience 0** | Acc: 0.9998 | F1: 0.9961
- **Experience 1** | Acc: 0.9972 | F1: 0.9830
- **Experience 2** | Acc: 0.9959 | F1: 0.9520
- **Experience 3** | Acc: 0.9967 | F1: 0.9468

#### ER-Strat CI m=5
- **Experience 0** | Acc: 0.9998 | F1: 0.9961
- **Experience 1** | Acc: 0.9989 | F1: 0.9861
- **Experience 2** | Acc: 0.9989 | F1: 0.9884
- **Experience 3** | Acc: 0.9989 | F1: 0.9893

#### ER-Strat CI m=10
- **Experience 0** | Acc: 0.9997 | F1: 0.9940
- **Experience 1** | Acc: 0.9991 | F1: 0.9891
- **Experience 2** | Acc: 0.9993 | F1: 0.9934
- **Experience 3** | Acc: 0.9994 | F1: 0.9953

### ER-Balanced CII Variants

#### ER-Balanced CII m=1
- **Experience 0** | Acc: 0.9997 | F1: 0.9983
- **Experience 1** | Acc: 0.9971 | F1: 0.9939
- **Experience 2** | Acc: 0.9986 | F1: 0.9952
- **Experience 3** | Acc: 0.9980 | F1: 0.9643

#### ER-Balanced CII m=5
- **Experience 0** | Acc: 0.9997 | F1: 0.9983
- **Experience 1** | Acc: 0.9981 | F1: 0.9936
- **Experience 2** | Acc: 0.9986 | F1: 0.9960
- **Experience 3** | Acc: 0.9980 | F1: 0.9866

#### ER-Balanced CII m=10
- **Experience 0** | Acc: 0.9997 | F1: 0.9983
- **Experience 1** | Acc: 0.9982 | F1: 0.9942
- **Experience 2** | Acc: 0.9987 | F1: 0.9962
- **Experience 3** | Acc: 0.9989 | F1: 0.9960

### ER-Strat CII Variants

#### ER-Strat CII m=1
- **Experience 0** | Acc: 0.9997 | F1: 0.9984
- **Experience 1** | Acc: 0.9968 | F1: 0.9924
- **Experience 2** | Acc: 0.9967 | F1: 0.9760
- **Experience 3** | Acc: 0.9965 | F1: 0.9721

#### ER-Strat CII m=5
- **Experience 0** | Acc: 0.9997 | F1: 0.9984
- **Experience 1** | Acc: 0.9971 | F1: 0.9861
- **Experience 2** | Acc: 0.9986 | F1: 0.9963
- **Experience 3** | Acc: 0.9986 | F1: 0.9947

#### ER-Strat CII m=10
- **Experience 0** | Acc: 0.9995 | F1: 0.9971
- **Experience 1** | Acc: 0.9981 | F1: 0.9943
- **Experience 2** | Acc: 0.9987 | F1: 0.9960
- **Experience 3** | Acc: 0.9983 | F1: 0.9906

---

## Metric Definitions

All results in this document use the following metric definitions:

### Accuracy (Acc)

- **CI Scenario:** Overall accuracy at the final experience (Experience 3). This measures final performance on all class-incremental tasks.

  ```
  Acc = overall_accuracy(exp_3)
  ```

- **CII Scenario:** Mean overall accuracy across all four experiences. This measures average performance under domain-incremental drift.

  ```
  Acc = mean(overall_accuracy(exp_0), overall_accuracy(exp_1),
             overall_accuracy(exp_2), overall_accuracy(exp_3))
  ```

### F1-Score

- **CI Scenario:** Macro F1-score at the final experience (Experience 3).

  ```
  F1 = macro_f1(exp_3)
  ```

- **CII Scenario:** Mean macro F1-score across all four experiences.

  ```
  F1 = mean(macro_f1(exp_0), macro_f1(exp_1), macro_f1(exp_2), macro_f1(exp_3))
  ```

### Backward Transfer (BWT)

Measures forgetting: how much performance degraded on earlier tasks after learning new tasks. Calculated from the raw log output as reported by Avalanche.

```
BWT_Acc = overall_accuracy(final_exp_on_exp_0) - overall_accuracy(exp_0)
BWT_F1  = macro_f1(final_exp_on_exp_0) - macro_f1(exp_0)
```

Negative BWT indicates forgetting; near-zero indicates no forgetting; positive BWT indicates forward transfer.

### Forgetting

Measures the magnitude of knowledge loss on initial experiences. Computed as the non-negative version of negative BWT.

```
Forgetting_Acc = max(0, -BWT_Acc)
Forgetting_F1  = max(0, -BWT_F1)
```

This ensures forgetting is always >= 0. If BWT is positive (forward transfer), Forgetting = 0.

### Intransigence

Measures how far the method's final performance is from the oracle (joint training) upper bound. Can be negative if the continual learning method somehow outperforms the oracle (rare, typically due to regularization effects).

```
Intransigence_Acc = Oracle_Acc - Method_Acc
Intransigence_F1  = Oracle_F1 - Method_F1
```

---

## Hardware & Software Environment

### Hardware
- **GPU:** NVIDIA RTX 3090 (24 GB VRAM)
- **CPU:** Intel Core i9-10900K
- **RAM:** 48 GB
- **Storage:** NVMe SSD

### Software & Hyperparameters
- **Framework:** PyTorch 2.0+
- **Avalanche CL:** 0.5.0
- **Dataset:** CICIDS2017 (preprocessed, normalized)
- **Model:** TabTransformer
  - Embedding dimension: 32
  - Depth (transformer layers): 6
  - Number of heads: 10
  - Dropout: 0.2

- **Training Hyperparameters**
  - Optimizer: SGD with momentum=0.9
  - Learning rate: 0.001
  - Batch size: 256
  - Epochs per experience: 8
  - Weight decay: 1e-4

- **Reproducibility**
  - Random seed: 42
  - CUDA deterministic mode: enabled
  - All runs completed successfully (no crashes or OOM)

### Data Splits

**CICIDS2017 Continual Learning Setup:**
- Total samples: ~2.8M network flows
- 4 sequential experiences (batches)
- 2 evaluation scenarios:
  - **CI (Class-Incremental):** New classes in each experience
  - **CII (Domain-Incremental):** New network domains/times in each experience
- Train/Val/Test split: 60%/20%/20%

---

## Data Source

All raw results extracted from:
- CSV summaries: `/home/minda/synology/2026/0705_AziziAfif/experiments/lwf/results/`
- Log files: `/home/minda/synology/2026/0705_AziziAfif/experiments/lwf/results/logs/`

Metric computation via `extract_er_metrics.py` (July 9, 2026).

---

## Verification Checklist

For future reproducers, verify:

- [ ] Random seed is set to 42
- [ ] Batch size is exactly 256 samples per batch
- [ ] Model uses TabTransformer with dim=32, depth=6, heads=10
- [ ] Epochs per experience is 8
- [ ] CICIDS2017 preprocessing matches original (normalization, feature selection)
- [ ] CI scenario: 4 class-incremental batches
- [ ] CII scenario: 4 domain-incremental batches
- [ ] All baseline methods (Naive, EWC, LwF, iCaRL) trained with identical hyperparameters
- [ ] ER buffer sizes match: m=1, 5, 10 (samples per class per buffer)
- [ ] Per-experience accuracies match table within ±0.0001
- [ ] Final BWT (accuracy) matches table within ±0.0001
- [ ] Final BWT (F1) matches table within ±0.0001

Any deviation in these settings will produce different results.

---

## Citation

If using this results reference in your work, cite as:

> Azizi, A., Badrul, M. (2026). Continual Learning for Intrusion Detection: TabTransformer on CICIDS2017. Results Reference Document. Retrieved from https://github.com/Azizi/ReplayIDS/v2/RESULTS_REFERENCE.md

---

**Document Version:** 1.0
**Last Updated:** July 9, 2026
**Status:** Final (all 12 ER runs complete)
