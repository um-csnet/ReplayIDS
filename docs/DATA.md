# CICIDS2017 Data Contract

ReplayIDS uses the CICFlowMeter-labelled Monday, Tuesday and Wednesday CSVs from
the CICIDS2017 `GeneratedLabelledFlows` download.

Source: <https://www.unb.ca/cic/datasets/ids-2017.html>

Required filenames:

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`

The precise class mapping, expected counts and split shapes are stored in
`configs/data/cicids2017.yaml`. Categorical and backdoor-IAT columns are defined
by name in `configs/data/features.yaml`; numeric indices are generated only after
the input header is validated.

For exact historical replication, duplicate rows are retained and infinite or
missing numeric values are replaced with zero. This is the policy that produces
the released 1,138,612-row contract. It differs from the EAAI prose saying that
invalid flows were dropped and the data deduplicated; a corrected preprocessing
study must use a separate output directory and report its resulting counts.

Run:

```bash
uv run python scripts/prepare_cicids2017.py \
  --raw-dir /path/to/GeneratedLabelledFlows
```

Expected merged data:

| Class | Rows |
|---|---:|
| Benign | 872,105 |
| DoS GoldenEye | 10,293 |
| DoS Hulk | 231,073 |
| DoS Slowhttptest | 5,499 |
| DoS slowloris | 5,796 |
| FTP-Patator | 7,938 |
| Heartbleed | 11 |
| SSH-Patator | 5,897 |

Expected split shapes are train `(683167, 79)`, validation `(227722, 79)` and
test `(227723, 79)`. The final column is the integer label.

The generated `dataset/CICIDS2017/data_report.json` records raw-file hashes,
feature order, class counts, split distributions and hashes for all `.npy`
artifacts. Raw and processed data remain ignored by Git.
