import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_paper_contract_covers_eaai_tables_3_to_14():
    contract = yaml.safe_load(
        (ROOT / "configs/paper/eaai_expected.yaml").read_text(encoding="utf-8")
    )
    assert set(contract["tables"]) == {f"table_{number:02d}" for number in range(3, 15)}


def test_feature_contract_matches_released_indices():
    header = ROOT / "preprocess_csv/CICIDS2017/CICIDS2017_standardised.csv"
    # CI does not ship the large CSV. Use its canonical header when absent.
    if header.exists():
        columns = header.open(encoding="utf-8").readline().strip().split(",")[:-1]
    else:
        columns = [
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Fwd Packet Length Max",
            "Fwd Packet Length Min",
            "Fwd Packet Length Mean",
            "Fwd Packet Length Std",
            "Bwd Packet Length Max",
            "Bwd Packet Length Min",
            "Bwd Packet Length Mean",
            "Bwd Packet Length Std",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd IAT Total",
            "Fwd IAT Mean",
            "Fwd IAT Std",
            "Fwd IAT Max",
            "Fwd IAT Min",
            "Bwd IAT Total",
            "Bwd IAT Mean",
            "Bwd IAT Std",
            "Bwd IAT Max",
            "Bwd IAT Min",
            "Fwd PSH Flags",
            "Bwd PSH Flags",
            "Fwd URG Flags",
            "Bwd URG Flags",
            "Fwd Header Length",
            "Bwd Header Length",
            "Fwd Packets/s",
            "Bwd Packets/s",
            "Min Packet Length",
            "Max Packet Length",
            "Packet Length Mean",
            "Packet Length Std",
            "Packet Length Variance",
            "FIN Flag Count",
            "SYN Flag Count",
            "RST Flag Count",
            "PSH Flag Count",
            "ACK Flag Count",
            "URG Flag Count",
            "CWE Flag Count",
            "ECE Flag Count",
        ]
    features = yaml.safe_load(
        (ROOT / "configs/data/features.yaml").read_text(encoding="utf-8")
    )
    categorical = [columns.index(name) for name in features["categorical"]]
    iat = [columns.index(name) for name in features["backdoor_iat"]]
    assert categorical == [0, 31, 32, 33, 34, 43, 44, 45, 46, 47, 48, 49, 50]
    assert iat == list(range(16, 30))


def test_manifests_expand_to_expected_run_counts():
    primary = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiments.py",
            "--manifest",
            "configs/experiments/eaai-primary.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    attacks = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiments.py",
            "--manifest",
            "configs/experiments/eaai-attacks.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "22 run(s)" in primary.stdout
    assert "6 run(s)" in attacks.stdout


def test_eaai_reference_bundle_builds(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "analysis/build_paper_results.py",
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
    )
    verification = json.loads((tmp_path / "verification.json").read_text())
    assert verification["status"] == "verified-reference-bundle"
    assert set(verification["tables"]) == {str(number) for number in range(6, 15)}
    assert verification["tables"] == {
        "6": 11,
        "7": 11,
        "8": 11,
        "9": 11,
        "10": 4,
        "11": 4,
        "12": 9,
        "13": 15,
        "14": 15,
    }


def test_eaai_figures_render_from_reference_bundle(tmp_path):
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            "analysis/build_paper_results.py",
            "--output",
            str(tables),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "analysis/plot_paper_results.py",
            "--tables-dir",
            str(tables),
            "--output-dir",
            str(figures),
        ],
        cwd=ROOT,
        check=True,
    )
    expected = {
        "figure04_distribution.svg",
        "figure07_results.svg",
        "figure08_results.svg",
        "figure09_backdoor.svg",
        "figure10_cross_architecture.svg",
    }
    assert {path.name for path in figures.iterdir()} == expected
    assert all(
        path.read_text().startswith('<svg xmlns="http://www.w3.org/2000/svg"')
        for path in figures.iterdir()
    )


def test_completed_runs_export_to_builder_schema(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "naive_s1_seed42"
    run_dir.mkdir(parents=True)
    artifact = {
        "run_id": "naive_s1_seed42",
        "return_code": 0,
        "git_commit": "abc123",
        "configuration": {"strategy": "naive", "scenario": 1, "seed": 42},
        "metrics": {
            "complete": True,
            "checkpoints": [
                {"experience": 0, "accuracy": 0.9, "macro_f1": 0.8},
                {"experience": 1, "accuracy": 0.7, "macro_f1": 0.6},
            ],
            "bwt_accuracy": -0.2,
            "bwt_macro_f1": -0.3,
        },
    }
    (run_dir / "run.json").write_text(json.dumps(artifact), encoding="utf-8")
    output = tmp_path / "summaries"
    subprocess.run(
        [
            sys.executable,
            "analysis/export_run_summaries.py",
            "--runs",
            str(runs),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    assert (
        (output / "baselines.csv")
        .read_text()
        .splitlines()[1]
        .startswith("Naive,CI,0.7,0.6")
    )
    assert (output / "per_experience.csv").read_text().count("\n") == 3
    assert json.loads((output / "provenance.json").read_text())["sources"] == [
        {"run_id": "naive_s1_seed42", "git_commit": "abc123"}
    ]
