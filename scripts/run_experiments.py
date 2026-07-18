#!/usr/bin/env python3
"""Run a portable ReplayIDS experiment manifest and capture structured artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.result_io import parse_log  # noqa: E402


def expand(manifest: dict) -> list[dict]:
    defaults = {
        key: value
        for key, value in manifest.get("defaults", {}).items()
        if key != "env"
    }
    runs = []
    for group in manifest["runs"]:
        matrix = group["matrix"]
        keys = list(matrix)
        for values in itertools.product(*(matrix[key] for key in keys)):
            runs.append({**defaults, **dict(zip(keys, values))})
    return runs


def run_id(run: dict) -> str:
    parts = [str(run["strategy"]), f"s{run.get('scenario', 1)}"]
    if "memory" in run:
        parts.append(f"b{run['memory']}")
    if "poison_rate" in run:
        parts.append(f"p{run['poison_rate']}")
    parts.append(f"seed{run.get('seed', 42)}")
    return "_".join(parts)


def command(run: dict) -> list[str]:
    cmd = [
        sys.executable,
        "main.py",
        "--scenario",
        str(run.get("scenario", 1)),
        "--seed",
        str(run.get("seed", 42)),
    ]
    strategy = run["strategy"]
    flags = {
        "naive": [],
        "ewc": ["--ewc"],
        "lwf": ["--lwf"],
        "icarl": ["--icarl"],
        "er_stratified": ["--er"],
        "er_balanced": ["--er", "--balanced", "True"],
        "oracle": ["--oracle"],
        "label_flip": ["--er", "--balanced", "True", "--lf"],
        "backdoor": ["--er", "--balanced", "True", "--mp"],
    }
    if strategy not in flags:
        raise ValueError(f"Unsupported strategy {strategy!r}")
    cmd.extend(flags[strategy])
    if "memory" in run:
        cmd.extend(["--mem", str(run["memory"])])
    if "poison_rate" in run:
        cmd.extend(["--poison_rate", str(run["poison_rate"])])
    return cmd


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--only", action="append", help="Run ID substring to include")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing run directory"
    )
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    selected = expand(manifest)
    if args.only:
        selected = [
            run for run in selected if any(token in run_id(run) for token in args.only)
        ]
    print(f"Manifest {manifest['name']}: {len(selected)} run(s)")

    for run in selected:
        identifier = run_id(run)
        cmd = command(run)
        print(identifier, "=>", " ".join(cmd))
        if args.dry_run:
            continue
        out = args.output_dir / identifier
        if out.exists() and any(out.iterdir()) and not args.overwrite:
            raise SystemExit(
                f"{identifier} already exists; choose another output directory or pass "
                "--overwrite"
            )
        out.mkdir(parents=True, exist_ok=True)
        log = out / "run.log"
        env = os.environ.copy()
        env.update(
            {
                key: str(value)
                for key, value in manifest.get("defaults", {}).get("env", {}).items()
            }
        )
        env["RUN_TAG"] = identifier + "_"
        started = dt.datetime.now(dt.timezone.utc)
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        finished = dt.datetime.now(dt.timezone.utc)
        artifact = {
            "schema_version": 1,
            "run_id": identifier,
            "configuration": run,
            "project_config": yaml.safe_load(
                (ROOT / "configs/config.yaml").read_text(encoding="utf-8")
            ),
            "command": cmd,
            "git_commit": git_sha(),
            "uv_lock_sha256": sha256(ROOT / "uv.lock"),
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "return_code": result.returncode,
            "host": {"platform": platform.platform(), "python": sys.version},
            "metrics": parse_log(log),
        }
        (out / "run.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        if result.returncode and not args.keep_going:
            raise SystemExit(f"{identifier} failed; see {log}")


if __name__ == "__main__":
    main()
