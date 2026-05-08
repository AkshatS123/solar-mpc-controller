"""Smoke test for the Tesphase replay script using a synthetic stand-in."""

import subprocess
import sys
from pathlib import Path

import pytest

from solar_mpc.traces import synthetic_trace

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "replay_tesphase.py"


@pytest.fixture()
def synthetic_parquet(tmp_path: Path) -> Path:
    trace = synthetic_trace(days=2, step_minutes=15)
    out = tmp_path / "synthetic_replay.parquet"
    trace.to_parquet(out)
    return out


def test_replay_script_runs_against_synthetic_trace(synthetic_parquet: Path, tmp_path: Path) -> None:
    out_fig = tmp_path / "out.png"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace",
            str(synthetic_parquet),
            "--out",
            str(out_fig),
            "--soc-initial",
            "0.4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out_fig.exists()
    assert "MPC" in result.stdout
    assert "RobustMPC" in result.stdout


def test_replay_script_errors_when_trace_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.parquet"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--trace", str(missing)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "not found" in result.stderr
