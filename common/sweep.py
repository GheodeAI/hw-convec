"""Helpers shared by the absolute wet-bulb threshold sweeps."""
from __future__ import annotations
import subprocess

from common.paths import PROJECT_DIR

REPO = PROJECT_DIR
SCRIPTS = REPO / "scripts"
DEFAULT_THRESHOLDS = [18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0]


def format_threshold_suffix(threshold: float) -> str:
    # Matches build_wetbulb_heatwaves.py's format_threshold_suffix exactly, so
    # directory names line up: 20 -> "20", 18.5 -> "18p5".
    return f"{threshold:g}".replace(".", "p")


def method_label(threshold: float) -> str:
    return f"90p{format_threshold_suffix(threshold)}"


def run(cmd: list[str], dry_run: bool) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO)
