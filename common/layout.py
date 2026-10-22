"""Locations of robustness checks associated with an analysis run."""
from pathlib import Path

from common.paths import OUTPUTS_DIR


def sensitivity_dir(run):
    """Use the global sensitivity root for main_analysis, isolate custom runs."""
    run = Path(run)
    if run.resolve() == (OUTPUTS_DIR / "main_analysis").resolve():
        return OUTPUTS_DIR / "sensitivity"
    return run / "sensitivity"
