#!/usr/bin/env python3
"""Validate figures across the main-run, supporting and sensitivity directories.

This command does not copy, rename or delete images. The paper figure index lives in
common/figures.py and is used by the plotting scripts at save time.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.figures import FIGURES, EXTRA_FIGURES, SENSITIVITY_FIGURES, figure_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['all', 'figures', 'sensitivity'], default='all',
                        help='Check all 23 figures, the 19 from the figures stage, or the 4 from the sensitivity stage.')
    args = parser.parse_args()
    selected = {number: stem for number, stem in FIGURES.items()
                if args.group == 'all'
                or (number in SENSITIVITY_FIGURES) == (args.group == 'sensitivity')}
    stems = list(selected.values())
    if args.group != 'sensitivity':
        stems.extend(sorted(EXTRA_FIGURES))
    missing = [str(figure_path(stem)) for stem in stems if not figure_path(stem).is_file()]
    if missing:
        sys.exit('Missing paper figures:\n  ' + '\n  '.join(missing))
    print(f'Checked {len(selected)} paper figures and {len(stems) - len(selected)} extra supporting figures in their assigned directories.')


if __name__ == '__main__':
    main()
