"""Compatibility entry point: render the probability figure in outputs/main_analysis/occurrence."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.figures.plot_standalone_heatwave_probability import plot as plot_paper


def plot(out):
    plot_paper(out.parent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Existing occurrence directory containing the input tables.')
    plot(parser.parse_args().output_dir)
