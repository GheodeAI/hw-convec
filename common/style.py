"""Journal figure style: Arial 8-10 pt, bold panel letters outside the axes."""
import matplotlib.pyplot as plt

BLUE = '#1665a3'
ORANGE = '#ff8512'

RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
      'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 10, 'legend.fontsize': 8,
      'xtick.labelsize': 8, 'ytick.labelsize': 8}


def apply_style():
    plt.rcParams.update(RC)


def panel_letter(ax, c, x=-.14, y=1.08):
    ax.text(x, y, c, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')
