"""Journal figure style: Arial 8-10 pt, bold panel letters outside the axes."""
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

FULL_WIDTH = 7.2  # inches; double-column journal width, figures are drawn 1:1
MID_WIDTH = 4.7   # inches; 1.5-column width, for single-panel figures

BLUE = '#1665a3'
ORANGE = '#ff8512'

# Liberation Sans is metrically identical to Arial and is used where Arial is not installed.
RC = {'font.family': 'sans-serif',
      'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica', 'Nimbus Sans', 'DejaVu Sans'],
      'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 10, 'legend.fontsize': 8,
      'xtick.labelsize': 8, 'ytick.labelsize': 8}


def apply_style():
    plt.rcParams.update(RC)


def panel_letter(ax, c, x=-.14, y=1.08):
    ax.text(x, y, c, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')


def label_panels(fig, axes, letters, dy=3):
    """Bold letters at the top-left corner of each panel's full extent (frame
    plus tick and axis labels), dy points above the frame, so every letter
    sits at its panel's outer corner whatever the label widths. Call once the
    layout is final (after tight_layout / subplots_adjust)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    shift = mtransforms.offset_copy(fig.transFigure, fig=fig, y=dy, units='points')
    for ax, c in zip(axes, letters):
        x0 = to_fig.transform((ax.get_tightbbox(renderer).x0, 0))[0]
        fig.text(x0, ax.get_position().y1, c, transform=shift, fontsize=11,
                 fontweight='bold', va='bottom', ha='left')
