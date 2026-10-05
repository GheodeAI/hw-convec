"""Standalone grouped CG probabilities; figure 03 stays intact."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':9,'axes.titlesize':10,'axes.labelsize':10,'legend.fontsize':8,
    'xtick.labelsize':8,'ytick.labelsize':8})


def plot(source):
    report=source/'warmseason_fe_report_v1'
    originals=[source/'figures/03_intensity_FE.png',report/'CG_probability_comparison.png']
    hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in originals]
    raw=pd.read_csv(report/'unadjusted_probabilities.csv')
    effects=pd.read_csv(report/'adjusted_effects_95CI.csv').set_index('metric')
    values=np.column_stack([raw[raw.result==name].sort_values('heatwave').probability.to_numpy()
                            for name in raw.result.unique()]+[effects.loc[['p_nonheatwave','p_heatwave'],'estimate'].to_numpy()])*100
    blue,orange='#1665a3','#ff8512'
    # Plain house style (matches quintile_probability.png / 09_first_CG_compound.png):
    # default spines, no grid, centered title, bordered legend.
    fig,top=plt.subplots(figsize=(7.6,4.6))
    width=.27; groups=np.arange(3)
    for j,color,hatch,label in [(0,orange,'','Non-heatwave days'),(1,blue,'///','Heatwave days')]:
        x=groups+(j-.5)*width
        top.bar(x,values[j],width=width,color=color,hatch=hatch,label=label,zorder=3)
        row=effects.loc['p_nonheatwave' if j==0 else 'p_heatwave']
        lower,upper=row.ci95_lower*100,row.ci95_upper*100
        top.errorbar(x[2],values[j,2],yerr=[[values[j,2]-lower],[upper-values[j,2]]],
                     fmt='none',ecolor='#222222',elinewidth=1.1,capsize=3,zorder=4)
    # Scale to the data (including the FE-adjusted CI caps) rather than a fixed
    # ceiling, which silently clipped the heatwave bars once the corrected
    # lightning data pushed P(CG | heatwave) above 10%.
    top_value=max(values.max(),effects.loc[['p_nonheatwave','p_heatwave'],'ci95_upper'].max()*100)
    top.set(ylabel='CG occurrence probability (%)',
            ylim=(0,top_value*1.12),xlim=(-.6,2.6),xticks=groups,
            xticklabels=['Unadjusted\nFull sample','Unadjusted\nFE-informative sample','FE-adjusted\nFE-informative sample'])
    top.grid(axis='y', alpha=.2, zorder=0)
    fig.tight_layout()
    target=source/'figures/13_probability.png'
    fig.savefig(target,dpi=1200);plt.close(fig)
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in originals]
    print(target)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    plot(parser.parse_args().source)
