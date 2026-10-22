"""Fig 2: CG occurrence probability on heatwave and non-heatwave days, unadjusted
and fixed-effects adjusted."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.figures import figure_path, save_figure
from common.style import BLUE, ORANGE, MID_WIDTH, apply_style

apply_style()

SAMPLES = ['All land days', 'FE-informative\nsample', 'FE-informative\nsample']


def plot(source):
    report=source/'occurrence'
    raw=pd.read_csv(report/'tables/unadjusted_probabilities.csv')
    effects=pd.read_csv(report/'tables/adjusted_effects_95CI.csv').set_index('metric')
    values=np.column_stack([raw[raw.result==name].sort_values('heatwave').probability.to_numpy()
                            for name in raw.result.unique()]+[effects.loc[['p_nonheatwave','p_heatwave'],'estimate'].to_numpy()])*100
    ratio=values[1]/values[0]
    rr=effects.loc['occurrence_ratio']
    assert np.isclose(ratio[2],rr.estimate)

    fig,ax=plt.subplots(figsize=(MID_WIDTH,3.4))
    groups=np.arange(3); width=.34; gap=.03

    # Probability bars: unadjusted groups light, the adjusted estimate in full colour.
    with plt.rc_context({'hatch.color':'white','hatch.linewidth':.7}):
        for j,(color,hatch) in enumerate([(ORANGE,''),(BLUE,'///')]):
            x=groups+(j-.5)*(width+gap)
            for k in range(3):
                ax.bar(x[k],values[j,k],width=width,color=color,alpha=1 if k==2 else .45,hatch=hatch,linewidth=0,zorder=3)
            row=effects.loc['p_nonheatwave' if j==0 else 'p_heatwave']
            ax.errorbar(x[2],values[j,2],yerr=[[values[j,2]-row.ci95_lower*100],[row.ci95_upper*100-values[j,2]]],
                        fmt='none',ecolor='#222222',elinewidth=1.1,capsize=3,zorder=4)
    top_value=max(values.max(),effects.loc[['p_nonheatwave','p_heatwave'],'ci95_upper'].max()*100)
    ax.set(ylabel='CG occurrence probability (%)',ylim=(0,top_value*1.1),xlim=(-.6,2.6))
    ax.set_xticks(groups,SAMPLES)
    ax.tick_params(axis='x',length=0)
    ax.grid(axis='y',alpha=.2,zorder=0)

    # Second tier of x labels: a bracket under each group of samples, just
    # below the two-line tick labels.
    tr=blended_transform_factory(ax.transData,ax.transAxes)
    y=-.135
    for (x0,x1),label in [((-.45,1.45),'Unadjusted'),((1.55,2.45),'Adjusted')]:
        ax.plot([x0,x0,x1,x1],[y+.02,y,y,y+.02],transform=tr,c='black',lw=.7,clip_on=False)
        ax.text((x0+x1)/2,y-.02,label,transform=tr,ha='center',va='top',fontsize=9)

    target=figure_path('probability', source)
    save_figure(fig, 'probability', run_dir=source, dpi=1200, bbox_inches='tight');plt.close(fig)
    print(f'HW/non-HW ratio: {ratio[0]:.3f} (all land days), {ratio[1]:.3f} (FE-informative), '
          f'{ratio[2]:.3f} [{rr.ci95_lower:.3f}, {rr.ci95_upper:.3f}] (adjusted)')
    print(target)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    plot(parser.parse_args().source)
