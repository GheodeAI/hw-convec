"""Compact manuscript figure using saved FE estimates; no model refitting."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(out):
    raw=pd.read_csv(out/'unadjusted_probabilities.csv')
    effect=pd.read_csv(out/'adjusted_effects_95CI.csv').set_index('metric')
    colors=['#4383A6','#D67A3D']
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,
                         'axes.linewidth':.7,'xtick.major.width':.7,'ytick.major.width':.7}):
        fig,axes=plt.subplots(1,3,figsize=(9,3.5),sharey=True)
        titles=['Unadjusted\nFull sample','Unadjusted\nFE-informative sample',
                'FE-adjusted\nFE-informative sample']
        for k,ax in enumerate(axes):
            if k<2:
                rows=raw[raw.result==raw.result.unique()[k]].sort_values('heatwave')
                values=rows.probability.to_numpy()*100
            else:
                rows=effect.loc[['p_nonheatwave','p_heatwave']]
                values=rows.estimate.to_numpy()*100
            ax.bar([0,1],values,width=.5,color=colors,zorder=3)
            upper=values
            if k==2:
                upper=rows.ci95_upper.to_numpy()*100
                ax.errorbar([0,1],values,yerr=np.vstack([values-rows.ci95_lower.to_numpy()*100,upper-values]),
                            fmt='none',ecolor='#222222',elinewidth=1,capsize=3,capthick=1,zorder=4)
            for x,value,top in zip([0,1],values,upper):
                ax.text(x,top+.15,f'{value:.3f}',ha='center',fontsize=9)
            ax.set(xlim=(-.6,1.6),ylim=(0,9),xticks=[0,1],xticklabels=['Non-heatwave','Heatwave'],
                   yticks=np.arange(0,9,2),title=titles[k])
            ax.title.set_fontsize(10)
            ax.text(-.03,1.13,f'({"abc"[k]})',transform=ax.transAxes,fontsize=10,weight='bold')
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(axis='x',length=0,pad=7)
            ax.grid(axis='y',color='#E9E9E9',linewidth=.5,zorder=0)
        axes[0].set_ylabel('CG occurrence probability (%)')
        fig.subplots_adjust(left=.075,right=.99,bottom=.16,top=.78,wspace=.2)
        fig.savefig(out/'CG_probability_comparison.png',dpi=400,bbox_inches='tight',facecolor='white')
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    plot(parser.parse_args().output_dir)
