"""Diagnose peak daily land area and year balance in existing matched pairs.

No rematching and no changes to the source event catalogue or match pairs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(source,output):
    output.mkdir(parents=True,exist_ok=False)
    events=pd.read_csv(source/'tables/regional_events.csv')
    pairs=pd.read_csv(source/'tables/match_pairs.csv')
    with np.load(source/'data/grids.npz') as grid:
        lat=grid['latitude']; land=grid['era5_land']
    area_by_lat=6371.0**2*np.deg2rad(.25)*(np.sin(np.deg2rad(lat+.125))-np.sin(np.deg2rad(lat-.125)))
    rows=[]
    for year in range(2006,2026):
        with np.load(source/f'data/labels_{year}.npz') as data:
            labels=data['labels']; offset=int(data['offset']);dates=pd.DatetimeIndex(data['dates'])
            for label,slices in enumerate(ndimage.find_objects(labels),1):
                t,r,c=np.nonzero(labels[slices]==label)
                r+=slices[1].start;c+=slices[2].start
                assert land[r,c].all()
                daily=np.bincount(t,weights=area_by_lat[r],minlength=slices[0].stop-slices[0].start)
                peak=int(np.argmax(daily)); maximum=float(daily[peak])
                rows.append(dict(event_id=offset+label,max_daily_land_area_km2=maximum,
                                 log_max_daily_land_area=np.log(maximum),peak_area_date=str(dates[slices[0].start+peak].date())))
        print('Area calculated',year,flush=True)
    area=pd.DataFrame(rows)
    assert len(area)==len(events) and not area.event_id.duplicated().any()
    full=events.merge(area,on='event_id',validate='one_to_one')
    assert (full.max_daily_land_area_km2>0).all()
    assert (full.max_daily_land_area_km2<=full.area_km2+1e-6).all()
    full.to_csv(output/'events_with_peak_daily_land_area.csv',index=False)
    e=full[full.start_month.between(4,9)&(full.cg_coverage_fraction==1)].copy().set_index('event_id')
    comp=e[e.has_lightning]; non=e[~e.has_lightning]
    mc=e.loc[pairs.compound_event_id]; mn=e.loc[pairs.control_event_id]
    assert mc.has_lightning.all() and (~mn.has_lightning).all()
    area_results=[]
    for variable in ['max_daily_land_area_km2','log_max_daily_land_area']:
        scale=np.sqrt((comp[variable].var(ddof=1)+non[variable].var(ddof=1))/2)
        row=dict(variable=variable,baseline_pooled_sd=scale,
                 smd_before=(comp[variable].mean()-non[variable].mean())/scale,
                 smd_after=(mc[variable].mean()-mn[variable].mean())/scale)
        for name,group in [('compound_before',comp),('noncompound_before',non),('compound_after',mc),('noncompound_after',mn)]:
            row[name+'_n']=len(group);row[name+'_mean']=group[variable].mean();row[name+'_median']=group[variable].median()
        area_results.append(row)
    at=pd.DataFrame(area_results);at.to_csv(output/'area_smd.csv',index=False)
    year_results=[]
    for year in range(2006,2026):
        counts=[int((g.year==year).sum()) for g in [comp,non,mc,mn]]
        pc,pn,ac,an=[count/len(g) for count,g in zip(counts,[comp,non,mc,mn])]
        scale=np.sqrt((pc*(1-pc)+pn*(1-pn))/2)
        year_results.append(dict(year=year,compound_n_before=counts[0],noncompound_n_before=counts[1],
             compound_n_after=counts[2],noncompound_n_after=counts[3],
             compound_proportion_before=pc,noncompound_proportion_before=pn,
             compound_proportion_after=ac,noncompound_proportion_after=an,
             baseline_binary_pooled_sd=scale,smd_before=(pc-pn)/scale,smd_after=(ac-an)/scale,
             compound_share_within_year_before=counts[0]/sum(counts[:2]),
             compound_share_within_year_after=counts[2]/sum(counts[2:])))
    yt=pd.DataFrame(year_results);yt.to_csv(output/'year_indicator_smd.csv',index=False)
    fig,axs=plt.subplots(2,1,figsize=(12,7.5),sharex=True)
    for ax,stage,title in zip(axs,['before','after'],['Before matching','After existing matching']):
        ax.bar(yt.year-.2,yt['compound_proportion_'+stage]*100,width=.38,color='#1665a3',label='compound')
        ax.bar(yt.year+.2,yt['noncompound_proportion_'+stage]*100,width=.38,color='#ff8512',label='non-compound')
        ax.set(title=title,ylabel='Events within each group (%)');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    for ax in axs:ax.set_ylim(0,max(yt.filter(like='proportion_').max())*115)
    axs[-1].set_xticks(range(2006,2026));axs[-1].tick_params(axis='x',rotation=45);axs[-1].set_xlabel('Regional heatwave start year')
    fig.suptitle('Land regional heatwave year balance (CG grid-cell fraction ≥10% defines compound)')
    fig.tight_layout();fig.savefig(output/'year_proportions_before_after.png',dpi=200);plt.close(fig)
    result=dict(existing_pairs=len(pairs),compound_before=len(comp),noncompound_before=len(non),
                log_area_smd_before=float(at.iloc[1].smd_before),log_area_smd_after=float(at.iloc[1].smd_after),
                max_abs_year_smd_before=float(yt.smd_before.abs().max()),max_abs_year_smd_after=float(yt.smd_after.abs().max()),
                worst_year_before=int(yt.loc[yt.smd_before.abs().idxmax(),'year']),worst_year_after=int(yt.loc[yt.smd_after.abs().idxmax(),'year']),
                years_abs_smd_gt_0_1_before=int((yt.smd_before.abs()>.1).sum()),years_abs_smd_gt_0_1_after=int((yt.smd_after.abs()>.1).sum()))
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'README.md').write_text('''# 最大单日陆地面积与年份平衡诊断

本次仅检查原有 1:1 匹配对，没有加入新协变量重新匹配，也未覆盖原事件表、匹配表和图 06–08。

面积为区域事件在每个日期实际标记的陆地热浪网格面积之和，再取事件期间最大值，单位 km²。使用 0.25° 球面网格面积（地球半径 6371 km，与原目录一致），海岸网格按既有网格中心陆地掩膜纳入，不另乘陆地面积分数。原 area_km2 是整个事件空间并集面积，与新变量不同。

主诊断使用自然对数 ln(max_daily_land_area_km2)，无须加 1，因为所有事件面积为正。面积的 SMD = (复合均值 − 非复合均值) / 匹配前两组 pooled SD；匹配前后均固定使用该分母。另附原始面积的 SMD 供比较。

每个年份作为一个二元变量，分母为 sqrt([p_C(1−p_C)+p_N(1−p_N)]/2)，p_C/p_N 来自匹配前样本。匹配后沿用同一分母。SMD 为有符号值，最大值按绝对值比较。

样本限定为 2006—2025 年、4—9 月开始、CG 覆盖完整的陆地区域事件；复合阈值保持 CG 网格比例 ≥10%。

year_proportions_before_after.png 画 P(年份 | 组别)，每组 20 年比例合计 100%，直接对应年份平衡诊断；不是 P(复合 | 年份)。后者额外保存在 year_indicator_smd.csv 的 compound_share_within_year_* 列中。
''')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.source,args.output)
