"""Render the ten requested figures from the confirmed land-only analysis."""
import argparse
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm,ListedColormap
import shapefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import YEARS, ROOT
from common.environment import MATCHV, ORDER, LABELS, UNITS
from common.io import dump
from common.stats import bh
from common.style import BLUE, ORANGE, FULL_WIDTH, MID_WIDTH, apply_style, label_panels, panel_letter as letter

apply_style()

def save(fig,out,name,dpi=1200):
    fig.savefig(out/'figures'/f'{name}.png',dpi=dpi,bbox_inches='tight')
    plt.close(fig)

def maps(out):
    grid=np.load(out/'data/grids.npz'); mask=grid['cg_land']; ids=grid['cg_ids']; lat=grid['cg_latitude'];lon=grid['cg_longitude']
    n=len(ids); hn=np.zeros(n,np.int64); hs=hn.copy();total=hn.copy();cg=hn.copy()
    for year in YEARS:
        with np.load(out/f'data/daily_{year}.npz') as d:
            month=pd.DatetimeIndex(d['dates']).month; use=(month>=4)&(month<=9)
            h=d['hw'][use].astype(bool);y=d['cg'][use].astype(bool)
            hn+=h.sum(axis=0);hs+=(h&y).sum(axis=0);cg+=y.sum(axis=0);total+=len(h)
    pa=cg/total;ph=np.divide(hs,hn,out=np.full(n,np.nan),where=hn>=5)
    ratio=np.divide(ph,pa,out=np.full(n,np.nan),where=pa>0);diff=(ph-pa)*100
    z=np.divide(ph-pa,np.sqrt(pa*(1-pa)/np.maximum(hn,1)),out=np.full(n,np.nan),where=(pa>0)&(pa<1))
    # 71% of cells are significant, so stippling those would cover the map and
    # hide the field itself; mark the minority that are NOT significant instead.
    # NaN z (pa outside (0,1)) fails both comparisons and stays unmarked.
    not_significant=2*stats.norm.sf(abs(z))>=.05
    def field(x):
        a=np.full(mask.size,np.nan);a[ids]=x;return a.reshape(mask.shape)
    # Grey should mean "land cell, but fewer than 5 heatwave days in the record"
    # (no meaningful heatwave sample there) -- not ocean, which is simply
    # outside `ids` and stays the plain white figure background.
    no_heatwave_land=field(np.where(np.isfinite(ph),np.nan,1.0))
    xr.Dataset({k:(('latitude','longitude'),field(v)) for k,v in dict(hw_n=hn,hw_cg=hs,all_n=total,all_cg=cg,p_hw=ph,p_all=pa,ratio=ratio,difference_pp=diff,p_value=2*stats.norm.sf(abs(z))).items()},coords=dict(latitude=lat,longitude=lon),attrs=dict(min_heatwave_days=5,comparison='HW versus all warm-season days, land CG centres',significance='pointwise normal binomial approximation against climatology, no multiplicity adjustment; descriptive')).to_netcdf(out/'data/occurrence_maps.nc')
    cmap=LinearSegmentedColormap.from_list('orangeblue',[(0,'#add7e5'),(.5,'#ffffff'),(.75,'#ffdfa6'),(1,'#ce793c')])
    # log2(ratio) so a halving and a doubling of CG occurrence get equal visual
    # weight either side of the neutral colour at ratio = 1 (log2 = 0); exact
    # zero ratios (heatwave days present but literally no CG) are floored, not
    # dropped, so they still render as the coldest colour instead of vanishing.
    log_ratio=np.log2(np.clip(ratio,2.0**-6,None))
    # Limits re-tuned for the corrected lightning data (median ratio 2.8, p95
    # 10.2; median difference +8.4 pp, p95 23 pp). The old 4x / +-8 pp range
    # saturated 29% and 52% of cells respectively. The difference scale is
    # deliberately asymmetric -- only 5% of cells fall below -1.4 pp -- while
    # keeping white pinned at zero.
    ratio_vmax=8.0
    diff_vmin,diff_vmax=-8.0,24.0
    fig,axs=plt.subplots(1,2,figsize=(FULL_WIDTH,4.1)); xx,yy=np.meshgrid(lon,lat)
    with shapefile.Reader(str(ROOT/'data/shapefiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')) as sf:
        shapes=sf.shapes()
    for ax,data,norm,title,label,c in zip(axs,[log_ratio,diff],[TwoSlopeNorm(vmin=-np.log2(ratio_vmax),vcenter=0,vmax=np.log2(ratio_vmax)),TwoSlopeNorm(vmin=diff_vmin,vcenter=0,vmax=diff_vmax)],['Occurrence ratio','Occurrence difference'],['P(CG | heatwave) / P(CG | climatology)','P(CG | heatwave) − P(CG | climatology) (pp)'],'ab'):
        ax.pcolormesh(xx,yy,no_heatwave_land,cmap=ListedColormap(['#d9d9d9']),vmin=0,vmax=1,shading='auto',rasterized=True)
        mesh=ax.pcolormesh(xx,yy,field(data),cmap=cmap,norm=norm,shading='auto',rasterized=True)
        for shp in shapes:
            x0,y0,x1,y1=shp.bbox
            if x1< -10 or x0>33 or y1<36 or y0>72:continue
            pts=np.asarray(shp.points); parts=list(shp.parts)+[len(pts)]
            for a,b in zip(parts[:-1],parts[1:]):ax.plot(pts[a:b,0],pts[a:b,1],c='#555555',lw=.35)
        ax.set(xlim=(-10,33),ylim=(36,72),xlabel='Longitude',ylabel='Latitude');ax.set_aspect('equal')
        ax.scatter(xx.ravel()[ids[not_significant]],yy.ravel()[ids[not_significant]],s=.06,c='black',alpha=.6,linewidth=0,rasterized=True)
        cb=fig.colorbar(mesh,ax=ax,orientation='horizontal',fraction=.05,pad=.13,extend='both');cb.set_label(label)
        if title=='Occurrence ratio':
            ticks=np.array([.125,.25,.5,1,2,4,8]);cb.set_ticks(np.log2(ticks));cb.set_ticklabels([f'{t:g}' for t in ticks])
    fig.tight_layout();label_panels(fig,axs,'ab');save(fig,out,'01_occurrence_maps',dpi=600)

def panel_intensity_enhancement(ax,raw,base):
    ratio=raw.p_cg/base; tau,p=stats.kendalltau(raw.mean_excess,ratio)
    right=ax.twinx()
    right.bar(raw.mean_excess,raw.p_cg*100,width=.1,color='#aac7d8',alpha=.95)
    right.axhline(base*100,c='grey',ls='--')
    ax.plot(raw.mean_excess,ratio,'o-',c='#be5225',lw=1.6,ms=5)
    ax.set_zorder(right.get_zorder()+1);ax.patch.set_visible(False)
    for q,x,r in zip(raw.q,raw.mean_excess,ratio):ax.annotate(f'Q{q}',(x,r),xytext=(0,10),textcoords='offset points',ha='center')
    ax.set(xlabel='Mean heatwave intensity, Twmax above threshold (°C)',ylabel='Enhancement ratio',ylim=(min(1,float(ratio.min())*.95),float(ratio.max())*1.09));right.set_ylabel('CG occurrence probability (%)');right.set_ylim(0,float(raw.p_cg.max())*115)
    ax.grid(axis='y',alpha=.25)
    return ratio,tau,p

def panel_intensity_fe(ax,a):
    ax.fill_between(a.q,a.bootstrap_ci95_lower*100,a.bootstrap_ci95_upper*100,color=BLUE,alpha=.18)
    ax.plot(a.q,a.adjusted_probability*100,'o-',c=BLUE,lw=1.6,ms=5);ax.set_xticks(range(1,6),[f'Q{i}' for i in range(1,6)])
    ax.set(xlabel='Heatwave intensity quintile',ylabel='Adjusted CG occurrence probability (%)');ax.grid(axis='y',alpha=.25)

def quintiles(out):
    raw=pd.read_csv(out/'tables/quintile_raw.csv'); base=pd.read_csv(out/'tables/occurrence_probability.csv').p_all.iloc[0]
    fig,ax=plt.subplots(figsize=(MID_WIDTH,3.2))
    ratio,tau,p=panel_intensity_enhancement(ax,raw,base)
    fig.tight_layout();save(fig,out,'02_intensity_enhancement')
    raw.assign(p_climatology=base,enhancement_ratio=ratio,kendall_tau=tau,kendall_p=p).to_csv(out/'tables/intensity_enhancement.csv',index=False)

    a=pd.read_csv(out/'tables/quintile_adjusted.csv')
    fig,ax=plt.subplots(figsize=(MID_WIDTH,3.2))
    panel_intensity_fe(ax,a)
    fig.tight_layout();save(fig,out,'03_intensity_FE')

def balance(out,only_descriptive=False):
    e=pd.read_csv(out/'area_year_diagnostics_v1/events_with_peak_daily_land_area.csv');e=e[e.start_month.between(4,9)&(e.cg_coverage_fraction==1)]
    groups=[e[~e.has_lightning],e[e.has_lightning]]
    cov=['center_latitude','center_longitude','duration','intensity','log_max_daily_land_area'];names=['Centre latitude','Centre longitude','Duration','Intensity','log(peak daily land area)']
    b=pd.read_csv(out/'area_match_pair_year_FE_v1/match_balance.csv').set_index('variable')
    # Three rows at journal width: two centre-density maps (shared colourbar)
    # and start month; three CDFs sharing y; start year across the full width.
    fig=plt.figure(figsize=(FULL_WIDTH,7.6))
    rows=fig.add_gridspec(3,1,height_ratios=[1.2,1,.85],hspace=.45)
    top=rows[0].subgridspec(1,5,width_ratios=[1,1,.06,.42,1.1],wspace=.2)
    mid=rows[1].subgridspec(1,3,wspace=.12)
    map_a=fig.add_subplot(top[0]);map_b=fig.add_subplot(top[1],sharex=map_a,sharey=map_a)
    cax=fig.add_subplot(top[2]);month_ax=fig.add_subplot(top[4])
    cdf=[fig.add_subplot(mid[0])];cdf+=[fig.add_subplot(mid[i],sharey=cdf[0]) for i in (1,2)]
    axs=np.array([[map_a,map_b,month_ax],cdf]);h=[]
    for ax,g,title in zip(axs[0,:2],groups,['non-compound heatwave centres','compound heatwave centres']):
        hist,xe,ye=np.histogram2d(g.center_longitude,g.center_latitude,bins=[np.arange(-10,67.6,2.5),np.arange(36,73,1)])
        h.append((hist.T/len(g)*100,xe,ye,ax,title))
    vmax=1.4
    lon_limits=(2.5*np.floor(e.center_longitude.min()/2.5),2.5*np.ceil(e.center_longitude.max()/2.5))
    lat_limits=(np.floor(e.center_latitude.min()),np.ceil(e.center_latitude.max()))
    for (z,xe,ye,ax,title),c in zip(h,'ab'):
        mesh=ax.pcolormesh(xe,ye,z,cmap='Reds',vmin=0,vmax=vmax);ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)')
        ax.set(xlim=lon_limits,ylim=lat_limits)
    ax=axs[0,2]
    for g,shift,col,hatch,label in zip(groups,[-.2,.2],[ORANGE,BLUE],['','///'],['non-compound','compound']):
        freq=g.start_month.value_counts().reindex(range(4,10),fill_value=0)/len(g)*100;ax.bar(np.arange(4,10)+shift,freq,width=.38,color=col,hatch=hatch,label=label)
    ax.set(xlabel='Event start month',ylabel='Events (%)',xlim=(3.5,9.5));ax.set_xticks(range(4,10))
    for ax,var,ls,c in zip(axs[1,:2],['duration','intensity'],['-','-'],'de'):
        for g,col,dash,label in zip(groups,[ORANGE,BLUE],['-','--'],['non-compound','compound']):
            x=np.sort(g[var].values);ax.step(x,np.arange(1,len(x)+1)/len(x),c=col,ls=dash,label=label)
        ax.set(xlabel='Duration (days)' if var=='duration' else 'Mean Tw − threshold (°C)',ylabel='Empirical cumulative\nprobability');ax.grid(alpha=.2)
    ax=axs[1,2]
    for g,col,dash,label in zip(groups,[ORANGE,BLUE],['-','--'],['non-compound','compound']):
        x=np.sort(g.log_max_daily_land_area.values);ax.step(x,np.arange(1,len(x)+1)/len(x),c=col,ls=dash,label=label)
    ax.set(xlabel='ln(peak daily land area / km²)',ylabel='Empirical cumulative probability');ax.grid(alpha=.2)
    year_ax=fig.add_subplot(rows[2])
    years=np.arange(2006,2026);year_rows=[]
    for g,shift,col,hatch,label in zip(groups,[-.2,.2],[ORANGE,BLUE],['','///'],['non-compound','compound']):
        counts=g.year.value_counts().reindex(years,fill_value=0)
        freq=counts/len(g)*100
        assert counts.sum()==len(g) and np.isclose(freq.sum(),100)
        year_ax.bar(years+shift,freq,width=.38,color=col,hatch=hatch,label=label)
        year_rows.extend(dict(group=label,year=int(year),n_events=int(counts.loc[year]),group_total=len(g),within_group_percent=float(freq.loc[year])) for year in years)
    year_ax.set_xticks(years);year_ax.tick_params(axis='x',labelrotation=45)
    year_ax.set(xlabel='Event start year',ylabel='Within-group events (%)',xlim=(2005.4,2025.6))
    year_ax.grid(axis='y',alpha=.2);year_ax.set_axisbelow(True)
    pd.DataFrame(year_rows).to_csv(out/'tables/figure06_year_distribution.csv',index=False)
    # Shared axes: latitude labelled on a only, cumulative probability on d only.
    map_b.set_ylabel('');map_b.tick_params(labelleft=False)
    for ax in cdf[1:]:
        ax.set_ylabel('');ax.tick_params(labelleft=False)
    fig.subplots_adjust(left=.09,right=.98,bottom=.07,top=.97)
    cb=fig.colorbar(mesh,cax=cax,extend='max',ticks=np.arange(0,1.401,.2));cb.set_label('Events per bin (%)',fontsize=plt.rcParams['axes.labelsize'],labelpad=3);cb.ax.tick_params(labelsize=plt.rcParams['ytick.labelsize'],pad=2)
    label_panels(fig,[map_a,map_b,month_ax,*cdf,year_ax],'abcdefg')
    save(fig,out,'06_descriptive_balance',dpi=600)
    if only_descriptive:return
    fig,ax=plt.subplots(figsize=(10,6));ys=np.arange(len(cov))
    before=abs(b.loc[cov,'smd_before'].values);after=abs(b.loc[cov,'smd_after'].values)
    for y,x1,x2 in zip(ys,before,after):ax.annotate('',xy=(x2,y),xytext=(x1,y),arrowprops=dict(arrowstyle='->',ls='--',color='grey',lw=2))
    ax.scatter(before,ys,c='#ec5353',s=100);ax.scatter(after,ys,c='#4e79a7',s=100,marker='s');ax.axvline(.1,c='grey',ls='--');ax.set_yticks(ys,names);ax.invert_yaxis();ax.set(xlabel='Absolute standardized mean difference');ax.grid(axis='x',alpha=.2);fig.tight_layout();save(fig,out,'07_match_balance')

def adjusted_environment_plot(out):
    before=pd.read_csv(out/'tables/environment_before_after.csv').set_index('variable').loc[ORDER]
    after=pd.read_csv(out/'area_match_pair_year_FE_v1/environment_pair_year_FE.csv').set_index('variable').loc[ORDER]
    fig,ax=plt.subplots(figsize=(MID_WIDTH,3.4));y=np.arange(6)
    for yi,v in zip(y,ORDER):
        ax.annotate('',xy=(after.loc[v,'standardized_difference'],yi),xytext=(before.loc[v,'smd_before'],yi),arrowprops=dict(arrowstyle='->',ls='--',color='grey',lw=1.2))
    ax.scatter(before.smd_before,y,c='#ec5353',s=36)
    ax.errorbar(after.standardized_difference,y,xerr=[after.standardized_difference-after.standardized_ci95_lower,after.standardized_ci95_upper-after.standardized_difference],fmt='s',c='#4e79a7',ms=5,capsize=3)
    ax.set_yticks(y,[r'$q_{850}$','TCWV',r'$DPD_{2m}$',r'$qv_{850}$',r'$qu_{850}$',r'$Z_{500}$']);ax.invert_yaxis();ax.axvline(0,c='black');ax.grid(axis='x',alpha=.2)
    ax.set(xlabel='Standardized compound − non-compound difference')
    fig.tight_layout();save(fig,out,'08_environment_matching')
    table=after.copy();table['smd_before_matching']=before.smd_before
    table.to_csv(out/'tables/final_environment_figure08.csv')

def environments(out,only_means=False):
    env=pd.read_csv(out/'tables/event_environment_anomaly.csv').set_index('event_id');pairs=pd.read_csv(out/'tables/match_pairs.csv')
    rows=[];rng=np.random.default_rng(20260914)
    for v in ORDER:
        c=env.loc[env.has_lightning,v].dropna().values;n=env.loc[~env.has_lightning,v].dropna().values
        scale=np.sqrt((np.var(c,ddof=1)+np.var(n,ddof=1))/2)
        a=env.loc[pairs.compound_event_id,v].values;b=env.loc[pairs.control_event_id,v].values;valid=np.isfinite(a)&np.isfinite(b);delta=a[valid]-b[valid]
        boot=np.mean(delta[rng.integers(len(delta),size=(1000,len(delta)))],axis=1)/scale
        rows.append(dict(variable=v,n_compound=len(c),n_noncompound=len(n),compound_mean=c.mean(),noncompound_mean=n.mean(),
            compound_ci_half=stats.t.ppf(.975,len(c)-1)*stats.sem(c),noncompound_ci_half=stats.t.ppf(.975,len(n)-1)*stats.sem(n),
            difference=c.mean()-n.mean(),p=stats.ttest_ind(c,n,equal_var=False).pvalue,smd_before=(c.mean()-n.mean())/scale,
            smd_after=delta.mean()/scale,smd_after_lower=np.quantile(boot,.025),smd_after_upper=np.quantile(boot,.975),n_pairs=len(delta),paired_difference=delta.mean(),paired_p=stats.ttest_1samp(delta,0).pvalue))
    tab=pd.DataFrame(rows);tab['p_bh']=bh(tab.p);tab['paired_p_bh']=bh(tab.paired_p);tab.to_csv(out/'tables/environment_before_after.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(FULL_WIDTH,5.0))
    for ax,r,c in zip(axs.flat,tab.itertuples(),'abcdef'):
        ax.bar([0,1],[r.noncompound_mean,r.compound_mean],width=.4,yerr=[r.noncompound_ci_half,r.compound_ci_half],color=[ORANGE,BLUE],hatch=['','///'],capsize=2.5);ax.set_xlim(-.49,1.49)
        ax.set_xticks([0,1],['non-compound','compound']);ax.set_ylabel(UNITS[r.variable]);ax.axhline(0,c='black',lw=.7);ax.grid(axis='y',alpha=.2)
        lo,hi=ax.get_ylim()
        if hi<=0:ax.set_ylim(top=-.08*lo)  # keep the zero line off the top frame
    fig.tight_layout(w_pad=.3,h_pad=1.6);label_panels(fig,axs.flat,'abcdef');save(fig,out,'04_environment_means')
    if only_means:return
    if (out/'area_match_pair_year_FE_v1/environment_pair_year_FE.csv').exists():
        adjusted_environment_plot(out);return
    fig,ax=plt.subplots(figsize=(11,6));y=np.arange(6)
    for r,yi in zip(tab.itertuples(),y):ax.annotate('',xy=(r.smd_after,yi),xytext=(r.smd_before,yi),arrowprops=dict(arrowstyle='->',ls='--',color='grey',lw=2))
    ax.scatter(tab.smd_before,y,c='#ec5353',s=90,label='before matching')
    ax.errorbar(tab.smd_after,y,xerr=[tab.smd_after-tab.smd_after_lower,tab.smd_after_upper-tab.smd_after],fmt='s',c='#4e79a7',capsize=5,label='after matching (95% paired bootstrap CI)')
    ax.set_yticks(y,[r'$q_{850}$','TCWV',r'$DPD_{2m}$',r'$qv_{850}$',r'$qu_{850}$',r'$Z_{500}$']);ax.invert_yaxis();ax.axvline(0,c='black');ax.grid(axis='x',alpha=.2);ax.set(xlabel='Compound − non-compound difference / pre-match pooled SD');fig.tight_layout();save(fig,out,'08_environment_matching')

def spatial(out):
    e=pd.read_csv(out/'tables/regional_events.csv');e=e[e.start_month.between(4,9)&(e.cg_coverage_fraction==1)].reset_index(drop=True)
    index={eid:i for i,eid in enumerate(e.event_id)};sm=np.zeros((len(e),6,65,65),np.float32);n=np.zeros_like(sm,dtype=np.uint16)
    for p in sorted((out/'data').glob('spatial_environment_m*.npz')):
        with np.load(p) as d:
            ii=np.array([index[eid] for eid in d['event_ids']],dtype=np.intp);sm[ii]+=d['sums'];n[ii]+=d['counts']
    values=np.divide(sm,n,out=np.full(sm.shape,np.nan),where=n>0);del sm,n
    fig,axs=plt.subplots(2,3,figsize=(16.5,9));offset=np.arange(-8,8.001,.25);maps={}
    for ax,v,c in zip(axs.flat,ORDER,'abcdef'):
        j=MATCHV.index(v);a=values[e.has_lightning,j];b=values[~e.has_lightning,j]
        na=np.isfinite(a).sum(axis=0);nb=np.isfinite(b).sum(axis=0)
        with np.errstate(invalid='ignore'):
            diff=np.nanmean(a,axis=0)-np.nanmean(b,axis=0);p=stats.ttest_ind(a,b,axis=0,equal_var=False,nan_policy='omit').pvalue
        diff[(na<10)|(nb<10)]=np.nan
        lim=np.nanquantile(abs(diff),.98);lim=max(lim,1e-12)
        mesh=ax.pcolormesh(offset,offset,diff,cmap='RdBu_r',vmin=-lim,vmax=lim,shading='auto',rasterized=True)
        sig=(p<.05)&np.isfinite(diff);yy,xx=np.meshgrid(offset,offset,indexing='ij');ax.scatter(xx[sig],yy[sig],s=.12,c='black',alpha=.4,rasterized=True)
        ax.axhline(0,c='grey',lw=.7);ax.axvline(0,c='grey',lw=.7);ax.set(xlabel='Lon offset (deg)',ylabel='Lat offset (deg)');letter(ax,c);fig.colorbar(mesh,ax=ax,extend='both').set_label(UNITS[v])
        maps[v+'_difference']=(('lat_offset','lon_offset'),diff);maps[v+'_p_value']=(('lat_offset','lon_offset'),p);maps[v+'_n_compound']=(('lat_offset','lon_offset'),na);maps[v+'_n_noncompound']=(('lat_offset','lon_offset'),nb)
    xr.Dataset(maps,coords=dict(lat_offset=offset,lon_offset=offset),attrs=dict(center='land-event positive-excess intensity-weighted centre',average='time mean at each relative-coordinate pixel per event, then equal-weight event means',environment='sea retained; out-of-domain pixels missing',test='pointwise Welch p<0.05; no multiplicity adjustment; at least 10 events/group',matching='before matching')).to_netcdf(out/'data/event_centered_environment.nc')
    fig.tight_layout();save(fig,out,'05_event_centered_environment',dpi=600)

def curves(out):
    d=pd.read_parquet(out/'data/first_cg_event_values.parquet'); variables=['tcwv','cape','cin','omega700'];summary=[];changes=[]
    for kind in ['compound','nonhw_onset']:
        a=d[d.kind==kind]
        for v in variables:
            col=v+'_climatological_anomaly'
            for h,g in a.groupby('relative_hour'):
                x=g[col].dropna().values
                summary.append(dict(kind=kind,variable=v,hour=h,n=len(x),mean=np.mean(x),median=np.median(x),q25=np.quantile(x,.25),q75=np.quantile(x,.75),sem=stats.sem(x)))
            wide=a.pivot(index='id',columns='relative_hour',values=col)
            for prev,h in zip(range(-12,12,3),range(-9,13,3)):
                delta=(wide[h]-wide[prev]).dropna(); changes.append(dict(kind=kind,variable=v,hour=h,n=len(delta),difference=delta.mean(),p=stats.ttest_1samp(delta,0).pvalue))
    s=pd.DataFrame(summary);c=pd.DataFrame(changes);c['q_bh']=c.groupby(['kind','variable']).p.transform(lambda x:bh(x.values))
    s.to_csv(out/'tables/confirmed_curve_summary.csv',index=False);c.to_csv(out/'tables/confirmed_adjacent_changes.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(12.1,7.15))
    for ax,v in zip(axs.flat,variables):
        a=s[(s.kind=='compound')&(s.variable==v)].sort_values('hour');ax.fill_between(a.hour,a.q25,a.q75,color='#a5d0e5',alpha=.6,label='IQR');ax.plot(a.hour,a['mean'],'o-',c=BLUE,lw=2,label='Mean');ax.plot(a.hour,a['median'],'--',c='grey',label='Median')
        sig=c[(c.kind=='compound')&(c.variable==v)&(c.q_bh<.05)].hour; ss=a.set_index('hour').loc[sig];ax.scatter(ss.index,ss['mean'],marker='*',s=90,c='#a62a55',zorder=5,label='Adjacent change: BH q < 0.05')
        ax.axvline(0,c='black',lw=.7);ax.axhline(0,c='black',lw=.7);ax.set(title=LABELS[v],xlabel='Hours from first CG (nearest 3-hour reference)',ylabel=UNITS[v]);ax.set_xticks(range(-12,13,3));ax.grid(alpha=.2)
    axs[0,0].legend(ncol=2);fig.suptitle('Land grid-cell first CG within qualifying regional compound events: anomalies');fig.tight_layout();save(fig,out,'09_first_CG_compound')
    comparison_curve(out,s)

def comparison_curve(out,s=None):
    if s is None:s=pd.read_csv(out/'tables/confirmed_curve_summary.csv')
    variables=['tcwv','cape','cin','omega700']
    fig,axs=plt.subplots(2,2,figsize=(12.1,7.15))
    for ax,v in zip(axs.flat,variables):
        for kind,col,label in [('compound',BLUE,'compound-event first CG'),('nonhw_onset',ORANGE,'non-heatwave CG onset')]:
            a=s[(s.kind==kind)&(s.variable==v)].sort_values('hour');ax.plot(a.hour,a['mean'],'o-',c=col,lw=2,label=label)
        ax.axvline(0,c='black',lw=.7);ax.axhline(0,c='black',lw=.7);ax.set(title=LABELS[v],xlabel='Hours from CG reference (nearest 3-hour slot)',ylabel=UNITS[v]);ax.set_xticks(range(-12,13,3));ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=9);fig.suptitle('Land compound first CG vs non-heatwave CG onsets: Apr–Sep centres, 2006–2025');fig.tight_layout();save(fig,out,'10_compound_vs_nonHW_CG_onsets')

def timing(out):
    """Timing diagnostics using the same >=10% qualified compound sample."""
    a=pd.read_parquet(out/'data/confirmed_anchors.parquet')
    a=a[a.kind=='compound'].copy()
    a['start_day']=a.id.str.rsplit('_',n=1).str[-1].astype(int)
    a['duration']=0
    for year,g in a.groupby('year'):
        with np.load(out/f'data/labels_{year}.npz') as d:
            lab=d['labels'].reshape(len(d['labels']),-1)
            for row in g.itertuples():
                stop=np.flatnonzero(lab[row.start_day:,row.cell]==0)
                duration=int(stop[0]) if len(stop) else len(lab)-row.start_day
                a.loc[row.Index,'duration']=duration
    a['start_time']=pd.to_datetime(a.year.astype(str)+'-01-01')+pd.to_timedelta(a.start_day,unit='D')
    a['lag_hours']=(a.first_cg_time-a.start_time).dt.total_seconds()/3600
    assert ((a.lag_hours>=0)&(a.lag_hours<(a.duration+1)*24)).all()
    e=pd.read_csv(out/'tables/regional_events.csv',parse_dates=['first_cg_time'])
    e=e[e.has_lightning&(e.cg_coverage_fraction==1)&e.first_cg_time.dt.month.between(4,9)].copy()
    rows=[];fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for data,label,col in [(a,'Grid-cell first CG',BLUE),(e,'Regional first CG',ORANGE)]:
        phase=data.lag_hours/(24*data.duration)
        names=np.select([phase<1/3,phase<2/3,phase<1],['early','middle','late'],default='post_24h')
        counts=pd.Series(names).value_counts().reindex(['early','middle','late','post_24h'],fill_value=0)
        for name,n in counts.items():rows.append(dict(sample=label,phase=name,n=n,total=len(data),percentage=n/len(data)*100))
        x=np.sort(data.lag_hours);axs[0].step(x,np.arange(1,len(x)+1)/len(x),color=col,label=f'{label} (n={len(x):,})')
    tab=pd.DataFrame(rows)
    for label,shift,col in [('Grid-cell first CG',-.2,BLUE),('Regional first CG',.2,ORANGE)]:
        axs[1].bar(np.arange(4)+shift,tab.loc[tab['sample']==label,'percentage'],width=.38,color=col,label=label)
    axs[0].set(xlabel='Hours since heatwave onset',ylabel='Empirical cumulative probability');axs[0].legend();axs[0].grid(alpha=.2)
    axs[1].set_xticks(range(4),['First third','Middle third','Last third','Post-HW 24 h']);axs[1].set_ylabel('Events (%)');axs[1].legend()
    fig.suptitle('First-CG timing: land regional CG fraction ≥10%, Apr–Sep CG centres');fig.tight_layout();save(fig,out,'11_confirmed_first_CG_timing')
    a.to_parquet(out/'data/confirmed_compound_timing.parquet',index=False)
    tab.to_csv(out/'tables/confirmed_first_cg_timing.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['initial','environment','timing','all'],default='all');args=p.parse_args()
    if args.stage in ['initial','all']:maps(args.out);quintiles(args.out);balance(args.out)
    if args.stage in ['environment','all']:environments(args.out);spatial(args.out);curves(args.out)
    if args.stage in ['timing','all']:timing(args.out)
