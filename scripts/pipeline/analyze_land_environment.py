#!/usr/bin/env python3
"""Recompute event environment and first-CG curves on land-only footprints.

Baseline: 2006–2025 mean for the same calendar month/day and UTC 3-hour slot.
Regional means use the union of event land cells over the event duration.
First-CG curves use land neighbours in a 3x3 ERA5 box, -12 to +12 hours.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import calendar
import numpy as np
import pandas as pd
from scipy import sparse, stats, ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import YEARS
from common.environment import VARS, MATCHV, CURVEV, load_fields
from common.io import dump
from common.stats import bh

def anchors(out):
    e=pd.read_csv(out/'tables/regional_events.csv',parse_dates=['start_time','end_time','first_cg_time'])
    grid=np.load(out/'data/grids.npz'); lat=grid['latitude']; lon=grid['longitude']; nx=len(lon)
    records=[]
    for row in e.itertuples():
        if row.has_lightning and row.cg_coverage_fraction==1 and 4<=row.start_month<=9:
            records.append(dict(id=row.regional_heatwave_event_id,kind='regional',year=row.year,event_id=row.event_id,
                         cell=row.anchor_cell,first_cg_time=row.first_cg_time,start_time=row.start_time,end_time=row.end_time,duration=row.duration,lag_hours=row.lag_hours))
    for year in YEARS:
        labels=np.load(out/f'data/labels_{year}.npz')['labels']; ndays=labels.shape[0]
        hour=sparse.load_npz(out/f'data/cg_hourly_era5_{year}.npz').tocsc()
        mask=labels.reshape(ndays,-1)>0; start_date=pd.Timestamp(year,1,1)
        for cell in np.flatnonzero(mask.any(axis=0)):
            change=np.diff(np.r_[False,mask[:,cell],False].astype(np.int8))
            starts=np.flatnonzero(change==1); stops=np.flatnonzero(change==-1)
            h=hour.indices[hour.indptr[cell]:hour.indptr[cell+1]]
            for no,(a,b) in enumerate(zip(starts,stops),1):
                start=start_date+pd.Timedelta(days=int(a))
                if not 4<=start.month<=9: continue
                k=np.searchsorted(h,a*24)
                if k==len(h) or h[k]>=(b+1)*24: continue
                first=start_date+pd.Timedelta(hours=int(h[k]))
                records.append(dict(id=f'90p20_land_cell{cell}_{year}_event{no}',kind='gridcell',year=year,event_id=-1,cell=cell,
                                    first_cg_time=first,start_time=start,end_time=start_date+pd.Timedelta(days=int(b))-pd.Timedelta(seconds=1),duration=int(b-a),lag_hours=(first-start).total_seconds()/3600))
    a=pd.DataFrame(records)
    a['latitude']=lat[a.cell.to_numpy()//nx]; a['longitude']=lon[a.cell.to_numpy()%nx]
    a['reference_time']=pd.to_datetime(a.first_cg_time).dt.round('3h')
    elapsed=a.lag_hours/(a.duration*24)
    a['phase']=np.select([elapsed<1/3,elapsed<2/3,elapsed<1],['early','middle','late'],default='post_24h')
    a.to_csv(out/'tables/first_cg_anchors.csv',index=False)
    a.groupby(['kind','phase']).size().rename('n').reset_index().to_csv(out/'tables/first_cg_timing.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,kind in zip(axs,['regional','gridcell']):
        counts=a[a.kind==kind].phase.value_counts().reindex(['early','middle','late','post_24h'],fill_value=0)
        ax.bar(counts.index,counts.values); ax.set_title(kind); ax.set_ylabel('Events'); ax.tick_params(axis='x',rotation=20)
    fig.tight_layout(); fig.savefig(out/'figures/first_cg_timing.png',dpi=200); plt.close(fig)
    return a

def prepare_requests(a):
    frames=[]
    for h in range(-12,13,3):
        d=a.copy(); d['relative_hour']=h; d['target_time']=d.reference_time+pd.Timedelta(hours=h); frames.append(d)
    return pd.concat(frames,ignore_index=True)

def run(out, months=None, climatology_dir=None):
    # The baseline climatology (2006-2025 mean per calendar day/3h slot) only
    # depends on the calendar, not on which days are heatwave days, so it is
    # identical across every absolute-threshold run. Defaulting to out/'data/
    # climatology' reproduces the original per-run behaviour exactly; passing a
    # shared directory lets a multi-threshold sweep compute it once and reuse it,
    # instead of silently recomputing the same 20-year baseline for each threshold.
    climatology_dir = Path(climatology_dir) if climatology_dir is not None else out/'data/climatology'
    climatology_dir.mkdir(exist_ok=True, parents=True)
    a=pd.read_parquet(out/'data/confirmed_anchors.parquet')
    if months is not None:
        keep=(a.reference_time-pd.Timedelta(hours=12)).dt.month.isin(months)|(a.reference_time+pd.Timedelta(hours=12)).dt.month.isin(months)
        a=a.loc[keep].copy()
    req=prepare_requests(a)
    e=pd.read_csv(out/'tables/regional_events.csv',parse_dates=['start_time','end_time'])
    e=e[e.start_month.between(4,9)&(e.cg_coverage_fraction==1)].copy().reset_index(drop=True)
    grid=np.load(out/'data/grids.npz'); land=grid['era5_land']; ny,nx=land.shape
    footprint={}
    eligible_ids=set(e.event_id)
    for year in YEARS:
        d=np.load(out/f'data/labels_{year}.npz'); lab=d['labels']; offset=int(d['offset'])
        for i,s in enumerate(ndimage.find_objects(lab),1):
            if i+offset not in eligible_ids: continue
            tt,rr,cc=np.nonzero(lab[s]==i)
            footprint[i+offset]=np.unique((rr+s[1].start)*nx+cc+s[2].start)
    sums={v:np.zeros(len(e)) for v in VARS}; counts={v:np.zeros(len(e),np.int64) for v in VARS}
    for v in CURVEV: req[v+'_climatological_anomaly']=np.nan
    # Include every month intersecting an event or first-CG request; padded
    # climatology uses a leap template and month/day alignment, never shifted DOY.
    monthset=set(req.target_time.dt.month)
    for r in e.itertuples(): monthset.update(pd.date_range(r.start_time.normalize(),r.end_time.normalize()).month)
    if months is not None: monthset &= set(months)
    req=req[req.target_time.dt.month.isin(monthset)].copy()
    for month in sorted(monthset):
        sums={v:np.zeros(len(e)) for v in VARS}; counts={v:np.zeros(len(e),np.int64) for v in VARS}
        spatial_e=e[(e.start_time.dt.month<=month)&(e.end_time.dt.month>=month)]
        spatial_index={eid:i for i,eid in enumerate(spatial_e.event_id)}
        spatial_sum=np.zeros((len(spatial_e),6,65,65),np.float32)
        spatial_n=np.zeros((len(spatial_e),6,65,65),np.uint16)
        offsets=np.arange(-8,8.001,.25)
        print('Environment month',month,'climatology',flush=True)
        cp=climatology_dir/f'month_{month:02d}.npz'
        if not cp.exists():
            nt=calendar.monthrange(2000,month)[1]*8
            total={v:np.zeros((nt,ny,nx),np.float64) for v in VARS}
            number={v:np.zeros((nt,ny,nx),np.uint8) for v in VARS}
            for year in YEARS:
                dates,fields=load_fields(year,month); slots=(dates.day.values-1)*8+dates.hour.values//3
                for v,arr in fields.items():
                    valid=np.isfinite(arr); total[v][slots]+=np.where(valid,arr,0); number[v][slots]+=valid.astype(np.uint8)
                print(' baseline',year,month,flush=True)
            clim={v:np.divide(total[v],number[v],out=np.full(total[v].shape,np.nan),where=number[v]>0).astype(np.float32) for v in VARS}
            np.savez_compressed(cp,**clim,**{v+'_baseline_years':number[v] for v in VARS}); del total,number
        else:
            with np.load(cp) as d: clim={v:d[v] for v in VARS}
        print('Environment month',month,'extract',flush=True)
        for year in YEARS:
            dates,fields=load_fields(year,month); slots=(dates.day.values-1)*8+dates.hour.values//3
            subset=e[(e.year==year)&(e.start_time<=dates[-1])&(e.end_time>=dates[0])]
            qr=req[(req.target_time.dt.year==year)&(req.target_time.dt.month==month)]
            ti=dates.get_indexer(qr.target_time)
            if np.any(ti<0): raise ValueError('Missing first-CG environment dates')
            cells=qr.cell.to_numpy(); rr=cells//nx; cc=cells%nx
            neighbour=[]
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    r=rr+dr; c=cc+dc
                    valid=(r>=0)&(r<ny)&(c>=0)&(c<nx)
                    r=np.clip(r,0,ny-1); c=np.clip(c,0,nx-1)
                    valid&=land[r,c]; neighbour.append((r,c,valid))
            for v,arr in fields.items():
                anomaly=arr-clim[v][slots]
                flat=anomaly.reshape(len(dates),-1)
                for r in subset.itertuples():
                    start=dates.searchsorted(r.start_time); stop=dates.searchsorted(r.end_time,side='right')
                    values=flat[start:stop,footprint[r.event_id]]
                    sums[v][r.Index]+=np.nansum(values); counts[v][r.Index]+=np.isfinite(values).sum()
                    if v in MATCHV:
                        # Preserve each relative-coordinate pixel. Average over
                        # time at that location, then aggregate equally by event.
                        ri=(grid['latitude'][0]-(r.center_latitude+offsets))/.25
                        ci=((r.center_longitude+offsets)-grid['longitude'][0])/.25
                        r0=max(0,int(np.floor(ri.min())));r1=min(ny,int(np.ceil(ri.max()))+1)
                        c0=max(0,int(np.floor(ci.min())));c1=min(nx,int(np.ceil(ci.max()))+1)
                        crop=anomaly[start:stop,r0:r1,c0:c1]
                        valid=np.isfinite(crop); sm=np.where(valid,crop,0).sum(axis=0); nn=valid.sum(axis=0).astype(float)
                        ry,cx=np.meshgrid(ri-r0,ci-c0,indexing='ij')
                        # Six spatial fields are finite; strict bilinear domain
                        # handling leaves out-of-ERA5 pixels missing, not zero.
                        sfield=ndimage.map_coordinates(sm,[ry,cx],order=1,mode='constant',cval=np.nan)
                        nfield=ndimage.map_coordinates(nn,[ry,cx],order=1,mode='constant',cval=0)
                        pos=spatial_index[r.event_id];vi=MATCHV.index(v)
                        spatial_sum[pos,vi]+=np.nan_to_num(sfield)
                        spatial_n[pos,vi]+=np.rint(nfield).astype(np.uint16)
                if v in CURVEV and len(qr):
                    total=np.zeros(len(qr)); n=np.zeros(len(qr))
                    for r,c,valid in neighbour:
                        x=anomaly[ti,r,c]; valid=valid&np.isfinite(x); total+=np.where(valid,x,0); n+=valid
                    req.loc[qr.index,v+'_climatological_anomaly']=np.divide(total,n,out=np.full(len(n),np.nan),where=n>0)
            print(' extraction',year,month,flush=True)
        req[req.target_time.dt.month==month].to_parquet(out/f'data/first_cg_values_m{month:02d}.parquet',index=False)
        np.savez_compressed(out/f'data/event_environment_m{month:02d}.npz',event_ids=e.event_id.values,
                            **{v+'_sum':sums[v] for v in VARS},**{v+'_count':counts[v] for v in VARS})
        np.savez_compressed(out/f'data/spatial_environment_m{month:02d}.npz',event_ids=spatial_e.event_id.values,
                            sums=spatial_sum,counts=spatial_n,offsets=offsets)
    if months is None: finalize(out)

def finalize(out):
    for month in range(3,11):
        for name in [f'first_cg_values_m{month:02d}.parquet',f'event_environment_m{month:02d}.npz',f'spatial_environment_m{month:02d}.npz']:
            if not (out/'data'/name).is_file():
                raise ValueError(f'Missing monthly product: {name}')
    e=pd.read_csv(out/'tables/regional_events.csv')
    e=e[e.start_month.between(4,9)&(e.cg_coverage_fraction==1)].copy().reset_index(drop=True)
    env=e[['event_id','regional_heatwave_event_id','has_lightning']].copy()
    files=sorted((out/'data').glob('event_environment_m*.npz'))
    sums={v:np.zeros(len(e)) for v in VARS}; counts={v:np.zeros(len(e),np.int64) for v in VARS}
    for file in files:
        with np.load(file) as d:
            if not np.array_equal(d['event_ids'],e.event_id): raise ValueError('Monthly event IDs differ')
            for v in VARS: sums[v]+=d[v+'_sum']; counts[v]+=d[v+'_count']
    for v in VARS:
        env[v]=np.divide(sums[v],counts[v],out=np.full(len(e),np.nan),where=counts[v]>0); env[v+'_n']=counts[v]
    env.to_csv(out/'tables/event_environment_anomaly.csv',index=False)
    req=pd.concat([pd.read_parquet(p) for p in sorted((out/'data').glob('first_cg_values_m*.parquet'))],ignore_index=True)
    if req.duplicated(['kind','id','relative_hour']).any(): raise ValueError('Duplicate first-CG rows')
    a=pd.read_parquet(out/'data/confirmed_anchors.parquet')
    if len(req)!=len(a)*9: raise ValueError('Missing first-CG month products')
    req.to_parquet(out/'data/first_cg_event_values.parquet',index=False)
    # Final publication figures are generated separately after all checks.
    dump(out/'data/environment_complete.json',dict(baseline='2006–2025 same month/day/3-hour UTC including target year',variables=VARS,
         regional_event_mean='union footprint land grid cells, all 3-hour slots from start through end; equal cell-time weighting',
         first_cg='nearest 3-hour timestamp, 3x3 ERA5 neighbourhood using land centres only; offsets -12..+12 hours',
         z500_units='geopotential height m',missing_policy='finite cell-time values only; n per variable reported'))

def summarize(out,req,env):
    pairs=pd.read_csv(out/'tables/match_pairs.csv'); indexed=env.set_index('event_id'); rows=[]
    for v in MATCHV:
        a=indexed.loc[pairs.compound_event_id,v].to_numpy(); b=indexed.loc[pairs.control_event_id,v].to_numpy(); valid=np.isfinite(a)&np.isfinite(b); a=a[valid]; b=b[valid]; delta=a-b
        se=stats.sem(delta); critical=stats.t.ppf(.975,len(delta)-1)
        rows.append(dict(variable=v,n_pairs=len(delta),compound_mean=a.mean(),control_mean=b.mean(),paired_difference=delta.mean(),
                         ci95_lower=delta.mean()-critical*se,ci95_upper=delta.mean()+critical*se,p=stats.ttest_1samp(delta,0).pvalue))
    comp=pd.DataFrame(rows); comp['p_bh']=bh(comp.p); comp.to_csv(out/'tables/matched_environment_comparison.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(12,7))
    for ax,row in zip(axs.flat,comp.itertuples()):
        ax.bar(['Control','Compound'],[row.control_mean,row.compound_mean],color=['steelblue','tomato']); ax.set_title(f'{row.variable}, BH p={row.p_bh:.3g}'); ax.axhline(0,c='grey',lw=.6)
    fig.suptitle('Land regional events: matched calendar-day/hour anomalies'); fig.tight_layout(); fig.savefig(out/'figures/matched_environment.png',dpi=200); plt.close(fig)
    curves=[]; adjacent=[]
    for kind in ['regional','gridcell']:
        data=req[req.kind==kind]
        fig,axs=plt.subplots(2,3,figsize=(12,7))
        for ax,v in zip(axs.flat,CURVEV):
            col=v+'_climatological_anomaly'
            means=data.groupby('relative_hour')[col].agg(['mean','count','std'])
            # Descriptive spread shown; uncertainty of changes assessed with paired events below.
            ax.plot(means.index,means['mean'],'o-'); ax.axhline(0,c='grey',lw=.7); ax.axvline(0,c='grey',ls=':'); ax.set_title(v); ax.set_xlabel('Hours from reference first CG')
            for t,r in means.iterrows(): curves.append(dict(kind=kind,variable=v,relative_hour=t,mean=r['mean'],n=r['count'],sd=r['std']))
            wide=data.pivot(index='id',columns='relative_hour',values=col)
            for prev,current in zip(range(-12,12,3),range(-9,13,3)):
                d=(wide[current]-wide[prev]).dropna(); se=stats.sem(d); critical=stats.t.ppf(.975,len(d)-1)
                adjacent.append(dict(kind=kind,variable=v,previous_hour=prev,current_hour=current,n=len(d),difference=d.mean(),ci95_lower=d.mean()-critical*se,ci95_upper=d.mean()+critical*se,p=stats.ttest_1samp(d,0).pvalue))
        fig.suptitle(f'Land {kind} events: first-CG environmental anomalies'); fig.tight_layout(); fig.savefig(out/f'figures/first_cg_{kind}_curves.png',dpi=200); plt.close(fig)
    pd.DataFrame(curves).to_csv(out/'tables/first_cg_curve_summary.csv',index=False)
    changes=pd.DataFrame(adjacent); changes['p_bh']=changes.groupby(['kind','variable']).p.transform(lambda s:bh(s.to_numpy()))
    changes.to_csv(out/'tables/first_cg_adjacent_changes.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--months',nargs='+',type=int); p.add_argument('--finalize',action='store_true')
    p.add_argument('--climatology-dir',type=Path,default=None,help='Shared cache dir; defaults to <out>/data/climatology.')
    args=p.parse_args(); finalize(args.out) if args.finalize else run(args.out,args.months,climatology_dir=args.climatology_dir)
