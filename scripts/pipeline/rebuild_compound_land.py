#!/usr/bin/env python3
"""Rebuild land-only regional heatwaves, daily samples, and matched catalogues.

Reusable arrays live in processed/analysis_inputs; results live in the run directory.
Original data and results are read-only inputs. Subcommands are restartable.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage, sparse
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.analysis_inputs import input_path, register_inputs
from common.cache import SHARED_ANALYSIS_INPUTS_DIR, save_array_cache, save_sparse_cache
from common.config import ROOT, CG, YEARS, heatwave_dir
from common.geo import nearest_indices, land_at
from common.io import dump

def event_structure():
    s=np.zeros((3,3,3),bool)
    s[1,:,:]=True  # eight spatial neighbours on the same day
    s[:,1,1]=True  # same grid cell on adjacent days
    return s

def rebuild(out, method='90p20'):
    # method selects the absolute wet-bulb threshold directory/event-id prefix/old-catalogue
    # baseline (all under the 90p+<threshold> naming convention).
    register_inputs(out, method)
    hw_dir = heatwave_dir(method)
    for name in ['checks', 'events', 'occurrence/tables', 'occurrence/figures', 'occurrence/data', 'matching/baseline', 'intensity/tables', 'intensity/data', 'environment/tables', 'environment/data', 'first_cg/tables', 'first_cg/data']:
        (out/name).mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(hw_dir/'wetbulb_heatwave_2006.nc') as d:
        lat, lon = d.latitude.values, d.longitude.values
    yy, xx = np.meshgrid(lat, lon, indexing='ij')
    land = land_at(yy, xx)
    with xr.open_dataset(CG/'2006/euclid_flash_cg_2006_04.nc') as d:
        clat, clon = d.latitude.values, d.longitude.values
    cy, cx = np.meshgrid(clat, clon, indexing='ij')
    cg_land = land_at(cy, cx) & (cy>=lat.min()) & (cy<=lat.max()) & (cx>=lon.min()) & (cx<=lon.max())
    ilat, ilon = nearest_indices(lat, cy.ravel()), nearest_indices(lon, cx.ravel())
    mapped = ilat*len(lon)+ilon
    # ERA5 events are defined only on land. A CG cell is eligible for an event
    # only if both the CG centre and its mapped ERA5 centre pass the land mask.
    event_cg_mask = cg_land.ravel() & land.ravel()[mapped]
    cg_ids = np.flatnonzero(cg_land.ravel())
    event_ids = np.flatnonzero(event_cg_mask)
    observed_era=np.zeros(land.size,bool)
    observed_era[mapped[event_ids]]=True
    grid_path = input_path(out, 'grids.npz')
    grid_values = dict(latitude=lat, longitude=lon, era5_land=land, cg_latitude=clat,
                       cg_longitude=clon, cg_land=cg_land, cg_ids=cg_ids,
                       cg_to_era=mapped, event_cg_mask=event_cg_mask)
    if grid_path.exists():
        with np.load(grid_path) as cached:
            if any(not np.array_equal(cached[k], v) for k, v in grid_values.items()):
                raise ValueError('Shared grid differs; select a fresh ANALYSIS_INPUTS_ROOT')
    else:
        save_array_cache(grid_path, **grid_values)
    old = pd.read_csv(ROOT/f'processed/compound/euclid_cg/regional_cg_heatwave_lightning_events_{method}_2006_2025.csv')
    oldcheck, catalogue, audit = [], [], []
    offset = 0
    ny, nx = yy.shape
    for year in YEARS:
        print('Rebuild', year, flush=True)
        with xr.open_dataset(hw_dir/f'wetbulb_heatwave_{year}.nc') as d:
            times = pd.DatetimeIndex(d.time.values)
            raw = d.heatwave_day.values.astype(bool)
            tw = d.tw_daily_max.values
            threshold = d.heatwave_threshold.values
        oldlabels, oldn = ndimage.label(raw, event_structure())
        o = old[old.start_time.str.startswith(str(year))]
        exact = np.array_equal(np.sort(np.bincount(oldlabels.ravel())[1:]), np.sort(o.n_gridcell_days))
        oldcheck.append(dict(year=year, old_events=len(o), rebuilt_unmasked_events=oldn, size_multiset_exact=exact))
        if not exact:
            raise RuntimeError(f'Unmasked regional connectivity does not reproduce old catalogue in {year}')
        del oldlabels
        labels, n = ndimage.label(raw & land[None], event_structure())
        assert not np.any(labels[:,~land])
        if not input_path(out, f'labels_{year}.npz').exists():
            save_array_cache(input_path(out, f'labels_{year}.npz'), labels=labels, offset=offset, dates=times.values)
        hour_path = input_path(out, f'cg_hourly_era5_{year}.npz')
        cg_daily_path = SHARED_ANALYSIS_INPUTS_DIR / f'cg_daily_{year}.npz'
        if hour_path.exists() and cg_daily_path.exists():
            hour = sparse.load_npz(hour_path).tocsc()
            with np.load(cg_daily_path) as cached:
                daily_cg = cached['cg']
            if daily_cg.shape != (len(times), len(cg_ids)):
                raise ValueError(f'Cached daily CG dimensions differ in {year}')
        else:
            rows, cols, vals = [], [], []
            daily_cg = np.zeros((len(times),len(cg_ids)),np.uint8)
            for month in range(1,13):
                p = CG/str(year)/f'euclid_flash_cg_{year}_{month:02d}.nc'
                with xr.open_dataset(p) as d:
                    tt = pd.DatetimeIndex(d.valid_time.values)
                    expected = pd.date_range(f'{year}-{month:02d}-01',periods=len(tt),freq='h')
                    if not tt.equals(expected) or len(tt)!=tt.days_in_month[0]*24:
                        raise ValueError(f'Incomplete hourly calendar: {p}')
                    a = d.flash_count.values.reshape(len(tt),-1)
                    if not np.all(np.isfinite(a)) or np.any(a<0):
                        raise ValueError(f'Invalid CG counts: {p}')
                    sub = a[:,event_ids]
                    h,c = np.nonzero(sub)
                    hour0 = int((tt[0]-times[0]).total_seconds()/3600)
                    rows.append((h+hour0).astype(np.int32)); cols.append(mapped[event_ids[c]].astype(np.int32))
                    vals.append(sub[h,c].astype(np.int32))
                    start = tt[0].dayofyear-1
                    daily_cg[start:start+len(tt)//24] = (a[:,cg_ids]>0).reshape(-1,24,len(cg_ids)).any(axis=1)
            hour = sparse.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(len(times)*24,ny*nx)).tocsc()
            if not hour_path.exists():
                save_sparse_cache(hour_path, hour)
            save_array_cache(cg_daily_path, cg=daily_cg)
        if not input_path(out, f'daily_{year}.npz').exists():
            save_array_cache(input_path(out, f'daily_{year}.npz'), cg=daily_cg,
                                hw=raw.reshape(len(times),-1)[:,mapped[cg_ids]],
                                excess=(tw-threshold).reshape(len(times),-1)[:,mapped[cg_ids]],
                                event=labels.reshape(len(times),-1)[:,mapped[cg_ids]], dates=times.values,
                                offset=offset)
        objects = ndimage.find_objects(labels)
        for label_id, slices in enumerate(objects,1):
            t,r,c = np.nonzero(labels[slices]==label_id)
            t=t+slices[0].start; r=r+slices[1].start; c=c+slices[2].start
            footprint = np.unique(r*nx+c)
            fr,fc = np.unravel_index(footprint,(ny,nx))
            start, stop = int(t.min()),int(t.max())
            # Old catalogue includes CG up to 24 hours after the final HW day.
            hits = hour[start*24:min((stop+2)*24,hour.shape[0]),footprint].tocoo()
            hitcells = np.unique(hits.col)
            required = max(1,int(np.ceil(.1*len(footprint))))
            compound = len(hitcells)>=required
            first = pd.NaT; anchor = -1
            if hits.nnz:
                first_hour = int(hits.row.min())
                first = times[start]+pd.Timedelta(hours=first_hour)
                use = hits.row==first_hour
                options = footprint[hits.col[use]]
                # largest CG count; deterministic lowest cell ID on ties
                order = np.lexsort((options,-hits.data[use]))
                anchor = int(options[order[0]])
            area = (6371.0**2)*np.deg2rad(.25)*(np.sin(np.deg2rad(lat[fr]+.125))-np.sin(np.deg2rad(lat[fr]-.125)))
            catalogue.append(dict(event_id=offset+label_id,regional_heatwave_event_id=f'{method}_land_event{offset+label_id:06d}',
                                  year=year,label_id=label_id,start_time=times[start],end_time=times[stop]+pd.Timedelta(hours=24)-pd.Timedelta(seconds=1),
                                  start_day=start,end_day=stop,duration=stop-start+1,start_month=times[start].month,
                                  n_gridcell_days=len(t),n_gridcells=len(footprint),area_km2=area.sum(),
                                  cg_coverage_fraction=float(observed_era[footprint].mean()),
                                  center_latitude=np.average(lat[fr],weights=area),center_longitude=np.average(lon[fc],weights=area),
                                  min_latitude=lat[fr].min(),max_latitude=lat[fr].max(),min_longitude=lon[fc].min(),max_longitude=lon[fc].max(),
                                  intensity=float(np.mean(tw[t,r,c]-threshold[t,r,c])),mean_tw=float(np.mean(tw[t,r,c])),max_tw=float(np.max(tw[t,r,c])),
                                  n_lightning_event_gridcells=len(hitcells),n_required_lightning_event_gridcells=required,
                                  lightning_event_gridcell_fraction=len(hitcells)/len(footprint),has_lightning=compound,
                                  any_cg=bool(hits.nnz),n_lightning=hits.nnz,total_flash_count=int(hits.data.sum()),
                                  first_cg_time=first,anchor_cell=anchor,lag_hours=(first-times[start]).total_seconds()/3600 if pd.notna(first) else np.nan))
        audit.append(dict(year=year,unmasked_events=oldn,land_events=n,unmasked_hw_gridcell_days=int(raw.sum()),land_hw_gridcell_days=int((labels>0).sum()),
                          cg_land_cells=len(cg_ids),cg_cells_mapping_to_land_era5=int(event_cg_mask.sum())))
        offset+=n
        pd.DataFrame(catalogue).to_csv(out/'events/regional_events.csv',index=False)
        pd.DataFrame(oldcheck).to_csv(out/'checks/unmasked_reproduction.csv',index=False)
        pd.DataFrame(audit).to_csv(out/'checks/land_rebuild_audit.csv',index=False)
    dump(out/'checks/rebuild_complete.json',dict(events=offset))

def probability(out):
    (out/'occurrence/tables').mkdir(parents=True, exist_ok=True)
    totals=np.zeros((2,2),np.int64)
    byyear=[]
    for year in YEARS:
        with np.load(input_path(out, f'daily_{year}.npz')) as d:
            dates=pd.DatetimeIndex(d['dates']); sel=(dates.month>=4)&(dates.month<=9)
            h=d['hw'][sel].astype(bool); y=d['cg'][sel].astype(bool)
            c=np.array([[np.sum(~h&~y),np.sum(~h&y)],[np.sum(h&~y),np.sum(h&y)]])
            totals+=c
            byyear.append(dict(year=year,nonhw_n=int(c[0].sum()),hw_n=int(c[1].sum()),nonhw_cg=int(c[0,1]),hw_cg=int(c[1,1])))
    p=totals[:,1]/totals.sum(axis=1); pa=totals[:,1].sum()/totals.sum()
    pd.DataFrame([dict(p_nonhw=p[0],p_hw=p[1],p_all=pa,hw_vs_nonhw=p[1]/p[0],hw_vs_all=p[1]/pa,
                       nonhw_n=totals[0].sum(),hw_n=totals[1].sum(),nonhw_cg=totals[0,1],hw_cg=totals[1,1])]).to_csv(out/'occurrence/tables/occurrence_probability.csv',index=False)
    pd.DataFrame(byyear).to_csv(out/'occurrence/tables/occurrence_by_year.csv',index=False)
    print('Probabilities',p,pa,flush=True)

def coverage(out):
    e=pd.read_csv(out/'events/regional_events.csv')
    grid=np.load(input_path(out, 'grids.npz')); mask=grid['event_cg_mask']; mapping=grid['cg_to_era']
    supported=np.zeros(grid['era5_land'].size,bool); supported[mapping[mask]]=True
    nx=len(grid['longitude']); fractions={}
    for year in YEARS:
        d=np.load(input_path(out, f'labels_{year}.npz')); lab=d['labels']; offset=int(d['offset'])
        for i,s in enumerate(ndimage.find_objects(lab),1):
            t,r,c=np.nonzero(lab[s]==i); foot=np.unique((r+s[1].start)*nx+c+s[2].start)
            fractions[i+offset]=supported[foot].mean()
    e['cg_coverage_fraction']=e.event_id.map(fractions)
    e['cg_classification']=np.where(e.cg_coverage_fraction<1,'incomplete_coverage',np.where(e.has_lightning,'compound','noncompound'))
    e.to_csv(out/'events/regional_events.csv',index=False)
    e.groupby('cg_classification').size().rename('n').to_csv(out/'checks/cg_coverage.csv')

def match(out):
    e=pd.read_csv(out/'events/regional_events.csv')
    e=e[e.start_month.between(4,9)].copy()
    n_before_coverage=len(e)
    e=e[e.cg_coverage_fraction==1].copy()
    cov=['center_latitude','center_longitude','duration','intensity']
    x=e[cov].to_numpy(); vi=np.linalg.pinv(np.cov(x,rowvar=False))
    treatment=e[e.has_lightning].copy(); controls=e[~e.has_lightning].copy()
    edges=[]
    for month in range(4,10):
        a=treatment[treatment.start_month==month]; b=controls[controls.start_month==month]
        dist=cdist(a[cov],b[cov],metric='mahalanobis',VI=vi)
        i,j=np.nonzero(dist<=.75)
        edges.extend(zip(dist[i,j].tolist(),a.event_id.values[i].tolist(),b.event_id.values[j].tolist()))
    used_a,used_b=set(),set(); pairs=[]
    for dist,a,b in sorted(edges):
        if a in used_a or b in used_b: continue
        used_a.add(a); used_b.add(b); pairs.append(dict(match_set_id=len(pairs)+1,compound_event_id=a,control_event_id=b,mahalanobis_distance=dist))
    pairs=pd.DataFrame(pairs); pairs.to_csv(out/'matching/baseline/match_pairs.csv',index=False)
    selected=[]
    for role,col in [('compound','compound_event_id'),('noncompound','control_event_id')]:
        a=pairs[['match_set_id',col,'mahalanobis_distance']].merge(e,left_on=col,right_on='event_id'); a['match_role']=role; selected.append(a.drop(columns=col))
    pd.concat(selected).sort_values(['match_set_id','match_role']).to_csv(out/'matching/baseline/matched_events.csv',index=False)
    before_scale=np.sqrt((treatment[cov].var()+controls[cov].var())/2)
    aa=e.set_index('event_id').loc[pairs.compound_event_id]; bb=e.set_index('event_id').loc[pairs.control_event_id]
    before=(treatment[cov].mean()-controls[cov].mean())/before_scale
    after=(aa[cov].mean()-bb[cov].mean())/before_scale
    pd.DataFrame(dict(variable=cov,smd_before=before.values,smd_after=after.values)).to_csv(out/'matching/baseline/match_balance.csv',index=False)
    dump(out/'matching/baseline/matching_summary.json',dict(compound_available=len(treatment),controls_available=len(controls),matched_pairs=len(pairs),compound_retention=len(pairs)/len(treatment),
         covariates=cov,exact_month=True,caliper=.75,replacement=False,algorithm='global ascending Mahalanobis distance greedy edge selection; pooled covariance of eligible events',
         warmseason_events_before_coverage=n_before_coverage,excluded_incomplete_cg_coverage=n_before_coverage-len(e),
         original_matching_algorithm='original source absent; parameter values recovered, exact ordering not recoverable'))
    print('Matched pairs',len(pairs),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['rebuild','coverage','probability','match']); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--method',default='90p20',help="Absolute wet-bulb threshold label, e.g. 90p18p5, 90p20, 90p22. Only used by the 'rebuild' stage; default reproduces the original 90p20 run exactly.")
    args=parser.parse_args()
    if args.stage=='rebuild':
        rebuild(args.out, method=args.method)
    else:
        globals()[args.stage](args.out)
