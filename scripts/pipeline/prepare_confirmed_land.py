"""Apply confirmed intensity centres and non-HW CG onset definitions."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage,sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.analysis_inputs import input_path
from common.config import YEARS, heatwave_dir
from common.io import dump

def prepare(out,method='90p20'):
    (out/'first_cg/data').mkdir(parents=True, exist_ok=True)
    (out/'first_cg/tables').mkdir(parents=True, exist_ok=True)
    hw_dir = heatwave_dir(method)
    grid=np.load(input_path(out, 'grids.npz')); lat=grid['latitude']; lon=grid['longitude']; nx=len(lon)
    e=pd.read_csv(out/'events/regional_events.csv',parse_dates=['start_time','end_time','first_cg_time'])
    qualified=set(e.loc[e.has_lightning&(e.cg_coverage_fraction==1),'event_id'])
    centers={}; records=[]; checks=[]
    for year in YEARS:
        print('Confirmed preparation',year,flush=True)
        ld=np.load(input_path(out, f'labels_{year}.npz')); lab=ld['labels']; offset=int(ld['offset']); ndays=len(lab)
        dates=pd.date_range(f'{year}-01-01',periods=ndays)
        with xr.open_dataset(hw_dir/f'wetbulb_heatwave_{year}.nc') as d:
            ex=(d.tw_daily_max-d.heatwave_threshold).values
        # Heatwave-related window of every regional event: its footprint from the
        # first day through the day after the last, the span in which the
        # compound rule counts CG. Non-HW onsets inside it are not controls.
        event_window=np.zeros((ndays,lab.shape[1]*nx),bool)
        for i,s in enumerate(ndimage.find_objects(lab),1):
            t,r,c=np.nonzero(lab[s]==i); t+=s[0].start; r+=s[1].start; c+=s[2].start
            w=np.maximum(ex[t,r,c],0).astype(float)
            if not w.sum()>0: raise ValueError('No positive intensity in event')
            centers[offset+i]=(np.average(lat[r],weights=w),np.average(lon[c],weights=w))
            event_window[t.min():min(t.max()+2,ndays),np.unique(r*nx+c)]=True
        hour=sparse.load_npz(input_path(out, f'cg_hourly_era5_{year}.npz')).tocsc()
        flat=lab.reshape(ndays,-1)
        month=np.repeat(dates.month.values,24)
        ncontrol=0; ncompound=0
        for cell in np.flatnonzero(np.diff(hour.indptr)>0):
            hours=hour.indices[hour.indptr[cell]:hour.indptr[cell+1]]
            # Integer hourly fields: previous 12 bins h-12 through h-1 must be empty.
            gap=np.r_[hours[0]+1,np.diff(hours)]
            candidates=hours[(gap>12)&(hours>=12)&(month[hours]>=4)&(month[hours]<=9)&(flat[hours//24,cell]==0)]
            previous=-10000
            for h in candidates:
                if h-previous<24: continue
                previous=int(h)
                if event_window[h//24,cell]: continue
                t0=dates[0]+pd.Timedelta(hours=int(h))
                records.append(dict(id=f'nonhw_onset_{year}_{cell}_{h}',kind='nonhw_onset',cell=cell,event_id=-1,
                                    first_cg_time=t0,reference_time=t0.round('3h'),year=year))
                ncontrol+=1
            active=flat[:,cell]>0
            change=np.diff(np.r_[False,active,False].astype(np.int8))
            for a,b in zip(np.flatnonzero(change==1),np.flatnonzero(change==-1)):
                eid=int(flat[a,cell])+offset
                if eid not in qualified: continue
                k=np.searchsorted(hours,a*24)
                if k>=len(hours) or hours[k]>=(b+1)*24: continue
                h=int(hours[k]); t0=dates[0]+pd.Timedelta(hours=h)
                if not 4<=t0.month<=9: continue
                records.append(dict(id=f'compound_{year}_{cell}_{a}',kind='compound',cell=cell,event_id=eid,
                                    first_cg_time=t0,reference_time=t0.round('3h'),year=year))
                ncompound+=1
        checks.append(dict(year=year,compound_gridcell_first_cg=ncompound,nonhw_onsets=ncontrol))
    if 'area_center_latitude_previous' not in e:  # keep the original area-weighted centres on reruns
        e['area_center_latitude_previous']=e.center_latitude
        e['area_center_longitude_previous']=e.center_longitude
    e['center_latitude']=[centers[i][0] for i in e.event_id]
    e['center_longitude']=[centers[i][1] for i in e.event_id]
    e.to_csv(out/'events/regional_events.csv',index=False)
    a=pd.DataFrame(records); a['latitude']=lat[a.cell.values//nx];a['longitude']=lon[a.cell.values%nx]
    assert a.first_cg_time.dt.month.between(4,9).all()
    controls=a[a.kind=='nonhw_onset'].sort_values(['cell','first_cg_time'])
    gaps=controls.groupby('cell').first_cg_time.diff().dt.total_seconds()/3600
    assert gaps.dropna().min()>=24
    assert np.all(grid['era5_land'].ravel()[a.cell])
    a.to_parquet(out/'first_cg/data/confirmed_anchors.parquet',index=False)
    pd.DataFrame(checks).to_csv(out/'first_cg/tables/confirmed_anchor_counts.csv',index=False)
    overlap=a.groupby(['cell','first_cg_time']).kind.nunique()
    dump(out/'first_cg/confirmed_definition.json',dict(center='sum over event land cell-days of positive (Twmax-combined threshold) times coordinate / sum of weights; no pre-averaging of environment',
      compound='grid-cell first CG during HW spell or subsequent 24 h, belonging to a fully CG-covered regional event with CG-cell fraction >=0.10',
      nonhw_onset='ERA5 land cell with CG at t0, not HW that day, no CG in previous 12 hourly bins, greedily retain onsets at least 24 h apart per cell; '
                  'then drop onsets inside any regional event footprint between its first day and the day after its last',
      reference='actual hourly t0 selected by April–September; nearest ERA5 3-hour reference retained separately',
      overlap_between_groups_at_same_cell_time=int((overlap>1).sum()),overlap_note='expected 0: compound first CGs lie inside their event window, which non-HW onsets exclude'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',default='90p20')
    args=p.parse_args();prepare(args.out,method=args.method)
