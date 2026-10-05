"""Apply confirmed intensity centres and non-HW CG onset definitions."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage,sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import YEARS, heatwave_dir
from common.io import dump

def prepare(out,method='90p20'):
    hw_dir = heatwave_dir(method)
    grid=np.load(out/'data/grids.npz'); lat=grid['latitude']; lon=grid['longitude']; nx=len(lon)
    e=pd.read_csv(out/'tables/regional_events.csv',parse_dates=['start_time','end_time','first_cg_time'])
    qualified=set(e.loc[e.has_lightning&(e.cg_coverage_fraction==1),'event_id'])
    centers={}; records=[]; checks=[]
    for year in YEARS:
        print('Confirmed preparation',year,flush=True)
        ld=np.load(out/f'data/labels_{year}.npz'); lab=ld['labels']; offset=int(ld['offset']); ndays=len(lab)
        dates=pd.date_range(f'{year}-01-01',periods=ndays)
        with xr.open_dataset(hw_dir/f'wetbulb_heatwave_{year}.nc') as d:
            ex=(d.tw_daily_max-d.heatwave_threshold).values
        for i,s in enumerate(ndimage.find_objects(lab),1):
            t,r,c=np.nonzero(lab[s]==i); t+=s[0].start; r+=s[1].start; c+=s[2].start
            w=np.maximum(ex[t,r,c],0).astype(float)
            if not w.sum()>0: raise ValueError('No positive intensity in event')
            centers[offset+i]=(np.average(lat[r],weights=w),np.average(lon[c],weights=w))
        hour=sparse.load_npz(out/f'data/cg_hourly_era5_{year}.npz').tocsc()
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
                previous=int(h); t0=dates[0]+pd.Timedelta(hours=int(h))
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
    e['area_center_latitude_previous']=e.center_latitude
    e['area_center_longitude_previous']=e.center_longitude
    e['center_latitude']=[centers[i][0] for i in e.event_id]
    e['center_longitude']=[centers[i][1] for i in e.event_id]
    e.to_csv(out/'tables/regional_events.csv',index=False)
    a=pd.DataFrame(records); a['latitude']=lat[a.cell.values//nx];a['longitude']=lon[a.cell.values%nx]
    assert a.first_cg_time.dt.month.between(4,9).all()
    controls=a[a.kind=='nonhw_onset'].sort_values(['cell','first_cg_time'])
    gaps=controls.groupby('cell').first_cg_time.diff().dt.total_seconds()/3600
    assert gaps.dropna().min()>=24
    assert np.all(grid['era5_land'].ravel()[a.cell])
    a.to_parquet(out/'data/confirmed_anchors.parquet',index=False)
    pd.DataFrame(checks).to_csv(out/'checks/confirmed_anchor_counts.csv',index=False)
    overlap=a.groupby(['cell','first_cg_time']).kind.nunique()
    dump(out/'checks/confirmed_definition.json',dict(center='sum over event land cell-days of positive (Twmax-combined threshold) times coordinate / sum of weights; no pre-averaging of environment',
      compound='grid-cell first CG during HW spell or subsequent 24 h, belonging to a fully CG-covered regional event with CG-cell fraction >=0.10',
      nonhw_onset='ERA5 land cell with CG at t0, not HW that day, no CG in previous 12 hourly bins, greedily retain onsets at least 24 h apart per cell',
      reference='actual hourly t0 selected by April–September; nearest ERA5 3-hour reference retained separately',
      overlap_between_groups_at_same_cell_time=int((overlap>1).sum()),overlap_note='possible when compound first CG is in the allowed post-HW 24 h; neither definition silently changed'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',default='90p20')
    args=p.parse_args();prepare(args.out,method=args.method)
