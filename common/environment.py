"""ERA5 environmental variables: names, labels, units and the monthly reader."""
import numpy as np
import pandas as pd
import xarray as xr

from common.config import ROOT

VARS=['q850','qu850','qv850','t2m-d2m','tcwv','z500','cape','cin','omega700']
MATCHV=VARS[:6]
CURVEV=['cape','cin','q850','tcwv','t2m-d2m','omega700']
# Display order of the six matched variables in figures and tables.
ORDER=['q850','tcwv','t2m-d2m','qv850','qu850','z500']
LABELS={'q850':'850 hPa specific humidity','tcwv':'Total column water vapour','t2m-d2m':'2 m dewpoint depression',
        'qv850':'850 hPa meridional moisture flux','qu850':'850 hPa zonal moisture flux','z500':'500 hPa geopotential height',
        'cape':'CAPE','cin':'CIN','omega700':'700 hPa omega'}
UNITS={'q850':r'kg kg$^{-1}$','tcwv':r'kg m$^{-2}$','t2m-d2m':'K','qv850':r'kg kg$^{-1}$ m s$^{-1}$','qu850':r'kg kg$^{-1}$ m s$^{-1}$','z500':'m','cape':r'J kg$^{-1}$','cin':r'J kg$^{-1}$','omega700':r'Pa s$^{-1}$'}


def load_fields(year,month):
    """3-hourly fields for one month, keyed by VARS; z500 in geopotential metres."""
    instant=ROOT/f'data/Era5/{year}/instant/{year}_{month:02d}.nc'
    pressure=ROOT/f'data/Era5/{year}/pressure/{year}_{month:02d}.nc'
    single=ROOT/f'processed/era5_3hourly/single_level/{year}.nc'
    with xr.open_dataset(instant) as ds:
        dates=pd.DatetimeIndex(ds.valid_time.values)
        dep=(ds.t2m-ds.d2m).values.astype(np.float32)
    with xr.open_dataset(pressure) as ds:
        q=ds.q.sel(pressure_level=850).values.astype(np.float32)
        qu=q*ds.u.sel(pressure_level=850).values
        qv=q*ds.v.sel(pressure_level=850).values
        z=ds.z.sel(pressure_level=500).values.astype(np.float32)/9.80665
        w=ds.w.sel(pressure_level=700).values.astype(np.float32)
    with xr.open_dataset(single) as ds:
        ds=ds.sel(time=slice(dates[0],dates[-1]))
        if not np.array_equal(ds.time.values,dates.values): raise ValueError('single-level time alignment failed')
        tcwv=ds.tcwv.values.astype(np.float32); cape=ds.cape.values.astype(np.float32); cin=ds.cin.values.astype(np.float32)
    return dates,dict(zip(VARS,[q,qu,qv,dep,tcwv,z,cape,cin,w]))
