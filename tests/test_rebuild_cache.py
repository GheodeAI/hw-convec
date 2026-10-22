import calendar
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr
from scripts.pipeline import rebuild_compound_land as rebuild


class RebuildCacheTests(unittest.TestCase):
    def test_cached_cg_reproduces_raw_rebuild_without_rereading_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); hw = root/'hw'; hw.mkdir()
            cg = root/'cg'; (cg/'2006').mkdir(parents=True)
            shared = root/'shared'; shared.mkdir()
            method = root/'method'; method.mkdir()
            dates = pd.date_range('2006-01-01', '2006-12-31')
            flags = np.zeros((365,1,1), dtype=np.uint8); flags[:3] = 1
            xr.Dataset({'heatwave_day':(('time','latitude','longitude'),flags),
                        'tw_daily_max':(('time','latitude','longitude'),np.full(flags.shape,22.0)),
                        'heatwave_threshold':(('time','latitude','longitude'),np.full(flags.shape,20.0))},
                       coords={'time':dates,'latitude':[50.0],'longitude':[0.0]}).to_netcdf(hw/'wetbulb_heatwave_2006.nc')
            for month in range(1,13):
                times = pd.date_range(f'2006-{month:02d}-01', periods=calendar.monthrange(2006,month)[1]*24, freq='h')
                counts = np.zeros((len(times),1,1),dtype=np.int32)
                if month == 1: counts[[1,50],0,0] = [2,1]
                xr.Dataset({'flash_count':(('valid_time','latitude','longitude'),counts)},
                           coords={'valid_time':times,'latitude':[50.0],'longitude':[0.0]}).to_netcdf(cg/f'2006/euclid_flash_cg_2006_{month:02d}.nc')
            old = root/'processed/compound/euclid_cg'; old.mkdir(parents=True)
            pd.DataFrame({'start_time':['2006-01-01'], 'n_gridcell_days':[3]}).to_csv(old/'regional_cg_heatwave_lightning_events_90p20_2006_2025.csv',index=False)
            def location(run,name):
                return (shared if name == 'grids.npz' or name.startswith('cg_hourly_') else method)/name
            patches = [patch.object(rebuild,'ROOT',root), patch.object(rebuild,'CG',cg),
                       patch.object(rebuild,'YEARS',[2006]), patch.object(rebuild,'heatwave_dir',return_value=hw),
                       patch.object(rebuild,'register_inputs'), patch.object(rebuild,'input_path',side_effect=location),
                       patch.object(rebuild,'SHARED_ANALYSIS_INPUTS_DIR',shared),
                       patch.object(rebuild,'land_at',side_effect=lambda lat,lon:np.ones(np.shape(lat),dtype=bool))]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for context in patches:stack.enter_context(context)
                first = root/'first'; rebuild.rebuild(first)
                hashes = {p:hashlib.sha256(p.read_bytes()).digest() for folder in [shared,method] for p in folder.glob('*.npz')}
                # Only the April grid source is needed once observations are cached.
                for p in (cg/'2006').glob('*.nc'):
                    if '_04.nc' not in p.name:p.unlink()
                second = root/'second'; rebuild.rebuild(second)
                self.assertTrue(all(hashlib.sha256(p.read_bytes()).digest()==value for p,value in hashes.items()))
                for name in ['events/regional_events.csv','checks/land_rebuild_audit.csv','checks/unmasked_reproduction.csv']:
                    pd.testing.assert_frame_equal(pd.read_csv(first/name),pd.read_csv(second/name))
                events = pd.read_csv(second/'events/regional_events.csv')
                self.assertEqual(int(events.total_flash_count.iloc[0]),3)
                self.assertTrue(events.has_lightning.iloc[0])


if __name__ == '__main__':
    unittest.main()
