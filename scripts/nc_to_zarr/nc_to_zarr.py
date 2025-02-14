import sys
import pickle
from pathlib import Path

import xarray as xr

basepath = Path('/gws/nopw/j04/hrcm/mmuetz/Lorenzo_u-cu087/')

paths = sorted(basepath.glob('pe_T/healpix/experimental_20200101T0000Z_pe???.hpz0-10.nc'))
# paths = paths[:5]
paths.extend(sorted(basepath.glob('pe_T/healpix/experimental_20200101T0000Z_pe????.hpz0-10.nc')))

outpath = Path('/gws/nopw/j04/hrcm/mmuetz/Lorenzo_u-cu087/pe_T/healpix/experimental_20200101T0000Z_pe.hpz0-10.full1.zarr')

if __name__ == '__main__':
    pkl_path = Path('ds.pkl')
    path = paths[int(sys.argv[1])]
    print(path)

    if not pkl_path.exists():
        raise Exception(f'pkl_path does not exist: {pkl_path} - create first')
    else:
        print('Loading from pickle')
        ds = pickle.load(pkl_path.open('rb'))
    ds_to_copy = xr.open_dataset(path)

    for t in ds_to_copy.time.values:
        print(t)
        ds.sel(time=slice(t, t)).drop_vars(['forecast_period', 'forecast_reference_time']).to_zarr(outpath, region='auto')


