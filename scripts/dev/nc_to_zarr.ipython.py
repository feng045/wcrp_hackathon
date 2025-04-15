# coding: utf-8
import sys

import xarray as xr

if __name__ == '__main__':
    if len(sys.argv) > 1:
        inpath = sys.argv[1]
        outpath = sys.argv[2]
    else:
        inpath = '/gws/nopw/j04/hrcm/mmuetz/Lorenzo_u-cu087/OLR/healpix/experimental_20200101T0000Z_pa*.hpz0-10.nc'
        outpath = '/gws/nopw/j04/hrcm/mmuetz/Lorenzo_u-cu087/OLR/healpix/experimental_20200101T0000Z_pa.combined.full.zarr/'
    print(sys.argv)

    ds = xr.open_mfdataset(inpath, concat_dim='time', combine='nested', data_vars="minimal", coords="minimal", compat="override", parallel=True)
    # ds = xr.open_mfdataset(inpath, concat_dim='time', combine='nested', data_vars="minimal", coords="minimal", compat="override").isel(time=slice(1, None)
    print(ds)
    ds = ds.chunk({'time': 1})
    ds.to_zarr(outpath, compute=False)

    for i in range(ds.time.shape[0]):
        print(i)
        ds.isel(time=slice(i, i + 1)).drop_vars(['forecast_period', 'forecast_reference_time']).to_zarr(outpath, region='auto')

