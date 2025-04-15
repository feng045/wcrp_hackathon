import numpy as np
import xarray as xr

if True:
    t = np.linspace(0, 90, 10)
    y = np.linspace(0, 99, 100)
    x = np.linspace(0, 99, 200)

    data = np.arange(10 * 100 * 200).reshape(10, 100, 200)

    ds = xr.Dataset(
        data_vars=dict(data2=(('t', 'y', 'x'), data)),
        coords=dict(t=t, y=y, x=x)
    )
    ds.to_zarr('data.zarr', mode='a')
else:
    ds = xr.open_dataset('data.zarr', engine='zarr')
