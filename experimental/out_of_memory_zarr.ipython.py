# coding: utf-8
dummies = dask.array.zeros((1000, 1000, 1000), chunks=(100, 100, 100))
import dask.array
import numpy as np
import xarray as xr
ds = xr.Dataset({'d1': (('t', 'y', 'x'), dummies)}, coords={'t': np.arange(1000), 'y': np.arange(1000), 'x': np.arange(1000)})
dummies = dask.array.zeros((1000, 1000, 1000), chunks=(100, 100, 100))
ds = xr.Dataset({'d1': (('t', 'y', 'x'), dummies)}, coords={'t': np.arange(1000), 'y': np.arange(1000), 'x': np.arange(1000)})
ds.to_zarr('huge3.zarr', compute=False)
dummies = dask.array.zeros((3200, 1600, 1600), chunks=(100, 100, 100))
ds = xr.Dataset({'d1': (('t', 'y', 'x'), dummies)}, coords={'t': np.arange(3200), 'y': np.arange(1600), 'x': np.arange(1600)})
ds
ds.d1.nbytes
ds.d1.nbytes / 1e9
ds.to_zarr('huge4.zarr', compute=False)
ds.to_zarr('huge4.zarr', compute=False)
dsw = xr.Dataset({"foo": ("x", np.arange(30))}, coords={"x": np.arange(30)})
ds
ds.isel(t=slice(100), y=slice(100), x=slice(100))
ds.isel(t=slice(100), y=slice(100), x=slice(100)).to_zarr('huge4.zarr/', region='auto')
ds.isel(t=slice(100), y=slice(100), x=slice(100)).to_zarr('huge4.zarr/', region='auto', compute=True)
ds.isel(t=slice(100), y=slice(100), x=slice(100)).values = 1
ds.isel(t=slice(100), y=slice(100), x=slice(100)).values[:] = 1
ds.isel(t=slice(100), y=slice(100), x=slice(100)).values()
ds.isel(t=slice(100), y=slice(100), x=slice(100)).values() = 1
ds.isel(t=slice(100), y=slice(100), x=slice(100)).values()[:]
ds.isel(t=slice(100), y=slice(100), x=slice(100)).values().sum()
ds.d1.isel(t=slice(100), y=slice(100), x=slice(100)).values
ds.d1.isel(t=slice(100), y=slice(100), x=slice(100)).values = 1
ds.d1.isel(t=slice(100), y=slice(100), x=slice(100)).values = np.arange(1000000).reshape(100, 100, 100)
ds.isel(t=slice(100), y=slice(100), x=slice(100)).to_zarr('huge4.zarr/', region='auto', compute=True)
zstore = ds.isel(t=slice(100), y=slice(100), x=slice(100)).to_zarr('huge4.zarr/', region='auto', compute=True)
zstore
zstore.sync()
dsr = xr.open_zarr('huge4.zarr/')
dsr
dsr.d1.isel(t=slice(100), y=slice(100), x=slice(100))
dsr.d1.isel(t=slice(100), y=slice(100), x=slice(100)).values
ds.d1.isel(t=slice(100), y=slice(100), x=slice(100)).values
ds.d1.isel(t=slice(100), y=slice(100), x=slice(100)).values = np.arange(1000000).reshape(100, 100, 100)
ds.d1.isel(t=slice(100), y=slice(100), x=slice(100)).values
ds.d1.isel(t=slice(100), y=slice(100), x=slice(100)).values = np.arange(1000000).reshape(100, 100, 100) 
dummies
dummies[:100, :100, :100] = np.arange(1000000).reshape(100, 100, 100)
dummies[:100, :100, :100]
dummies[:100, :100, :100].values
dummies[:100, :100, :100].compute()
ds = xr.Dataset({'d1': (('t', 'y', 'x'), dummies)}, coords={'t': np.arange(3200), 'y': np.arange(1600), 'x': np.arange(1600)})
ds.isel(t=slice(100), y=slice(100), x=slice(100)).to_zarr('huge4.zarr/', region='auto')
dummies[:1000, :1000, :1000] = np.arange(1000000000).reshape(1000, 1000, 1000)
ds = xr.Dataset({'d1': (('t', 'y', 'x'), dummies)}, coords={'t': np.arange(3200), 'y': np.arange(1600), 'x': np.arange(1600)})
ds.isel(t=slice(1000), y=slice(1000), x=slice(1000)).to_zarr('huge4.zarr/', region='auto')
