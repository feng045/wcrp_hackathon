import dask.array
import numpy as np
import xarray as xr

dummies = dask.array.zeros((3200, 1600, 1600), chunks=(100, 100, 100))
ds = xr.Dataset({'d1': (('t', 'y', 'x'), dummies)}, coords={'t': np.arange(3200), 'y': np.arange(1600), 'x': np.arange(1600)})
print(ds)
print(f'{ds.d1.nbytes / 1e9}G')
ds.to_zarr('out_of_memory.zarr', compute=False)
