from pathlib import Path
import pickle

import xarray as xr

from nc_to_zarr import paths, outpath

if __name__ == '__main__':
    print(f'n paths: {len(paths)}')

    pkl_path = Path('ds.pkl')
    if not pkl_path.exists():
        print('Generating ds pickle')
        ds = xr.open_mfdataset(paths, concat_dim='time', combine='nested', data_vars="minimal", coords="minimal", compat="override", parallel=True)
        ds = ds.chunk({'time': 1})
        pickle.dump(ds, pkl_path.open('wb'))
    else:
        print('Loading from pickle')
        ds = pickle.load(pkl_path.open('rb'))
    print(ds)
    if not outpath.exists():
        print('Generating zarr template')
        ds.to_zarr(outpath, compute=False)
