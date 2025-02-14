import os
from pathlib import Path

import fsspec
from kerchunk.hdf import SingleHdf5ToZarr
from kerchunk.combine import MultiZarrToZarr
import ujson
import xarray as xr

orig_cwd = Path.cwd()
os.chdir('/gws/nopw/j04/hrcm/mmuetz/Lorenzo_u-cu087/OLR/healpix')
try:

    fs = fsspec.filesystem('file')
    if False:
        paths = fs.glob('./*pa???.hpz0-10.nc') + fs.glob('./*pa????.hpz0-10.nc')
        for p in paths:
            print(p)
            with fs.open(p, 'rb') as fp:
                h5chunks = SingleHdf5ToZarr(fp, p)
                Path(p).with_suffix('.json').write_text(ujson.dumps(h5chunks.translate()))

        json_paths = fs.glob('./*pa???.hpz0-10.json') + fs.glob('./*pa????.hpz0-10.json')
    # json_paths = fs.glob('./*pa???.hpz0-10.json')
    # mzz = MultiZarrToZarr(json_paths, concat_dims=['time'])
    # d = mzz.translate()
    # Path('experimental_20200101T0000Z_pa.combined.json').write_text(ujson.dumps(d))

    # ds = xr.open_dataset("reference://", engine="zarr", backend_kwargs={
    #                     "consolidated": False,
    #                     "storage_options": {"fo": 'experimental_20200101T0000Z_pa.combined.json', 'lazy': True},
    #                                         })

    # print(ds)
    # ds.to_zarr('experimental_20200101T0000Z_pa.combined.zarr')
    # ds2 = xr.open_dataset('experimental_20200101T0000Z_pa.combined.zarr/', engine='zarr')
    # ds.to_zarr('experimental_20200101T0000Z_pa.combined.zarr', encoding={'toa_outgoing_longwave_flux_10': {'chunks': (6, 12582912 // 12)}})
    ds2 = xr.open_zarr('experimental_20200101T0000Z_pa.combined.zarr/')
    print(ds2)
finally:
    os.chdir(orig_cwd)
