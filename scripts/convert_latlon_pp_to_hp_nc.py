import sys

from pathlib import Path
import iris
import numpy as np
import torch
import xarray as xr

import earth2grid


def convert_latlon_pp_to_healpix_nc(inpath, outpath, varname, max_level=10):
    print(f'Load data from: {inpath}')
    cube = iris.load_cube(inpath)
    src = earth2grid.latlon.equiangular_lat_lon_grid(*cube.shape)

    ds = xr.Dataset()
    for level in range(max_level + 1):
        print(level)
        # Note, you pass in PixelOrder.NEST here. .XY() (as in example) is equivalent to .RING.
        hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.PixelOrder.NEST)
        regrid = earth2grid.get_regridder(src, hpx)

        # TODO: Why is this needed??!?!?
        # My guess is that the y-dir is indexed in reverse for some reason.
        # The underlying code needs data in the right order, which is why the .copy() is nec.
        z_torch = torch.as_tensor(cube.data.astype(np.double)[::-1].copy())
        z_hpx = regrid(z_torch)
        ds[f'{varname}_{level}'] = xr.DataArray(z_hpx.numpy(), coords={f'cell_{level}': np.arange(len(z_hpx))})

    print(f'Write data to: {outpath}')
    ds.to_netcdf(outpath)


if __name__ == '__main__':
    inpath = Path(sys.argv[1])
    outpath = Path(sys.argv[2])
    varname = sys.argv[3]

    convert_latlon_pp_to_healpix_nc(inpath, outpath, varname)

