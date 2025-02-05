import sys
from pathlib import Path
from itertools import product

import iris
import numpy as np
import torch
import xarray as xr

import earth2grid


def convert_latlon_cube_to_healpix(inpath, outpath, varname, max_level=10):
    tmpdir = Path('/work/scratch-nopw2/mmuetz/wcrp_hackathon/')
    tmppath = tmpdir.joinpath(*inpath.parts[1:]).with_suffix('.nc')
    tmppath.parent.mkdir(exist_ok=True, parents=True)

    # For some reason, loading .pp then saving as .nc using iris, then reloading with xarray
    # is way faster.
    print('load_cube')
    cube = iris.load_cube(inpath)
    print('save cube')
    iris.save(cube, tmppath)

    print('open da')
    da = xr.open_dataset(tmppath)[varname]
    dims = da.dims

    print('create ds')
    dsout = xr.Dataset(coords=da.copy().drop_vars(['latitude', 'longitude']).coords, attrs=da.attrs)
    print('load da')
    da.load()
    print('loaded')

    # all dims/coords with latitude, longitude.
    reduced_dims = [d for d in dims if d not in ['latitude', 'longitude']]
    reduced_coords = dsout.coords

    dim_shape = [v for v in dsout.sizes.values()]
    dim_ranges = [range(s) for s in dim_shape]

    lat_lon_shape = (len(da.latitude), len(da.longitude))
    src = earth2grid.latlon.equiangular_lat_lon_grid(*lat_lon_shape)

    # Build a slice to invert latitude (for passing to regridding).
    data_slice = [slice(None) if d != 'latitude' else slice(None, None, -1) for d in dims]
    target_data = da.values[*data_slice].copy().astype(np.double)

    for level in range(max_level + 1):
        # Note, you pass in PixelOrder.NEST here. .XY() (as in example) is equivalent to .RING.
        hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.PixelOrder.NEST)
        regrid = earth2grid.get_regridder(src, hpx)

        regridded_data = np.zeros(dim_shape + [hpx.shape[0]])

        # Note this works for empty dim_ranges (goes through loop once).
        for idx in product(*dim_ranges):
            print(level, idx)

            # TODO: Why is this needed??!?!?
            # My guess is that the y-dir is indexed in reverse for some reason.
            # The underlying code needs data in the right order, which is why the .copy() is nec.
            z_torch = torch.as_tensor(target_data[idx])
            # print('  created z_torch')
            z_hpx = regrid(z_torch)
            # if idx == () this still works (i.e. does nothing to regridded_data).
            regridded_data[idx] = z_hpx.numpy()

        dsout[f'{varname}_{level}'] = xr.DataArray(
            regridded_data,
            dims=reduced_dims + [f'cell_{level}'],
            coords={f'cell_{level}': np.arange(len(z_hpx))}
        )

    dsout.attrs['grid_mapping'] = 'healpix_nested'
    dsout.attrs['input_file'] = str(inpath)
    dsout.attrs['suite'] = 'u-cu087'

    outpath.parent.mkdir(exist_ok=True, parents=True)
    dsout.to_netcdf(outpath)
    tmppath.unlink()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        inpath = Path('/gws/nopw/j04/hrcm/cache/torau/Lorenzo_u-cu087/pe_T/20200101T0000Z_pe000.pp')
        outpath = Path('/work/scratch-nopw2/mmuetz/wcrp_hackathon/20200101T0000Z_pe000.hpz0-10.nc')
        varname = 'air_temperature'
    else:
        inpath = Path(sys.argv[1])
        outpath = Path(sys.argv[2])
        varname = sys.argv[3]

    convert_latlon_cube_to_healpix(inpath, outpath, varname)
