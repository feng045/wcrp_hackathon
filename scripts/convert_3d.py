from pathlib import Path
from itertools import product
import iris
import numpy as np
import torch
import xarray as xr

import earth2grid

# def convert_latlon_cube_to_healpix(inpath, outpath, varname):
def convert_latlon_cube_to_healpix():
    print('load_cube')
    cube = iris.load_cube('/gws/nopw/j04/hrcm/cache/torau/Lorenzo_u-cu087/pe_T/20200101T0000Z_pe000.pp')
    print('save cube')
    iris.save(cube, '/work/scratch-nopw2/mmuetz/wcrp_hackathon/20200101T0000Z_pe000.nc')

    print('open da')
    da = xr.open_dataset('/work/scratch-nopw2/mmuetz/wcrp_hackathon/20200101T0000Z_pe000.nc').air_temperature

    dims = da.dims

    print('create ds')
    ds = xr.Dataset(coords=da.copy().drop_vars(['latitude', 'longitude']).coords, attrs=da.attrs)
    print('load da')
    da.load()
    print('loaded')

    reduced_dims = [d for d in dims if d not in ['latitude', 'longitude']]
    reduced_coords = ds.coords

    dim_shape = [v for v in ds.sizes.values()]
    dim_ranges = [range(s) for s in dim_shape]

    lat_lon_shape = (len(da.latitude), len(da.longitude))
    src = earth2grid.latlon.equiangular_lat_lon_grid(*lat_lon_shape)

    data_slice = [slice(None) if d != 'latitude' else slice(None, None, -1) for d in dims]
    target_data = da.values[*data_slice].copy().astype(np.double)

    max_level = 10
    for level in range(max_level + 1):
        # Note, you pass in PixelOrder.NEST here. .XY() (as in example) is equivalent to .RING.
        hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.PixelOrder.NEST)
        regrid = earth2grid.get_regridder(src, hpx)

        regridded_data = np.zeros(dim_shape + [hpx.shape[0]])

        for idx in product(*dim_ranges):
            print(level, idx)

            # TODO: Why is this needed??!?!?
            # My guess is that the y-dir is indexed in reverse for some reason.
            # The underlying code needs data in the right order, which is why the .copy() is nec.
            z_torch = torch.as_tensor(target_data[idx])
            # print('  created z_torch')
            z_hpx = regrid(z_torch)
            regridded_data[idx] = z_hpx.numpy()
            # break
        ds[f'{varname}_{level}'] = xr.DataArray(
            regridded_data,
            dims=reduced_dims + [f'cell_{level}'],
            coords={f'cell_{level}': np.arange(len(z_hpx))}
        )

    ds.to_netcdf('/work/scratch-nopw2/mmuetz/wcrp_hackathon/20200101T0000Z_pe000.hpz0-10.nc')


if __name__ == '__main__':
    convert_latlon_cube_to_healpix()
