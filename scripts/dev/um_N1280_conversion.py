import sys
from pathlib import Path

import iris
import xarray as xr
from loguru import logger

sys.path.insert(0, '/home/users/mmuetz/deploy/global_hackathon_tools/dataset_transforms')
from um_latlon_pp_to_healpix_nc import UMLatLon2HealpixRegridder, gen_weights


def weights_filename(da, lonname, latname):
    lon0, lonN = da[lonname].values[[0, -1]]
    lat0, latN = da[latname].values[[0, -1]]
    lonstr = f'({lon0.item():.3f},{lonN.item():.3f},{len(da[lonname])})'
    latstr = f'({lat0.item():.3f},{latN.item():.3f},{len(da[latname])})'

    return f'regrid_weights_N1280_hpz10.cyclic_lon.lon={lonstr}.lat={latstr}.nc'


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level='TRACE')
    pp_file = '/gws/nopw/j04/hrcm/torau/u-dc009/1hrly/dc009a.p919800101.pp'
    out_file = '/gws/nopw/j04/hrcm/mmuetz/u-dc009/1hrly/dc009a.p919800101.pp.hpz9.nc'
    cubes = iris.load(pp_file)
    weights = {}
    regridders = {}
    ds = xr.Dataset()
    for cube in cubes:
        name = cube.name()
        logger.info(f'Regridding: {name}')
        da = xr.DataArray.from_iris(cube)
        lonname = [c for c in da.coords if c.startswith('longitude')][0]
        latname = [c for c in da.coords if c.startswith('latitude')][0]

        weights_path = Path('/gws/nopw/j04/hrcm/mmuetz/weights') / weights_filename(da, lonname, latname)
        if not weights_path.exists():
            gen_weights(da, zoom=10, lonname=lonname, latname=latname, weights_path=weights_path)
            # Not a regional model, and you need to add a cyclic column to the lat/lon domain
            # to make sure that it's larger than the healpix domain.
        if not weights_path in regridders:
            regridders[weights_path] = UMLatLon2HealpixRegridder(method='easygems_delaunay', weights_path=weights_path,
                                                                 regional=False, add_cyclic=True)
        regridder = regridders[weights_path]
        da_hp = regridder.regrid(da, lonname, latname)
        da_hp = da_hp.rename(name)
        logger.debug(da_hp)
        ds[cube.name()] = da_hp
        # da_hp.to_netcdf(f'{name}.nc')
        logger.debug('-> regridded')
    ds.to_netcdf(out_file)
