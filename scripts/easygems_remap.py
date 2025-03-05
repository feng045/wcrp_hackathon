# Make a .nc file we can use for testing.
from pathlib import Path

import cartopy.crs as ccrs
import healpix as hp
import iris
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import easygems.healpix as egh
import easygems.remap as egr


ncpath = Path('/gws/nopw/j04/hrcm/mmuetz/Lorenzo_u-cu087/OLR/20200101T0000Z_pa000.nc')
weights_path = Path('/gws/nopw/j04/hrcm/mmuetz/Lorenzo_u-cu087/OLR/regrid_weights_N2560_hpz10.nc')
hppath = Path('/gws/nopw/j04/hrcm/mmuetz/Lorenzo_u-cu087/OLR/healpix_20200101T0000Z_pa000.nc')


if not ncpath.exists():
    cube = iris.load_cube('/gws/nopw/j04/hrcm/cache/torau/Lorenzo_u-cu087/OLR/20200101T0000Z_pa000.pp')
    iris.save(cube, ncpath)

ds = xr.open_dataset(ncpath)
print(ds)

order = zoom = 10
nside = hp.order2nside(order)
npix = hp.nside2npix(nside)

hp_lon, hp_lat = hp.pix2ang(nside=nside, ipix=np.arange(npix), lonlat=True, nest=True)
hp_lon = hp_lon % 360  # [0, 360)
hp_lon += 360 / (4 * nside) / 4  # shift quarter-width

ds_flat = ds.stack(cell=('longitude', 'latitude'))

if not weights_path.exists():
    print('computing weights')
    weights = egr.compute_weights_delaunay((ds_flat.longitude.values, ds_flat.latitude.values), (hp_lon, hp_lat))
    weights.to_netcdf(weights_path)
else:
    print('loading weights')
    weights = xr.open_dataset(weights_path)

olr_hp = egr.apply_weights(ds_flat.toa_outgoing_longwave_flux, **weights)
olr_hp.to_netcdf(hppath)



