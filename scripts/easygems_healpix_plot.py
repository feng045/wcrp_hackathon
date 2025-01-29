# https://easy.gems.dkrz.de/Processing/healpix/healpix_cartopy.html

import intake
import cartopy.crs as ccrs
import cartopy.feature as cf
import cmocean
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import easygems.healpix as egh

def worldmap(var, **kwargs):
    # projection = ccrs.Robinson(central_longitude=-135.5808361)
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True
    )
    ax.set_global()

    egh.healpix_show(var, ax=ax, **kwargs)
    ax.add_feature(cf.COASTLINE, linewidth=0.8)
    ax.add_feature(cf.BORDERS, linewidth=0.4)


# data = xr.open_dataset('/home/markmuetz/mirrors/jasmin/gws/nopw/j04/hrcm/cache/torau/Lorenzo_u-cu087/data/healpix/20200101T0000Z_pfu4728.hpz9.nc')
# worldmap(data.isel(t=0, p=0), cmap=cmocean.cm.thermal)

NSIDE = 2
print(
    "Approximate resolution at NSIDE {} is {:.2} deg".format(
        NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60
    )
)

NPIX = hp.nside2npix(NSIDE)
print(NPIX)

m = np.arange(NPIX)
worldmap(m, cmap=cmocean.cm.thermal, nest=True)
plt.show()