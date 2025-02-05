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

# def worldmap(var, **kwargs):
#     # projection = ccrs.Robinson(central_longitude=-135.5808361)
#     projection = ccrs.PlateCarree()
#     fig, ax = plt.subplots(
#         figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True
#     )
#     ax.set_global()
#
#     egh.healpix_show(var, ax=ax, **kwargs)
#     ax.add_feature(cf.COASTLINE, linewidth=0.8)
#     ax.add_feature(cf.BORDERS, linewidth=0.4)


# data = xr.open_dataset('/home/markmuetz/mirrors/jasmin/gws/nopw/j04/hrcm/cache/torau/Lorenzo_u-cu087/data/healpix/20200101T0000Z_pfu4728.hpz9.nc')
# worldmap(data.isel(t=0, p=0), cmap=cmocean.cm.thermal)

# NSIDE = 2
# print(
#     "Approximate resolution at NSIDE {} is {:.2} deg".format(
#         NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60
#     )
# )
#
# NPIX = hp.nside2npix(NSIDE)
# print(NPIX)
#
# m = np.arange(NPIX)
# worldmap(m, cmap=cmocean.cm.thermal, nest=True)
# plt.show()

data = xr.open_dataset('/home/markmuetz/mirrors/jasmin/gws/nopw/j04/hrcm/cache/torau/Lorenzo_u-cu087/data/healpix/20200101T0000Z_pfu4728.hpz9.nc')

d = data.u.isel(t=0, p=0).values
# print(d)

def hp_coarsen(data):
    assert data.size % 12 == 0 and data.size // 12 % 4 == 0
    return data.reshape(-1, 4).mean(axis=1)

dc = d.copy()
for i in range(1):
    dc = hp_coarsen(dc)


def plot_storm(level, focus_on_storm, gdata, vmin, vmax):
    # Setting dpi sets the resolution of the lat/lon interp'd data.
    # kwargs = {'dpi': 500, 'cmap': cmocean.cm.thermal}
    if focus_on_storm:
        kwargs = {'dpi': 500, 'cmap': cmocean.cm.thermal}
    else:
        kwargs = {'dpi': 200, 'cmap': cmocean.cm.thermal}

    projection = ccrs.PlateCarree()
    if focus_on_storm:
        figsize = (8, 8)
    else:
        figsize = (8, 4)

    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": projection}, constrained_layout=True
    )
    ax.set_global()
    mean = gdata.mean()
    ax.set_title(f'UM u, healpix Level {level}, global mean = {mean:.8f} m/s')

    egh.healpix_show(gdata, ax=ax, vmin=vmin, vmax=vmax, **kwargs)
    ax.add_feature(cf.COASTLINE, linewidth=0.8)
    ax.add_feature(cf.BORDERS, linewidth=0.4)
    if focus_on_storm:
        ax.set_extent((-70, -40, 30, 60))
    plt.savefig(f'figs/u_healpix.level_{level}.storm_{focus_on_storm}.png')
    # plt.show()

vmin, vmax = np.percentile(d, (1, 99))
for focus_on_storm in (True, False):
    plot_storm(9, focus_on_storm, d, vmin, vmax)

    dc = d.copy()
    for level in range(8, -1, -1):
        dc = hp_coarsen(dc)
        plot_storm(level, focus_on_storm, dc, vmin, vmax)
        print(dc.mean())