import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import numpy as np
import healpy as hp
import xarray as xr

NSIDE = 32
print(
    "Approximate resolution at NSIDE {} is {:.2} deg".format(
        NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60
    )
)

NPIX = hp.nside2npix(NSIDE)
print(NPIX)

m = np.arange(NPIX)
hp.mollview(m, title="Mollview image RING")
hp.graticule()
plt.show()

fig, ax = plt.subplots()
# ax.coastlines()

data = xr.open_dataset('/home/markmuetz/mirrors/jasmin/gws/nopw/j04/hrcm/cache/torau/Lorenzo_u-cu087/data/healpix/20200101T0000Z_pfu4728.hpz9.nc')
hp.mollview(fig=fig, map=data.u.isel(t=0, p=0).values, title="Mollview image nest", nest=True, reuse_axes=True)
plt.show()
