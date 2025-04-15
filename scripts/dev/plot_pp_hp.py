import cartopy.crs as ccrs
import cartopy.feature as cf
import iris
import matplotlib.pyplot as plt
import xarray as xr

import easygems.healpix as egh


inpath = Path('/gws/nopw/j04/hrcm/cache/torau/Lorenzo_u-cu087/OLR/20200101T0000Z_pa000.pp')
# No write perms.
# outpath = inpath.parent / 'healpix' / ('experimental_' + inpath.stem + '.hpz0-10.nc')
outpath = Path('/gws/nopw/j04/hrcm/mmuetz/Lorenzo_u-cu087/OLR/healpix') / ('experimental_' + inpath.stem + '.hpz0-10.nc')

cube = iris.load_cube(inpath)
ds = xr.open_array(outpath)

# Plot results.
projection = ccrs.PlateCarree()
fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1,
	figsize=(8, 16), subplot_kw={"projection": projection}, constrained_layout=True
)
ax0.set_title('OLR lat/lon from native .pp')
ax0.set_global()
ax0.add_feature(cf.COASTLINE, linewidth=0.8)
ax0.pcolormesh(cube.coord('longitude').points, cube.coord('latitude').points, cube.data)

for level, ax in zip([3, 6, 10], [ax1, ax2, ax3]):

	ax.set_title(f'OLR regridded to healpix nest level {level}')
	ax.set_global()

	egh.healpix_show(ds[f'OLR_{level}'].values, ax=ax)
	ax.add_feature(cf.COASTLINE, linewidth=0.8)

plt.savefig('/home/users/mmuetz/tmp/OLR_pp_hp.png')

