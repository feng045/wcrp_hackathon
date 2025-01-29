from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import torch

import earth2grid

src = earth2grid.latlon.equiangular_lat_lon_grid(2560 * 2, int(2560 * 1.5))
z = np.cos(np.deg2rad(src.lat)) * np.cos(np.deg2rad(src.lon))


for level in range(11):
    start = timer()
    # level is the resolution
    # level = 10
    hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
    regrid = earth2grid.get_regridder(src, hpx)

    z_torch = torch.as_tensor(z)
    z_hpx = regrid(z_torch)
    elapsed = timer() - start
    print(f'level {level}: {elapsed:.2f}s')

if False:

    fig, (a, b) = plt.subplots(2, 1)
    a.pcolormesh(src.lon, src.lat, z)
    a.set_title("Lat Lon Grid")

    b.scatter(hpx.lon, hpx.lat, c=z_hpx, s=0.1)
    b.set_title("Healpix")
    plt.show()
