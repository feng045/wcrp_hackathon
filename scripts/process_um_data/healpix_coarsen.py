import sys
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr
from loguru import logger

from processing_config import processing_config


def nan_weight(arr, axis):
    return ((~np.isnan(arr)).sum(axis=axis) / 4).astype(np.float32)


def map_regional_to_global(_da, src_zoom, dim):
    logger.debug(_da.name)
    coords = {
        n: c for n, c in _da.coords.items()
        if n != 'cell'
    }
    coords['cell'] = np.arange(12 * 4 ** src_zoom)

    if dim == '2d':
        dims = ['time', 'cell']
    elif dim == '3d':
        dims = ['time', 'pressure', 'cell']
    # dims = [d for d in coords if d != 'crs']
    shape = list(_da.shape[:-1]) + [len(coords['cell'])]
    # Would be nice, but it's actually quite a bit slower?
    # dummies = dask.array.zeros(shape, dtype=np.float32)
    # da_new = xr.DataArray(dummies, dims=dims, name=_da.name, coords=coords)
    da_new = xr.DataArray(np.full(shape, np.nan, np.float32), dims=dims, name=_da.name, coords=coords)
    # Dangerously wrong! Will silently ignore this.
    # da_new.isel(cell=_da.cell).values = _da.values

    # This on the other hand works perfectly, because .loc allows assignment.
    da_new.loc[dict(cell=_da.cell)] = _da
    return da_new


def map_global_to_regional(_da, src_ds_region, tgt_ds_store, dim):
    logger.debug(_da.name)
    coords = {
        n: c for n, c in _da.coords.items()
        if n != 'cell'
    }
    coords['cell'] = tgt_ds_store.cell
    if dim == '2d':
        dims = ['time', 'cell']
    elif dim == '3d':
        dims = ['time', 'pressure', 'cell']
    da = list(src_ds_region.data_vars.values())[0]
    shape = list(da.shape[:-1]) + [len(coords['cell'])]

    da_new = xr.DataArray(np.full(shape, np.nan, np.float32), dims=dims, name=_da.name, coords=coords)
    da_new.values = _da.isel(cell=tgt_ds_store.cell.values)
    return da_new


def calc_tgt_weights(src_ds, src_ds_region, tgt_ds, tgt_zoom, dim):
    if 'weights' in src_ds_region:
        # TODO: Need to handle lower zooms when weights already exist.
        src_weights = src_ds_region['weights']
        weights = src_weights.coarsen(cell=4).mean()
    else:
        # It is enough to calc the weights for one field, and store.
        if dim == '2d':
            src_da = list(src_ds.data_vars.values())[0].isel(time=0)
        elif dim == '3d':
            src_da = list(src_ds.data_vars.values())[0].isel(time=0, pressure=0)
        weights = src_da.coarsen(cell=4).reduce(nan_weight).compute()
    cells = np.arange(12 * 4 ** tgt_zoom)
    coords = {'cell': cells}
    # Use a global weights DataArray to convert between the src domain and the tgt domain.
    weights_global = xr.DataArray(np.full(12 * 4 ** tgt_zoom, np.nan, np.float32), dims=['cell'], coords=coords)
    # TODO: I'm having difficulty figuring out how to use cells to map into weights_global
    tgt_cell_from_src = ((src_ds.cell.values.reshape(-1, 4).mean(axis=-1) - 1.5) / 4).astype(int)
    weights_global.loc[dict(cell=tgt_cell_from_src)] = weights.values
    tgt_weights = weights_global.isel(cell=tgt_ds.cell)
    logger.debug(float(np.isnan(tgt_weights).sum().values))
    return tgt_weights


def coarsen_healpix_zarr_region(src_ds, tgt_store, tgt_zoom, dim, start_idx, end_idx, chunks, regional=False):
    time_slice = slice(start_idx, end_idx)
    src_zoom = tgt_zoom + 1
    logger.info(f'Coarsen to {tgt_zoom}, time_slice={time_slice}')

    tgt_chunks = chunks[tgt_zoom]

    src_ds_region = src_ds.isel(time=time_slice)
    if regional:
        src_ds_region = src_ds_region.map(map_regional_to_global, src_zoom=src_zoom, dim=dim)

    # To compute or not to compute...
    # Computing would force the download and loading into mem, but might stop contention for the resource?
    # tgt_ds = src_ds_region.coarsen(cell=4).mean()
    logger.debug('compute tgt_ds')
    tgt_ds = src_ds_region.coarsen(cell=4).mean().compute()

    logger.info('modify metadata')
    if dim == '2d':
        preferred_chunks = {'time': tgt_chunks[0], 'cell': tgt_chunks[1]}
    else:
        preferred_chunks = {'time': tgt_chunks[0], 'pressure': tgt_chunks[1], 'cell': tgt_chunks[2]}
    for da in tgt_ds.data_vars.values():
        da.attrs['healpix_zoom'] = tgt_zoom
        da.encoding['chunks'] = tgt_chunks
        da.encoding['preferred_chunks'] = preferred_chunks

    if regional:
        zarr_chunks = {'time': chunks[tgt_zoom][0], 'cell': -1}
        tgt_ds_store = xr.open_zarr(tgt_store, chunks=zarr_chunks)
        tgt_ds = tgt_ds.map(map_global_to_regional, src_ds_region=src_ds_region, tgt_ds_store=tgt_ds_store, dim=dim)
        if src_ds_region.time[0] == src_ds.time[0]:
            tgt_weights = calc_tgt_weights(src_ds, src_ds_region, tgt_ds, tgt_zoom, dim)
            # TODO: Need to create weights in empty zarr store for this to work.
            tgt_ds['weights'] = tgt_weights
            logger.debug(float(np.isnan(tgt_ds.weights).sum().values))

    tgt_ds = tgt_ds.drop_vars('crs')

    if dim == '2d':
        region = {'time': time_slice, 'cell': slice(None)}
    elif dim == '3d':
        region = {'time': time_slice, 'pressure': slice(None), 'cell': slice(None)}
    for da in tgt_ds.data_vars.values():
        logger.debug(f'  writing {da.name}')
        if da.name == 'weights':
            da.chunk({'cell': preferred_chunks['cell']}).to_zarr(tgt_store, region={'cell': slice(None)})
        else:
            da.chunk(preferred_chunks).to_zarr(tgt_store, region=region)


def main():
    s3cfg = dict([l.split(' = ') for l in Path('/home/users/mmuetz/.s3cfg').read_text().split('\n') if l])
    sim = sys.argv[1]
    dim = sys.argv[2]
    zoom = int(sys.argv[3])
    start_idx = int(sys.argv[4])
    end_idx = int(sys.argv[5])
    freqs = {
        '2d': 'PT1H',
        '3d': 'PT3H',
    }
    freq = freqs[dim]
    urls = {
        z: 'sim-data/dev/{sim}/v4/data.healpix.{freq}.z{zoom}.zarr'.format(sim=sim, freq=freq, zoom=z)
        for z in range(11)
    }

    config = processing_config[sim]
    chunks = config['groups'][dim]['chunks']

    jasmin_s3 = s3fs.S3FileSystem(
        anon=False, secret=s3cfg['secret_key'],
        key=s3cfg['access_key'],
        client_kwargs={'endpoint_url': 'http://hackathon-o.s3.jc.rl.ac.uk'}
    )
    src_store = s3fs.S3Map(root=urls[zoom], s3=jasmin_s3, check=False)
    test_tgt_store = s3fs.S3Map(root=urls[zoom - 1], s3=jasmin_s3, check=False)

    test_zarr_chunks = {'time': chunks[zoom - 1][0], 'cell': -1}
    test_src_ds = xr.open_zarr(src_store, chunks=test_zarr_chunks)

    # tgt_calcs = find_tgt_calcs(urls, chunks=chunks2d, dim='2d', variable='tas')
    # tgt_time_calcs = find_tgt_time_calcs(tgt_calcs, chunks2d)
    # start_idx = tgt_time_calcs[zoom - 1][0]['start_idx']
    # end_idx = tgt_time_calcs[zoom - 1][0]['end_idx']
    logger.debug((start_idx, end_idx))
    coarsen_healpix_zarr_region(test_src_ds, test_tgt_store, zoom - 1, dim, start_idx, end_idx, chunks,
                                regional=config['regional'])

if __name__ == '__main__':
    main()
