from pathlib import Path

import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar
from loguru import logger

from processing_config import chunks2d
from calc_completed_chunks import find_tgt_calcs, find_tgt_time_calcs


def coarsen_healpix_zarr_region(src_ds, tgt_store, tgt_zoom, dim, start_idx, end_idx, chunks, progress_bar=False):
    time_slice = slice(start_idx, end_idx)
    logger.info(f'Coarsen to {tgt_zoom}, time_slice={time_slice}')

    tgt_chunks = chunks[tgt_zoom]

    src_ds_region = src_ds.isel(time=time_slice)
    # To compute or not to compute...
    # Computing would force the download and loading into mem, but might stop contention for the resource?
    # tgt_ds = src_ds_region.coarsen(cell=4).mean()
    tgt_ds = src_ds_region.coarsen(cell=4).mean().compute()

    logger.info('modify metadata')
    if dim == '2d':
        preferred_chunks = {'time': tgt_chunks[0], 'cell': tgt_chunks[1]}
    else:
        preferred_chunks = {'time': tgt_chunks[0], 'pressure': tgt_chunks[1], 'cell': tgt_chunks[2]}
    for da in tgt_ds.data_vars.values():
        logger.debug(f'  modify {da.name}')
        da.attrs['healpix_zoom'] = tgt_zoom
        da.encoding['chunks'] = tgt_chunks
        da.encoding['preferred_chunks'] = preferred_chunks
    tgt_ds = tgt_ds.drop_vars('crs')

    if dim == '2d':
        region = {'time': time_slice, 'cell': slice(None)}
    elif dim == '3d':
        region = {'time': time_slice, 'pressure': slice(None), 'cell': slice(None)}
    if progress_bar:
        job = tgt_ds.chunk(preferred_chunks).to_zarr(tgt_store, region=region, compute=False)
        logger.info('compute to_zarr')
        with ProgressBar():
            job.compute()
    else:
        for da in tgt_ds.data_vars.values():
            logger.debug(f'  writing {da.name}')
            da.chunk(preferred_chunks).to_zarr(tgt_store, region=region)


if __name__ == '__main__':
    s3cfg = dict([l.split(' = ') for l in Path('/home/users/mmuetz/.s3cfg').read_text().split('\n') if l])
    urls = {
        z: 'sim-data/dev/{sim}/v4/data.healpix.{freq}.z{zoom}.zarr'.format(sim='glm.n2560_RAL3p3', freq='PT1H', zoom=z)
        for z in range(11)
    }

    jasmin_s3 = s3fs.S3FileSystem(
        anon=False, secret=s3cfg['secret_key'],
        key=s3cfg['access_key'],
        client_kwargs={'endpoint_url': 'http://hackathon-o.s3.jc.rl.ac.uk'}
    )
    src_store = s3fs.S3Map(root=urls[9], s3=jasmin_s3, check=False)
    tgt_store = s3fs.S3Map(root=urls[8], s3=jasmin_s3, check=False)

    zarr_chunks = {'time': chunks2d[8][0], 'cell': -1}
    src_ds = xr.open_zarr(src_store, chunks=zarr_chunks)

    tgt_calcs = find_tgt_calcs(urls, chunks=chunks2d, dim='2d', variable='tas')
    tgt_time_calcs = find_tgt_time_calcs(tgt_calcs, chunks2d)
    start_idx = tgt_time_calcs[8][0]['start_idx']
    end_idx = tgt_time_calcs[8][0]['end_idx']
    logger.debug((start_idx, end_idx))
    coarsen_healpix_zarr_region(src_ds, tgt_store, 8, start_idx, end_idx, chunks2d)


