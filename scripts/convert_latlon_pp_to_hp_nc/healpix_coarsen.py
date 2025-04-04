from pathlib import Path

import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar
from loguru import logger


def coarsen_healpix_region(z, start_time_idx, chunks, n_tgt_time_chunks=4):
    s3cfg = dict([l.split(' = ') for l in Path('/home/users/mmuetz/.s3cfg').read_text().split('\n') if l])

    jasmin_s3 = s3fs.S3FileSystem(
        anon=False, secret=s3cfg['secret_key'],
        key=s3cfg['access_key'],
        client_kwargs={'endpoint_url': 'http://hackathon-o.s3.jc.rl.ac.uk'}
    )

    z_tgt = z - 1
    chunks_src = chunks[z]
    chunks_tgt = chunks[z_tgt]
    n_times = chunks_tgt[0] * n_tgt_time_chunks

    # TODO: hardcoded stores.
    # url_tpl = 'http://hackathon-o.s3.jc.rl.ac.uk/sim-data/5km-RAL3/dev/data.2d.v1.z{zoom}.zarr'
    store = s3fs.S3Map(root='s3://sim-data/5km-RAL3/dev/data.2d.v1.z{zoom}.zarr'.format(zoom=z), s3=jasmin_s3,
                       check=False)
    store_tgt = s3fs.S3Map(root='s3://sim-data/5km-RAL3/dev/data.2d.v1.z{zoom}.zarr'.format(zoom=z_tgt), s3=jasmin_s3,
                           check=False)

    logger.info('opening input zarr store')
    zarr_chunks = {'time': chunks_tgt[0], 'cell': -1}
    logger.debug(zarr_chunks)
    # ds_src = xr.open_dataset(store, chunks=zarr_chunks, engine='zarr').isel(time=slice(start_time_idx, start_time_idx + n_time_chunks))
    ds_src = xr.open_zarr(store, chunks=zarr_chunks).isel(time=slice(start_time_idx, start_time_idx + n_times))
    ds_src = ds_src[['pr', 'tas']]
    logger.debug(ds_src)

    logger.info('lazy coarsen')
    ds_tgt = ds_src.coarsen(cell=4).mean()
    logger.debug(ds_tgt)

    logger.info('modify metadata')
    preferred_chunks = {'time': chunks_tgt[0], 'cell': chunks_tgt[1]}
    for da in ds_tgt.data_vars.values():
        da.attrs['healpix_zoom'] = z_tgt
        da.encoding['chunks'] = chunks_tgt
        da.encoding['preferred_chunks'] = preferred_chunks

    logger.info('shorten coarse ds')

    logger.info('lazy to_zarr')
    region = {'time': slice(start_time_idx, start_time_idx + n_times), 'cell': slice(None)}
    logger.debug(region)
    job = ds_tgt.chunk(preferred_chunks).to_zarr(store_tgt, region=region, compute=False)

    logger.info('compute to_zarr')
    with ProgressBar() as pbar:
        job.compute()
