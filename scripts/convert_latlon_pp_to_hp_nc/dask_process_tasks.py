import datetime as dt
import json
import sys
from collections import defaultdict
from functools import partial
from io import StringIO
from pathlib import Path

import dask
import dask.array
import iris
import numpy as np
import pandas as pd
import s3fs
import stratify
import xarray as xr
from dask.distributed import Client
from iris.experimental.stratify import relevel
from loguru import logger

from healpix_coarsen import coarsen_healpix_region
from processing_config import processing_config

sys.path.insert(0, '/home/users/mmuetz/deploy/global_hackathon_tools/dataset_transforms')
from um_latlon_pp_to_healpix_nc import UMLatLon2HealpixRegridder, gen_weights, get_limited_healpix

s3cfg = dict([l.split(' = ') for l in Path('/home/users/mmuetz/.s3cfg').read_text().split('\n') if l])
jasmin_s3 = s3fs.S3FileSystem(
    anon=False,
    secret=s3cfg['secret_key'],
    key=s3cfg['access_key'],
    client_kwargs={'endpoint_url': 'http://hackathon-o.s3.jc.rl.ac.uk'}
)

iris.FUTURE.date_microseconds = True


# @dask.delayed
def model_level_to_pressure(cube, p, z):
    from loguru import logger
    logger.debug(f're-level model level to pressure for {cube.name()}')
    # p = cubes.extract_cube('air_pressure')
    cube = cube[-p.shape[0]:]
    assert (p.coord('time').points == cube.coord('time').points).all()

    # z = cubes.extract_cube('geopotential_height')
    # Direction of pressure_levels must match that of air_pressure/p.
    pressure_levels = z.coord('pressure').points[::-1] * 100  # convert from hPa to Pa.
    interpolator = partial(stratify.interpolate,
                           interpolation=stratify.INTERPOLATE_LINEAR,
                           extrapolation=stratify.EXTRAPOLATE_LINEAR,
                           rising=False)
    new_cube_data = np.zeros((cube.shape[0], len(pressure_levels), cube.shape[2], cube.shape[3]))
    for i in range(cube.shape[0]):
        logger.trace(i)
        regridded_cube = relevel(cube[i], p[i], pressure_levels, interpolator=interpolator)
        logger.trace(f'regridded_cube.data.sum() {regridded_cube.data.sum()}')
        new_cube_data[i] = regridded_cube.data

    coords = [(cube.coord('time'), 0), (z.coord('pressure'), 1), (cube.coord('latitude'), 2),
              (cube.coord('longitude'), 3)]
    new_cube = iris.cube.Cube(new_cube_data,
                              long_name=cube.name(),
                              units=cube.units,
                              dim_coords_and_dims=coords,
                              attributes=cube.attributes)
    logger.trace(new_cube)
    return new_cube


# @dask.delayed
def da_to_zarr(da, zarr_store_url_tpl, group_name, group, zoom=10):
    from loguru import logger
    name = da.name
    logger.info(f'{name} to zarr for zoom level {zoom}')

    zarr_store_name = group['zarr_store']
    url = zarr_store_url_tpl.format(name=zarr_store_name, zoom=zoom)
    zarr_store = s3fs.S3Map(
        root=url,
        s3=jasmin_s3, check=False)

    half_time = find_halfpast_time(da)
    # Open the zarr store so I can get time coord to match with da's time coord.
    ds_tpl = xr.open_zarr(zarr_store)

    # Match source (da) to target (zarr_store) times.
    # Get an index into zarr store to allow me to write block of da's data.
    timename = [c for c in da.coords if c.startswith('time')][0]
    if timename == half_time:
        # This da is stored on the half hour because it's an hourly mean.
        # Need to add on half an hour here and then align with ds_tpl times.
        times_halfpast = pd.DatetimeIndex(da[timename].values)
        source_times_to_match = times_halfpast + pd.Timedelta(minutes=30)
    else:
        source_times_to_match = pd.DatetimeIndex(da[timename].values)
    zarr_time_name = 'time'
    target_times_to_match = pd.DatetimeIndex(ds_tpl[zarr_time_name].values)

    da = da.rename(**{timename: zarr_time_name})
    idx = np.argmin(np.abs(source_times_to_match[0] - target_times_to_match))
    assert (source_times_to_match - target_times_to_match[idx: idx + len(source_times_to_match)] < pd.Timedelta(
        minutes=5)).all(), 'source times do not match target times (thresh = 5 mins)'

    logger.debug(
        f'writing {name} to zarr store {url} (idx={idx}, time={source_times_to_match[0]})')
    if np.isnan(da.values).all():
        logger.error(da)
        raise Exception(f'da {da.name} is full of NaNs')

    if group_name == '2d':
        da.to_zarr(zarr_store,
                   region={zarr_time_name: slice(idx, idx + len(da[zarr_time_name])), 'cell': slice(None)})
    elif group_name == '3d':
        da.to_zarr(zarr_store,
                   region={zarr_time_name: slice(idx, idx + len(da[zarr_time_name])), 'pressure': slice(None),
                           'cell': slice(None)})
    return name


# @dask.delayed
def cube_to_da(cube):
    from loguru import logger
    logger.debug(f'convert {cube.name()} from iris to xarray')
    da = xr.DataArray.from_iris(cube)
    # For some cubes (ones with names like m01s30i461 the da gets a name like filled-XXXXXX...
    # Make sure it's got the actual cube name so I can rename it later.
    return da.rename(cube.name())


# @dask.delayed
def da_to_healpix(da, regrid_method, name_map, drop_vars, add_cyclic=True, regional=False, regional_chunks=None):
    from loguru import logger
    um_name = da.name
    long_name, short_name = name_map[um_name]

    lonname = [c for c in da.coords if c.startswith('longitude')][0]
    latname = [c for c in da.coords if c.startswith('latitude')][0]
    if regrid_method == 'easygems_delaunay':
        weights_path = weights_filepath(da, lonname, latname)
        if not weights_path.exists():
            logger.info(f'No weights for {da.name}, generating')
            gen_weights(da, 10, lonname, latname, add_cyclic=add_cyclic, regional=regional,
                        regional_chunks=regional_chunks, weights_path=weights_path)
        logger.trace(f'  - using weights: {weights_path}')
        regridder = UMLatLon2HealpixRegridder(method='easygems_delaunay', weights_path=weights_path,
                                              add_cyclic=add_cyclic, regional=regional, regional_chunks=regional_chunks)
    else:
        regridder = UMLatLon2HealpixRegridder(method=regrid_method, weights_path=None, add_cyclic=add_cyclic,
                                              regional=regional, regional_chunks=regional_chunks)

    # These have to be dropped before you cyclic pad *some* data arrays, or you will get a coord mismatch.
    drop_vars_exists = list(set(drop_vars) & set(k for k in da.coords.keys()))
    logger.debug(f'dropping {drop_vars_exists}')
    da = da.drop_vars(drop_vars_exists)

    da_hp = regridder.regrid(da, lonname, latname)
    da_hp = da_hp.rename(short_name)
    da_hp.attrs['UM_name'] = um_name
    da_hp.attrs['long_name'] = long_name
    da_hp.attrs['grid_mapping'] = 'healpix_nested'
    da_hp.attrs['healpix_zoom'] = 10

    return da_hp


def weights_filepath(da, lonname, latname):
    lon0, lonN = da[lonname].values[[0, -1]]
    lat0, latN = da[latname].values[[0, -1]]
    lonstr = f'({lon0.item():.3f},{lonN.item():.3f},{len(da[lonname])})'
    latstr = f'({lat0.item():.3f},{latN.item():.3f},{len(da[latname])})'

    # lonstr = str((lon0.item(), lonN.item(), len(da[lonname]))).replace(' ', '')
    # latstr = str((lat0.item(), latN.item(), len(da[latname]))).replace(' ', '')
    # TODO: better name.
    return Path(f'/gws/nopw/j04/hrcm/mmuetz/weights/regrid_weights_N2560_hpz10.cyclic_lon.lon={lonstr}.lat={latstr}.nc')


def find_halfpast_time(ds):
    times = {name: pd.DatetimeIndex(ds[name].values)
             for name in ds.coords if name.startswith('time')}
    for name, time in times.items():
        if ((time.second == 0) & (time.minute == 30)).all():
            return name
    return None


class DaskProcessUMFilesToZarrStore:
    def __init__(self, config):
        self.config = config
        self.drop_vars = config['drop_vars']
        self.groups = config['groups']

        self.debug_log = StringIO()
        logger.add(self.debug_log)
        self.use_dask_delayed = False

    def create_empty_zarr_stores(self, task):
        inpaths = task['inpaths']
        # TODO: Store-level metadata.
        zarr_store_url_tpl = self.config['zarr_store_url_tpl']

        all_cubes = iris.load(inpaths)
        logger.trace(all_cubes)

        regional = config.get('regional', False)

        for zoom in range(11)[::-1]:
            if zoom != 10:
                # TODO:
                continue
            npix = 12 * 4 ** zoom
            ds_tpls = defaultdict(xr.Dataset)

            for group_name, group in self.config['groups'].items():
                time = group['time']
                zarr_time_name = 'time'
                zarr_time = time

                zarr_store_name = group['zarr_store']

                logger.info(f'Creating {group_name}')
                cubes = all_cubes.extract(group['constraint'])
                name_map = group['name_map']
                chunks = group['chunks'][zoom]
                # Multipls of 4**n chosen as these will align well with healpix grids.
                # Aim for 1-10MB per chunk, bearing in mind that this is saved with 4-byte float32s.
                logger.trace(chunks)

                if 'extra_constraints' in group:
                    extra_constraints = [group['extra_constraints'].get(n, n) for n in name_map.keys()]
                    cubes = cubes.extract(extra_constraints)

                logger.info(f'Found {len(cubes)} cubes for {group_name}')
                for cube in cubes:
                    cube_name = cube.name()
                    long_name, short_name = group['name_map'][cube_name]
                    logger.info(f'- creating da for {cube_name} -> {short_name}')

                    # Want to be able to pass extra kwargs to from_iris but can't...
                    # da = xr.DataArray.from_iris(cube, decode_timedelta=True)
                    da = xr.DataArray.from_iris(cube).compute()
                    timename = [c for c in da.coords if c.startswith('time')][0]

                    if regional:
                        lonname = [c for c in da.coords if c.startswith('longitude')][0]
                        latname = [c for c in da.coords if c.startswith('latitude')][0]
                        minlon, maxlon = da[lonname].values[[0, -1]]
                        minlat, maxlat = da[latname].values[[0, -1]]
                        extent = [minlon, maxlon, minlat, maxlat]
                        _, _, ichunk = get_limited_healpix(extent, zoom=zoom, chunksize=chunks[-1])
                        cells = ichunk
                    else:
                        cells = np.arange(npix)

                    if da.ndim == 3:
                        dims = ['time', 'cell']
                        coords = {zarr_time_name: zarr_time, 'cell': cells}
                        shape = (len(zarr_time), len(cells))
                    elif da.ndim == 4:
                        dims = ['time', 'pressure', 'cell']
                        pressure_levels = [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 750,
                                           800, 850, 875, 900, 925, 950, 975, 1000]
                        coords = {zarr_time_name: zarr_time,
                                  'pressure': (['pressure'], pressure_levels, {'units': 'hPa'}),
                                  'cell': cells}
                        shape = (len(zarr_time), len(pressure_levels), len(cells))
                    else:
                        raise Exception('ndim must be 3 or 4')

                    dummies = dask.array.zeros(shape, dtype=np.float32, chunks=chunks)

                    da = da.rename(**{timename: zarr_time_name})
                    da.attrs['UM_name'] = cube_name
                    da.attrs['long_name'] = long_name
                    da.attrs['grid_mapping'] = 'healpix_nested'
                    da.attrs['healpix_zoom'] = zoom
                    da.attrs.update(group['extra_attrs'].get(cube_name, {}))
                    da_tpl = xr.DataArray(dummies, dims=dims, coords=coords, name=short_name, attrs=da.attrs)

                    ds_tpls[zarr_store_name][short_name] = da_tpl

            for zarr_store_name, ds_tpl in ds_tpls.items():
                crs = xr.DataArray(
                    name="crs",
                    attrs={
                        "grid_mapping_name": "healpix",
                        "healpix_nside": 2 ** 10,
                        "healpix_order": "nest",
                    },
                )
                ds_tpl = ds_tpl.assign_coords(crs=crs)

                logger.info(f'Saving {config_key} zoom={zoom}')
                store_url = zarr_store_url_tpl.format(name=zarr_store_name, zoom=zoom)
                zarr_store = s3fs.S3Map(
                    root=store_url,
                    s3=jasmin_s3, check=False)
                logger.debug(store_url)
                logger.debug(ds_tpl)
                ds_tpl.to_zarr(zarr_store, compute=False)

        logger.info('completed')
        if task['donepath']:
            Path(task['donepath']).write_text(self.debug_log.getvalue())

    def process_task(self, task):
        inpaths = task['inpaths']

        logger.info('loading cubes')
        logger.trace(inpaths)
        cubes = iris.load(inpaths)

        add_cyclic = self.config.get('add_cyclic', True)
        regional = self.config.get('regional', False)

        if self.use_dask_delayed:
            client = Client(threads_per_worker=1, n_workers=1)
            # client = Client()
            logger.debug(client)
            outputs = []

        p = cubes.extract_cube('air_pressure')
        z = cubes.extract_cube('geopotential_height')

        for group_name, group in self.groups.items():
            # if group_name != '3d_ml':
            #     continue
            logger.info(f'processing group {group_name}')
            group_constraint = group['constraint']
            name_map = group['name_map']
            chunks = group['chunks'][10]

            for name in name_map.keys():
                logger.info(f'adding {name} to processing chain')
                constraint = group['extra_constraints'].get(name, name)
                if group.get('interpolate_model_levels_to_pressure', False):
                    ml_cube = cubes.extract(group_constraint).extract_cube(constraint)
                    cubes.remove(ml_cube)
                    cube = model_level_to_pressure(ml_cube, p, z)
                else:
                    cube = cubes.extract(group_constraint).extract_cube(constraint)
                    cubes.remove(cube)

                da = cube_to_da(cube)
                da_hp = da_to_healpix(da, self.config['regrid_method'], group['name_map'], self.drop_vars, add_cyclic,
                                      regional, regional_chunks=chunks[-1])
                output = da_to_zarr(da_hp, self.config['zarr_store_url_tpl'], group_name, group)

                if self.use_dask_delayed:
                    outputs.append(output)

        if self.use_dask_delayed:
            logger.info('computing')
            dask.compute(*outputs)

        logger.info('completed')
        if task['donepath']:
            Path(task['donepath']).write_text(self.debug_log.getvalue())

    def coarsen_healpix_region(self, task):
        z = task['z']
        start_time_idx = task['start_time_idx']
        chunks = task['chunks']
        n_tgt_time_chunks = task['n_tgt_time_chunks']
        coarsen_healpix_region(z, start_time_idx, chunks, n_tgt_time_chunks)

        logger.info('completed')
        if task['donepath']:
            Path(task['donepath']).write_text(self.debug_log.getvalue())


def slurm_run(tasks, array_index):
    task = tasks[array_index]
    logger.debug(task)
    proc = DaskProcessUMFilesToZarrStore(processing_config[task['config_key']])
    if task['task_type'] == 'regrid':
        proc.process_task(task)
    elif task['task_type'] == 'create_empty_zarr_stores':
        proc.create_empty_zarr_stores(task)
    elif task['task_type'] == 'coarsen_healpix_region':
        proc.coarsen_healpix_region(task)
    else:
        raise Exception(f'unknown task type {task["task_type"]}')
    return proc


if __name__ == '__main__':
    logger.remove()
    custom_fmt = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                  "<level>{level: <8}</level> | "
                  "<blue>[{process.id}]</blue><cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(sys.stderr, level="TRACE", format=custom_fmt, colorize=not sys.argv[1].startswith('slurm'))
    filepath = Path(__file__)
    logger.debug('{} last edited: {:%Y-%m-%d %H:%M:%S}'.format(filepath.name,
                                                               dt.datetime.fromtimestamp(filepath.stat().st_mtime)))

    logger.debug(sys.argv)

    if sys.argv[1] == 'slurm':
        tasks_path = sys.argv[2]
        logger.info(tasks_path)
        with Path(tasks_path).open('r') as f:
            tasks = json.load(f)

        slurm_run(tasks, int(sys.argv[3]))
    elif sys.argv[1] == 'create_empty_zarr_stores':
        logger.info(sys.argv)
        config_key = sys.argv[2]
        test_date = sys.argv[3]
        config = processing_config[config_key]
        basedir = config['basedir']
        test_inpaths = [
            basedir / f'field.pp/apver{s}.pp/{config_key}.apver{s}_{test_date}.pp'
            for s in 'abcd'
        ]

        for path in test_inpaths:
            assert path.exists(), f'{path} does not exist'

        task = {
            'task_type': 'create_empty_zarr_stores',
            'config_key': config_key,
            'date': test_date,
            'inpaths': test_inpaths,
            'donepath': None,
        }
        slurm_run([task], 0)
    elif sys.argv[1] == 'regrid':
        config_key = sys.argv[2]
        test_date = sys.argv[3]
        config = processing_config[config_key]
        basedir = config['basedir']
        test_inpaths = [
            basedir / f'field.pp/apver{s}.pp/{config_key}.apver{s}_{test_date}.pp'
            for s in 'abcd'
        ]
        for path in test_inpaths:
            assert path.exists(), f'{path} does not exist'
        task = {
            'task_type': 'regrid',
            'config_key': config_key,
            'date': test_date,
            'inpaths': test_inpaths,
            'donepath': None,
        }
        slurm_run([task], 0)
    elif sys.argv[1] == 'coarsen_healpix_region':
        config_key = sys.argv[2]
        config = processing_config[config_key]
        chunks = config['2d']['chunks']
        task = {
            'task_type': 'coarsen_healpix_region',
            'config_key': config_key,
            'z': int(sys.argv[3]),
            'start_time_idx': int(sys.argv[4]),
            'chunks': chunks,
            'n_tgt_time_chunks': int(sys.argv[5]),
        }
        slurm_run([task], 0)
    elif sys.argv[1] == 'gen_weights':
        config_key = sys.argv[2]
        cube_name = sys.argv[3]

        test_date = '20200120T00'
        config = processing_config[config_key]
        basedir = config['basedir']

        cubes = iris.load(basedir / f'field.pp/apvera.pp/{config_key}.apvera_{test_date}.pp')
        cube = cubes.extract_cube(cube_name)
        da = xr.DataArray.from_iris(cube)
        chunks = config['groups']['2d']['chunks'][10]
        gen_weights(da, 10, weights_path=weights_filepath(da, 'longitude', 'latitude'), add_cyclic=False,
                    regional=True, regional_chunks=chunks[-1])
    else:
        raise Exception(f'Unknown command {sys.argv[1]}')
