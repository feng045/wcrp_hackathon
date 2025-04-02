import json
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path

import dask.array
import iris
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

sys.path.insert(0, '/home/users/mmuetz/deploy/global_hackathon_tools/dataset_transforms')
from um_latlon_pp_to_healpix_nc import UMLatLon2HealpixRegridder, gen_weights
from processing_config import processing_config

iris.FUTURE.date_microseconds = True

logger.remove()
logger.add(sys.stderr, level="TRACE")


def weights_filepath(da, lonname, latname):
    lon0, lonN = da[lonname].values[[0, -1]]
    lat0, latN = da[latname].values[[0, -1]]
    lonstr = str((lon0.item(), lonN.item(), len(da[lonname]))).replace(' ', '')
    latstr = str((lat0.item(), latN.item(), len(da[latname]))).replace(' ', '')
    return f'/gws/nopw/j04/hrcm/mmuetz/weights/regrid_weights_N2560_hpz10.cyclic_lon.lon={lonstr}.lat={latstr}.nc'


def find_main_time(ds):
    times = {name: pd.DatetimeIndex(ds[name].values)
             for name in ds.coords if name.startswith('time')}
    for name, time in times.items():
        if ((time.second == 0) & (time.minute == 0)).all():
            return name
    return None


def find_halfpast_time(ds):
    times = {name: pd.DatetimeIndex(ds[name].values)
             for name in ds.coords if name.startswith('time')}
    for name, time in times.items():
        if ((time.second == 0) & (time.minute == 30)).all():
            return name
    return None


class ProcessUMFilesToZarrStore:
    def __init__(self, config, save_nc=False):
        self.first_date = config['first_date']
        self.tmp_dir = config['tmp_dir']
        self.drop_vars = config['drop_vars']
        self.groups = config['groups']
        self.regrid_method = config['regrid_method']
        self.save_nc = save_nc

        self.all_ds = None
        self.all_hp_ds = None
        self.debug_log = StringIO()
        logger.add(self.debug_log)
        self.tmppaths = []

    def process_task(self, task):
        date = task['date']
        inpaths = task['inpaths']
        if task['donepath']:
            donepath = Path(task['donepath'])
        else:
            donepath = None

        if self.save_nc:
            all_cubes = self.load_cubes(inpaths)
            all_ds = {}
            for group, cubes in all_cubes.items():
                ds = xr.Dataset()
                for cube in cubes:
                    logger.debug(f'convert {cube.name()} from iris to xr')
                    ds[cube.name()] = xr.DataArray.from_iris(cube)
                all_ds[group] = ds
        else:
            need_to_gen_nc = False
            ncpaths = {}
            for group in self.groups.keys():
                tmppath = self.tmp_dir / f'{date}.{group}.nc'
                ncpaths[group] = tmppath
                if not tmppath.exists():
                    need_to_gen_nc = True
                self.tmppaths.append(tmppath)

            if need_to_gen_nc:
                all_cubes = self.load_cubes(inpaths)
                for group, cubes in all_cubes.items():
                    tmppath = self.tmp_dir / f'{date}.{group}.nc'
                    logger.info(f'save {group} {tmppath}')
                    tmppath.parent.mkdir(exist_ok=True, parents=True)
                    if not tmppath.exists():
                        for cube in cubes:
                            logger.debug(f'saving {cube.name()} to {group}')
                        iris.save(cubes, tmppath)

            all_ds = {}
            for group, path in ncpaths.items():
                # if group != '3d':
                #     continue
                logger.info(f'opening dataset {group}: {path}')
                all_ds[group] = xr.open_dataset(path, decode_timedelta=False)
                logger.trace(all_ds[group])
        self.all_ds = all_ds

        all_hp_ds = {}

        if False and date == self.first_date:
            # Gen weights if not already there.
            ds = all_ds['2d']
            for da in ds[['air_temperature', 'x_wind']].data_vars.values():
                lonname = [c for c in da.coords if c.startswith('longitude')][0]
                latname = [c for c in da.coords if c.startswith('latitude')][0]
                weights_path = weights_filepath(da, lonname, latname)
                if not Path(weights_path).exists():
                    logger.info(f'creating weights file for {da.name}')
                    logger.trace(da)
                    gen_weights(da, 10, lonname, latname, weights_path)

        for group, ds in all_ds.items():
            # if group != '3d':
            #     continue

            logger.info(f'creating {group} healpix dataset')
            logger.debug(ds.mass_fraction_of_cloud_ice_in_air.values.sum())
            hp_ds = self.regrid_to_hp(group, ds)

            logger.trace(hp_ds)
            logger.debug(hp_ds.cli.values.sum())
            all_hp_ds[group] = hp_ds

        for hp_ds_zmax in all_hp_ds.values():
            coarse_hp_ds = self.calc_coarse_hp_ds(hp_ds_zmax)

            if date == self.first_date:
                for zoom, hp_ds in coarse_hp_ds.items():
                    logger.info(f'creating zarr store for {zoom}')
                    self.create_empty_zarr_store(group, zoom, hp_ds)

            for zoom, hp_ds in coarse_hp_ds.items():
                logger.trace(hp_ds)
                logger.debug(hp_ds.cli.values.sum())
                self.populate_zarr_store(group, zoom, hp_ds)

        self.all_hp_ds = all_hp_ds
        # Delete self.tmppaths Paths here.

        if donepath:
            donepath.write_text(self.debug_log.getvalue())

        return all_hp_ds

    def load_cubes(self, inpaths):
        """load cubes from paths, combine into a large CubeList, then extract groups for further processing."""
        stream_cubes = {}

        for path in inpaths:
            stream_name = Path(path).parts[-2]
            logger.info(f'load {path}')
            cubes = iris.load(path)
            stream_cubes[stream_name] = cubes

        cubes = iris.cube.CubeList()
        for group, v in stream_cubes.items():
            cubes.extend(v)

        group_cubes = {}
        for group_name, group in self.groups.items():
            filtered_cubes = cubes.extract(group['constraint'])
            name_map = group['name_map']

            if 'extra_constraints' in group:
                extra_constraints = [group['extra_constraints'].get(n, n) for n in name_map.keys()]
                filtered_cubes = filtered_cubes.extract(extra_constraints)

            logger.debug(f'extracted {len(filtered_cubes)} cubes from {group_name}')
            logger.trace('\n'.join([str((c.name(), c.dtype)) for c in filtered_cubes]))
            if 'extra_processing' in self.groups[group_name]:
                for name, fn in self.groups[group_name]['extra_processing'].items():
                    logger.info(f'applying {fn} to {name}')
                    cube = cubes.extract_cube(name)
                    cube_proc = fn(cube, cubes)
                    filtered_cubes.remove(cube)
                    filtered_cubes.append(cube_proc)

            if group_name == '3d':
                filtered_cubes.remove(cubes.extract_cube('air_pressure'))
            group_cubes[group_name] = filtered_cubes
        return group_cubes

    def regrid_to_hp(self, group, ds):
        hp_ds = xr.Dataset()
        # regridders = {}
        data_vars_to_regrid = {}
        # Filter out data_vars not to be regridded.
        for name, da in ds.data_vars.items():
            # Is there a better/more robust way to do this?
            if 'latitude' not in da.coords and 'latitude_0' not in da.coords:
                continue
            if name in ['pressure']:
                continue
            data_vars_to_regrid[name] = da

        for i, (name, da) in enumerate(data_vars_to_regrid.items()):
            long_name, short_name = self.groups[group]['name_map'][name]
            logger.info(f'regridding {i + 1}/{len(data_vars_to_regrid)}: {name} -> {short_name} ({long_name})')
            da = da.compute()

            lonname = [c for c in da.coords if c.startswith('longitude')][0]
            latname = [c for c in da.coords if c.startswith('latitude')][0]

            if self.regrid_method == 'easygems_delaunay':
                weights_path = weights_filepath(da, lonname, latname)
                logger.debug(f'  - using weights: {weights_path}')
                # if weights_path not in regridders:
                #     regridders[weights_path] = UMLatLon2HealpixRegridder(method='easygems_delaunay', weights_path=weights_path)
                # regridder = regridders[weights_path]
                regridder = UMLatLon2HealpixRegridder(method='easygems_delaunay', weights_path=weights_path)
            else:
                regridder = UMLatLon2HealpixRegridder(method=self.regrid_method, weights_path=None)

            logger.debug(f'  - NaNs for time for orig data: {np.isnan(da.values).sum()}')
            hp_da = regridder.regrid(da, lonname, latname)
            logger.debug(f'  - NaNs for time for hp data: {np.isnan(hp_da.values).sum()}')
            if np.isnan(hp_da.values).all():
                raise Exception(f'{short_name} hp all values are NaN')
            hp_da.rename(short_name)
            hp_da.attrs['UM_name'] = name
            hp_da.attrs['long_name'] = long_name
            hp_da.attrs['grid_mapping'] = 'healpix_nested'
            hp_da.attrs['healpix_zoom'] = 10
            hp_ds[short_name] = hp_da

        drop_vars_exists = list(set(self.drop_vars) & set(k for k in hp_ds.coords.keys()))
        hp_ds = hp_ds.drop_vars(drop_vars_exists)

        return hp_ds

    @staticmethod
    def calc_coarse_hp_ds(hp_ds_zmax):
        # after it has run (and all zoom levels have been combined into datasets).
        # DONE: because I was renaming dims with different values to the same name!
        coarse_hp_ds = defaultdict(xr.Dataset)
        logger.trace(hp_ds_zmax)
        for name, da_zmax in hp_ds_zmax.data_vars.items():
            logger.info(f'coarsening {name}')
            coarse_das = UMLatLon2HealpixRegridder.coarsen(da_zmax)
            for zoom, da in coarse_das.items():
                coarse_hp_ds[zoom][name] = da
        logger.trace(coarse_hp_ds[10])
        return coarse_hp_ds

    def create_empty_zarr_store(self, group, zoom, hp_ds):
        time = self.groups[group]['time']
        time_half = self.groups[group]['time_half']

        ds_tpl = xr.Dataset()
        half_time = find_halfpast_time(hp_ds)
        for name, da in hp_ds.data_vars.items():
            timename = [c for c in da.coords if c.startswith('time')][0]
            reduced_dims = [d for d in da.dims if d != timename and not d.startswith('time')]
            # TODO: Chunks for different zooms.
            # TODO: Note, I can't chunk across time dim because I can't write across chunk boundaries.
            # TODO: I think this is an argument for saving to .nc then converting these after I've done all the conversion,
            # TODO: then I could rechunk how I wanted.
            if timename == half_time:
                zarr_time_name = 'time_halfpast'
                zarr_time = time_half
            else:
                zarr_time_name = 'time'
                zarr_time = time
            dims = [zarr_time_name] + reduced_dims
            if da.ndim == 2:
                coords = {zarr_time_name: zarr_time, 'cell': hp_ds.cell}
                chunks = (1, 4 ** 10)
            elif da.ndim == 3:
                coords = {zarr_time_name: zarr_time, 'pressure': hp_ds.pressure, 'cell': hp_ds.cell}
                chunks = (1, 5, 4 ** 10)
            else:
                raise Exception('ndim must be 2 or 3')
            dummies = dask.array.zeros((len(zarr_time), *da.shape[1:]), dtype=np.float32, chunks=chunks)

            da = da.rename(**{timename: zarr_time_name})
            da_tpl = xr.DataArray(dummies, dims=dims, coords=coords, name=name, attrs=da.attrs)
            ds_tpl[name] = da_tpl

        logger.debug(ds_tpl)
        ds_tpl.to_zarr(self.groups[group]['stores'][zoom], compute=False)

    def populate_zarr_store(self, group, zoom, hp_ds):
        half_time = find_halfpast_time(hp_ds)
        logger.info(f'saving data for {zoom}')
        ds_tpl = xr.open_zarr(self.groups[group]['stores'][zoom])
        for name, da in hp_ds.data_vars.items():
            timename = [c for c in da.coords if c.startswith('time')][0]
            if timename == half_time:
                zarr_time_name = 'time_halfpast'
            else:
                zarr_time_name = 'time'
            da = da.rename(**{timename: zarr_time_name})
            idx = np.argmin(np.abs(da[zarr_time_name].values[0] - ds_tpl[zarr_time_name].values))
            logger.debug(
                f'writing {name} to zarr store {self.groups[group]["stores"][zoom]} (idx={idx}, time={da[zarr_time_name].values[0]})')
            if np.isnan(da.values).all():
                logger.debug(da)
                raise Exception(f'da {da.name} is full of NaNs')
            if group == '2d':
                da.to_zarr(self.groups[group]['stores'][zoom],
                           region={zarr_time_name: slice(idx, idx + len(da[zarr_time_name])), 'cell': slice(None)})
            elif group == '3d':
                da.to_zarr(self.groups[group]['stores'][zoom],
                           region={zarr_time_name: slice(idx, idx + len(da[zarr_time_name])), 'pressure': slice(None),
                                   'cell': slice(None)})


def slurm_run(tasks_path, array_index, paths_per_job=10):
    print(tasks_path, array_index)
    with Path(tasks_path).open('r') as f:
        tasks = json.load(f)

    proc = None
    for task in tasks[array_index * paths_per_job: (array_index + 1) * paths_per_job]:
        if task['task_type'] == 'regrid':
            proc = ProcessUMFilesToZarrStore(processing_config['5km-RAL3'])
            proc.process_task(task)
    return proc


if __name__ == '__main__':
    if sys.argv[1] == 'test':
        config_key = sys.argv[2]
        test_date = sys.argv[3]
        test_inpaths = [
            f'/gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3/glm/field.pp/apvera.pp/glm.n2560_RAL3p3.apvera_{test_date}.pp',
            f'/gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3/glm/field.pp/apverb.pp/glm.n2560_RAL3p3.apverb_{test_date}.pp',
            f'/gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3/glm/field.pp/apverc.pp/glm.n2560_RAL3p3.apverc_{test_date}.pp',
            f'/gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3/glm/field.pp/apverd.pp/glm.n2560_RAL3p3.apverd_{test_date}.pp'
        ]
        task = {
            'task_type': 'regrid',
            'date': test_date,
            'inpaths': test_inpaths,
            'donepath': None,
        }

        proc = ProcessUMFilesToZarrStore(processing_config[config_key])
        proc.process_task(task)
    elif sys.argv[1] == 'slurm':
        slurm_run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
