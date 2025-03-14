import json
import sys
from pathlib import Path

import dask.array
import iris
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from jedi.inference.gradual.annotation import find_type_from_comment_hint_for

sys.path.insert(0, '/home/users/mmuetz/deploy/global_hackathon_tools/dataset_transforms')
from um_latlon_pp_to_healpix_nc import UMHealpixRegridder

s3cfg = dict([l.split(' = ') for l in Path('/home/users/mmuetz/.s3cfg').read_text().split('\n') if l])
jasmin_s3 = s3fs.S3FileSystem(
    anon=False,
    secret=s3cfg['secret_key'],
    key=s3cfg['access_key'],
    client_kwargs={'endpoint_url': 'http://hackathon-o.s3.jc.rl.ac.uk'}
)
store = s3fs.S3Map(root='s3://sim-data/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3/data.2d.z10.zarr',
                   s3=jasmin_s3, check=False)

iris.FUTURE.date_microseconds = True

name_map_3d = {
    'x_wind': ('eastward_wind', 'ua'),
    'geopotential_height': ('geopotential height', 'zg'),
    'mass_fraction_of_cloud_ice_in_air': ('mass_fraction_of_cloud_ice_in_air', 'cli'),
    'mass_fraction_of_cloud_liquid_water_in_air': ('mass_fraction_of_cloud_liquid_water_in_air', 'clw'),
    'mass_fraction_of_graupel_in_air': ('mass_fraction_of_graupel_in_air', 'qg'),
    'mass_fraction_of_rain_in_air': ('mass_fraction_of_rain_in_air', 'qr'),
    'mass_fraction_of_cloud_ice_crystals_in_air': ('mass_fraction_of_snow_water_in_air', 'qs'),
    'y_wind': ('northtward_wind', 'va'),
    'relative_humidity': ('relative_humidity', 'hur'),
    'specific_humidity': ('specific_humidity', 'hus'),
    'air_temperature': ('temperature', 'ta'),
    'upward_air_velocity': ('upward_air_velocity', 'wa')
}

name_map_2d = {
    'air_pressure_at_sea_level': ('air_pressure_at_mean_sea_level', 'psl'),
    'air_temperature': ('air_temperature', 'tas'),
    'atmosphere_cloud_liquid_water_content': ('atmosphere_mass_content_of_cloud_condensed_water', 'clwvi'),
    'atmosphere_cloud_ice_content': ('atmosphere_mass_content_of_cloud_ice', 'clivi'),
    'm01s30i461': ('atmosphere_mass_content_of_water_vapor', 'prw'),
    'cloud_area_fraction_assuming_maximum_random_overlap': ('cloud_area_fraction', 'clt'),
    'x_wind': ('eastward_wind', 'uas'),
    'y_wind': ('northward_wind', 'vas'),
    'stratiform_rainfall_flux': ('precipitation_flux', 'pr'),
    'stratiform_snowfall_flux': ('solid_precipitation_flux', 'prs'),
    'specific_humidity': ('specific_humidity', 'huss'),
    'surface_air_pressure': ('surface_air_pressure', 'ps'),
    'surface_upward_latent_heat_flux': ('surface_downward_latent_heat_flux', 'hflsd'),
    'surface_upward_sensible_heat_flux': ('surface_downward_sensible_heat_flux', 'hfssd'),
    'surface_downwelling_longwave_flux_in_air': ('surface_downwelling_longwave_flux_in_air', 'rldt'),
    'surface_downwelling_longwave_flux_in_air_assuming_clear_sky': (
        'surface_downwelling_longwave_flux_in_air_clear_sky', 'rldscs'),
    'surface_downwelling_shortwave_flux_in_air': ('surface_downwelling_shortwave_flux_in_air', 'rsds'),
    'surface_downwelling_shortwave_flux_in_air_assuming_clear_sky': (
        'surface_downwelling_shortwave_flux_in_air_clear_sky', 'rsdscs'),
    'surface_temperature': ('surface_temperature', 'ts'),
    'toa_incoming_shortwave_flux': ('toa_incoming_shortwave_flux', 'rsdt'),
    'toa_outgoing_longwave_flux': ('toa_outgoing_longwave_flux', 'rlut'),
    'toa_outgoing_longwave_flux_assuming_clear_sky': ('toa_outgoing_longwave_flux_clear_sky', 'rlutcs'),
    'toa_outgoing_shortwave_flux': ('toa_outgoing_shortwave_flux', 'rsut'),
    'toa_outgoing_shortwave_flux_assuming_clear_sky': ('toa_outgoing_shortwave_flux_clear_sky', 'rsutcs')
}
drop_vars = [
    'latitude_0',
    'longitude_0',
    # 'bnds',  # gets dropped automatically and causes problems if you try to drop it.
    'forecast_period',
    'forecast_reference_time',
    'forecast_period_0',
    'height',
    'height_0',
    'forecast_period_1',
    'forecast_period_2',
    'forecast_period_3',
    'latitude_longitude',
    'time_1_bnds',
    'time_0_bnds',
    'forecast_period_1_bnds',
]

first_date = '20200120T00'

def has_dimensions(*dims):
    """Returns an Iris constraint that filters cubes based on dimensions."""
    def dim_filter(cube):
        cube_dims = [c.name() for c in cube.dim_coords]
        return set(cube_dims) == set(dims)
    return iris.Constraint(cube_func=dim_filter)

def cube_cell_method_is_not_empty(cube):
    return cube.cell_methods != tuple()


cube_groups = {
    '2d': {
        'name_map': name_map_2d,
        'constraint': has_dimensions("time", "latitude", "longitude"),
        'extra_constraints': {
            'stratiform_rainfall_flux': iris.Constraint(name='stratiform_rainfall_flux') & iris.Constraint(
                cube_func=cube_cell_method_is_not_empty),
            'stratiform_snowfall_flux': iris.Constraint(name='stratiform_snowfall_flux') & iris.Constraint(
                cube_func=cube_cell_method_is_not_empty),
            'toa_outgoing_shortwave_flux': iris.Constraint(name='toa_outgoing_shortwave_flux') & iris.AttributeConstraint(
                STASH='m01s01i208'),
        },
        'drop_vars': drop_vars,
    },
    '3d': {
        'name_map': name_map_3d,
        # 'constraint': [has_dimensions("time", "pressure", "latitude", "longitude"), has_dimensions("height", "model_level_number", "latitude", "longitude")],
        'constraint': has_dimensions("time", "pressure", "latitude", "longitude"),
        'extra_constraints': {
            'specific_humidity': iris.Constraint(name='specific_humidity') & iris.AttributeConstraint(STASH='m01s16i256'),
        }
    },
}

TMPDIR = Path('/work/scratch-nopw2/mmuetz/wcrp_hackathon/')


def load_cubes(inpaths):
    """load cubes from paths, combine into a large CubeList, then extract groups for further processing."""
    all_cubes = {}

    for path in inpaths:
        stream_name = Path(path).parts[-2]
        print(f'load {path}')
        cubes = iris.load(path)
        all_cubes[stream_name] = cubes

    cubes = iris.cube.CubeList()
    for k, v in all_cubes.items():
        cubes.extend(v)

    all_cubes = {}
    for group_name, cube_group in cube_groups.items():
        filtered_cubes = cubes.extract(cube_group['constraint'])
        if 'extra_constraints' in cube_group:
            extra_constraints = [cube_group['extra_constraints'].get(n, n) for n in name_map_2d.keys()]
            filtered_cubes = filtered_cubes.extract(extra_constraints)
            all_cubes[group_name] = filtered_cubes
    return all_cubes


def weights_filepath(da, lonname, latname):
    lon0, lonN = da[lonname].values[[0, -1]]
    lat0, latN = da[latname].values[[0, -1]]
    lonstr = str((lon0.item(), lonN.item(), len(da[lonname]))).replace(' ', '')
    latstr = str((lat0.item(), latN.item(), len(da[latname]))).replace(' ', '')
    return f'/gws/nopw/j04/hrcm/mmuetz/weights/regrid_weights_N2560_hpz10.lon={lonstr}.lat={latstr}.nc'


def find_main_time(ds):
    times = {name: pd.DatetimeIndex(ds[name].values)
             for name in ds.coords if name.startswith('time')}
    for name, time in times.items():
        if ((time.second == 0) & (time.minute == 0)).all():
            return name
    raise Exception('No time found')

def find_halfpast_time(ds):
    times = {name: pd.DatetimeIndex(ds[name].values)
             for name in ds.coords if name.startswith('time')}
    for name, time in times.items():
        if ((time.second == 0) & (time.minute == 30)).all():
            return name
    raise Exception('No time found')


def create_zarr_store(hp_ds):
    time = pd.date_range('2020-01-20', '2021-03-01', freq='h')
    time_half = pd.date_range('2020-01-20 00:30', '2021-03-01', freq='h')
    ds_tpl = xr.Dataset()
    half_time = find_halfpast_time(hp_ds)
    for name, da in hp_ds.data_vars.items():
        timename = [c for c in da.coords if c.startswith('time')][0]
        reduced_dims = [d for d in da.dims if d != timename]
        if timename == half_time:
            dims = ['time_halfpast'] + reduced_dims
            coords = {'time_halfpast': time_half, 'cell': hp_ds.cell}
            dummies = dask.array.zeros((len(time_half), *da.shape[1:]), chunks=(1, 4 ** 10))
        else:
            dims = ['time'] + reduced_dims
            coords = {'time': time, 'cell': hp_ds.cell}
            dummies = dask.array.zeros((len(time), *da.shape[1:]), chunks=(1, 4 ** 10))
        # long_name, short_name = name_map_2d[name]
        da_tpl = xr.DataArray(dummies, dims=dims, coords=coords, name=name, attrs=da.attrs)

        ds_tpl[name] = da_tpl

    print(ds_tpl)
    ds_tpl.to_zarr(store, compute=False)


def process_task(date, inpaths, donepath):
    need_to_gen_nc = False
    ncpaths = {}
    for k in cube_groups.keys():
        tmppath = TMPDIR / 'DYAMOND3_example_data/5km-RAL3' / f'{date}.{k}.nc'
        ncpaths[k] = tmppath
        if not tmppath.exists():
            need_to_gen_nc = True

    if need_to_gen_nc:
        all_cubes = load_cubes(inpaths)
        for k, cubes in all_cubes.items():
            tmppath = TMPDIR / 'DYAMOND3_example_data/5km-RAL3' / f'{date}.{k}.nc'
            print(f'save {k} {tmppath}')
            tmppath.parent.mkdir(exist_ok=True, parents=True)
            if not tmppath.exists():
                iris.save(cubes, tmppath)

    all_ds = {}
    for k, path in ncpaths.items():
        all_ds[k] = xr.open_dataset(path, decode_timedelta=False)

    all_hp_ds = {}

    for k, ds in all_ds.items():
        if k != '2d':
            continue

        ds = ds.compute()
        print('creating healpix dataset')
        hp_ds = xr.Dataset()

        regridders = {}
        for name, da in ds.data_vars.items():
            if 'latitude' not in da.coords and 'latitude_0' not in da.coords:
                continue

            lonname = [c for c in da.coords if c.startswith('longitude')][0]
            latname = [c for c in da.coords if c.startswith('latitude')][0]

            weights_path = weights_filepath(da, lonname, latname)
            print(f'using weights: {weights_path}')
            if weights_path not in regridders:
                regridders[weights_path] = UMHealpixRegridder(method='easygems_delaunay', weights_path=weights_path)
            regridder = regridders[weights_path]

            print(f'regrid {name}')
            hp_da = regridder.regrid(da, lonname, latname)
            long_name, short_name = name_map_2d[name]
            hp_da.rename(short_name)
            hp_da.attrs['long_name'] = long_name
            hp_da.attrs['grid_mapping'] = 'healpix_nested'
            hp_da.attrs['healpix_zoom'] = 10
            hp_ds[short_name] = hp_da

        drop_vars_exists = list(set(drop_vars) & set(k for k in hp_ds.coords.keys()))
        # raise Exception(f'dropped {drop_vars_exists}')
        # print(drop_vars_exists)
        # print(list(hp_ds.keys()))
        hp_ds = hp_ds.drop_vars(drop_vars_exists)

        print(hp_ds)
        all_hp_ds[k] = hp_ds

    for hp_ds in all_hp_ds.values():
        if date == first_date:
            create_zarr_store(hp_ds)
        # for name, da in hp_ds.data_vars.items():
        #     da.to_zarr(store, region={'time': slice(da.shape[0]), 'cell': slice(None)})
        ds_tpl = xr.open_zarr(store)
        # main_time = find_main_time(hp_ds)
        half_time = find_halfpast_time(hp_ds)
        # hp_ds = hp_ds.drop_vars(['forecast_reference_time', 'height', 'height_0', 'forecast_period_0', 'forecast_period_1', 'forecast_period_2', 'forecast_period_3'])
        for name, da in hp_ds.data_vars.items():
            timename = [c for c in da.coords if c.startswith('time')][0]
            if timename == half_time:
                zarr_time_name = 'time_halfpast'
            else:
                zarr_time_name = 'time'
            da = da.rename(**{timename: zarr_time_name})
            idx = np.argmin(np.abs(da[zarr_time_name].values[0] - ds_tpl[zarr_time_name].values))
            print(f'writing {name} to zarr store')
            print(da)
            da.to_zarr(store, region={zarr_time_name: slice(idx, idx + len(da[zarr_time_name])), 'cell': slice(None)})

    return all_hp_ds


def main(input_output_files, array_index, paths_per_job=10):
    print(input_output_files, array_index)
    with Path(input_output_files).open('r') as f:
        todo_dates_to_paths = json.load(f)

    for date, inpaths, donepath in todo_dates_to_paths[array_index * paths_per_job: (array_index + 1) * paths_per_job]:
        all_hp_ds = process_task(date, inpaths, donepath)
    return all_hp_ds

# date = '20200120T00'
# inpaths = """/gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3/glm/field.pp/apvera.pp/glm.n2560_RAL3p3.apvera_20200120T00.pp
# /gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3/glm/field.pp/apverb.pp/glm.n2560_RAL3p3.apverb_20200120T00.pp
# /gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3/glm/field.pp/apverc.pp/glm.n2560_RAL3p3.apverc_20200120T00.pp
# /gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3/glm/field.pp/apverd.pp/glm.n2560_RAL3p3.apverd_20200120T00.pp""".split('\n')

date = '20200120T12'
inpaths = """/gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3/glm/field.pp/apvera.pp/glm.n2560_RAL3p3.apvera_20200120T12.pp
/gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3/glm/field.pp/apverb.pp/glm.n2560_RAL3p3.apverb_20200120T12.pp
/gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3/glm/field.pp/apverc.pp/glm.n2560_RAL3p3.apverc_20200120T12.pp
/gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3/glm/field.pp/apverd.pp/glm.n2560_RAL3p3.apverd_20200120T12.pp""".split('\n')


if __name__ == '__main__':
    # main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    all_hp_ds = process_task(date, inpaths, '')
