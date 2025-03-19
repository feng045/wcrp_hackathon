from functools import partial
from pathlib import Path

import iris
import iris.cube
import numpy as np
import pandas as pd
import s3fs
import stratify
import xarray as xr
import xgcm
from iris.experimental.stratify import relevel
from loguru import logger

TMPDIR = Path('/work/scratch-nopw2/mmuetz/wcrp_hackathon/')

s3cfg = dict([l.split(' = ') for l in Path('/home/users/mmuetz/.s3cfg').read_text().split('\n') if l])
jasmin_s3 = s3fs.S3FileSystem(
    anon=False,
    secret=s3cfg['secret_key'],
    key=s3cfg['access_key'],
    client_kwargs={'endpoint_url': 'http://hackathon-o.s3.jc.rl.ac.uk'}
)


def has_dimensions(*dims):
    """Returns an Iris constraint that filters cubes based on dimensions."""

    def dim_filter(cube):
        cube_dims = [c.name() for c in cube.dim_coords]
        return set(cube_dims) == set(dims)

    return iris.Constraint(cube_func=dim_filter)


def cube_cell_method_is_not_empty(cube):
    return cube.cell_methods != tuple()


def iris_relevel_model_level_to_pressure(cube, cubes):
    logger.debug(f're-level model level to {cube}')
    p = cubes.extract_cube('air_pressure')
    cube = cube[-p.shape[0]:]
    assert (p.coord('time').points == cube.coord('time').points).all()

    z = cubes.extract_cube('geopotential_height')
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


def xgcm_model_level_to_pressure(da: xr.DataArray, ds: xr.Dataset):
    """No longer used - using iris.experimental.stratify on cubes instead"""
    raise NotImplemented
    da_p = ds.air_pressure
    # Align da and da_p on time dimension.
    tname1 = [c for c in da.coords if c.startswith('time')][0]
    tname2 = [c for c in da_p.coords if c.startswith('time')][0]
    t1 = da[tname1]
    t2 = da_p[tname2]
    assert len(t1) > len(t2)
    idx = np.argmin(np.abs(t1.values - t2.values[0]))
    # print(idx)
    s = slice(idx, idx + len(t2))
    assert (t1[s].values == t2.values).all()

    # Slice da to match on time dim.
    da = da.isel(**{tname1: s})
    # Rename time dim on da_p and convert Pa to hPa.
    da_p = da_p.rename(**{tname2: tname1})
    da_p.values /= 100

    pressure_values = ds.pressure.values
    ds_p = da_p.to_dataset(name='pressure')
    logger.trace(da_p)
    logger.trace(da)

    # xgcm magic.
    grid = xgcm.Grid(ds_p, coords=dict(pressure={'center': 'model_level_number'}), periodic=False)
    da_pressure_levels = grid.transform(
        da,
        'pressure',
        pressure_values,
        target_data=da_p,
        method='linear',
    )
    # Needed to rename the coordinate to the same as in ohter DataArrays.
    da_pressure_levels = da_pressure_levels.rename(air_pressure='pressure')
    logger.trace(da_pressure_levels)
    return da_pressure_levels


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
    'toa_outgoing_shortwave_flux_assuming_clear_sky': ('toa_outgoing_shortwave_flux_clear_sky', 'rsutcs'),
}
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
    # TODO: This is in height coords, not pressure/model levels. What to do?
    # 'upward_air_velocity': ('upward_air_velocity', 'wa'),
    'air_pressure': ('air_pressure', 'p'),
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

processing_config = {
    '5km-RAL3': {
        'first_date': '20200120T00',
        'tmp_dir': TMPDIR / 'DYAMOND3_example_data/5km-RAL3',
        'drop_vars': drop_vars,
        'regrid_method': 'easygems_delaunay',
        'groups': {
            '2d': {
                'name_map': name_map_2d,
                'constraint': has_dimensions("time", "latitude", "longitude"),
                'extra_constraints': {
                    'stratiform_rainfall_flux': iris.Constraint(name='stratiform_rainfall_flux') & iris.Constraint(
                        cube_func=cube_cell_method_is_not_empty),
                    'stratiform_snowfall_flux': iris.Constraint(name='stratiform_snowfall_flux') & iris.Constraint(
                        cube_func=cube_cell_method_is_not_empty),
                    'toa_outgoing_shortwave_flux': iris.Constraint(
                        name='toa_outgoing_shortwave_flux') & iris.AttributeConstraint(
                        STASH='m01s01i208'),
                },
                'stores': {zoom: s3fs.S3Map(
                    root=f's3://sim-data/DYAMOND3_example_data/5km-RAL3/2d/data.2d.z{zoom}.zarr',
                    s3=jasmin_s3, check=False)
                    for zoom in range(11)},
                # Date range as per Dasha's message.
                'time': pd.date_range('2020-01-20', '2021-04-01', freq='h'),
                'time_half': pd.date_range('2020-01-20 00:30', '2021-04-01', freq='h'),
            },
            '3d': {
                'name_map': name_map_3d,
                'constraint': [has_dimensions("time", "pressure", "latitude", "longitude"),
                               has_dimensions("time", "model_level_number", "latitude", "longitude")],
                'extra_constraints': {
                    'relative_humidity': iris.Constraint(name='relative_humidity') & iris.AttributeConstraint(
                        STASH='m01s16i256'),
                },
                'stores': {zoom: s3fs.S3Map(
                    root=f's3://sim-data/DYAMOND3_example_data/5km-RAL3/3d/data.full.v2.3d.z{zoom}.zarr',
                    s3=jasmin_s3, check=False)
                    for zoom in range(11)},
                'extra_processing': {
                    # Convert these fields from model level to pressure as vertical coord.
                    'mass_fraction_of_cloud_ice_in_air': iris_relevel_model_level_to_pressure,
                    'mass_fraction_of_cloud_liquid_water_in_air': iris_relevel_model_level_to_pressure,
                    'mass_fraction_of_graupel_in_air': iris_relevel_model_level_to_pressure,
                    'mass_fraction_of_rain_in_air': iris_relevel_model_level_to_pressure,
                    'mass_fraction_of_cloud_ice_crystals_in_air': iris_relevel_model_level_to_pressure,
                },
                # Date range as per Dasha's message, but 3 hourly.
                'time': pd.date_range('2020-01-20', '2021-04-01', freq='3h'),
                'time_half': pd.date_range('2020-01-20 00:30', '2021-04-01', freq='3h'),
            },
        }
    },
}

processing_config['5km-RAL3-e2g'] = processing_config['5km-RAL3'].copy()
processing_config['5km-RAL3-e2g']['regrid_method'] = 'earth2grid'
processing_config['5km-RAL3-e2g']['groups']['2d']['stores'] = {
    zoom: s3fs.S3Map(
        root=f's3://sim-data/DYAMOND3_example_data/5km-RAL3/2d/data.2d.z{zoom}.earth2grid.zarr',
        s3=jasmin_s3, check=False)
    for zoom in range(11)}
