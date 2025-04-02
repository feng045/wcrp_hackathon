from pathlib import Path

import iris
import iris.cube
import pandas as pd

TMPDIR = Path('/work/scratch-nopw2/mmuetz/wcrp_hackathon/')


def has_dimensions(*dims):
    """Returns an Iris constraint that filters cubes based on dimensions."""

    def dim_filter(cube):
        cube_dims = tuple([c.name() for c in cube.dim_coords])
        return cube_dims == dims

    return iris.Constraint(cube_func=dim_filter)


def cube_cell_method_is_not_empty(cube):
    return cube.cell_methods != tuple()


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
    # Changed from rldt as per tobi's message on Mattermost.
    'surface_downwelling_longwave_flux_in_air': ('surface_downwelling_longwave_flux_in_air', 'rlds'),
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
    'y_wind': ('northtward_wind', 'va'),
    'relative_humidity': ('relative_humidity', 'hur'),
    'specific_humidity': ('specific_humidity', 'hus'),
    'air_temperature': ('temperature', 'ta'),
    'upward_air_velocity': ('upward_air_velocity', 'wa'),
}

name_map_3d_ml = {
    'mass_fraction_of_cloud_ice_in_air': ('mass_fraction_of_cloud_ice_in_air', 'cli'),
    'mass_fraction_of_cloud_liquid_water_in_air': ('mass_fraction_of_cloud_liquid_water_in_air', 'clw'),
    'mass_fraction_of_graupel_in_air': ('mass_fraction_of_graupel_in_air', 'qg'),
    'mass_fraction_of_rain_in_air': ('mass_fraction_of_rain_in_air', 'qr'),
    'mass_fraction_of_cloud_ice_crystals_in_air': ('mass_fraction_of_snow_water_in_air', 'qs'),
}

chunks2d = {
    # z10 has to have no chunking over time.
    10: (1, 4 ** 10),  # 12 chunks per time.
    # Increase temporal at same rate as reducing spatial.
    9: (4, 4 ** 9),
    8: (4 ** 2, 4 ** 8),
    7: (4 ** 3, 4 ** 7),
    # Transition to 3 chunks per time.
    6: (4 ** 3, 4 ** 7),
    # Transition to 1 chunk per time.
    5: (4 ** 3, 12 * 4 ** 5),
    4: (4 ** 4, 12 * 4 ** 4),
    3: (4 ** 5, 12 * 4 ** 3),
    2: (4 ** 6, 12 * 4 ** 2),
    1: (4 ** 7, 12 * 4 ** 1),
    0: (4 ** 8, 12 * 4 ** 0),
}

chunks3d = {
    # z10 has to have no chunking over time.
    10: (1, 5, 4 ** 9),  # 48 chunks per time.
    9: (4, 5, 4 ** 8),
    8: (4 ** 2, 5, 4 ** 7),
    7: (4 ** 3, 5, 4 ** 6),
    # transition to fewer chunks per time.
    6: (4 ** 3, 5, 4 ** 6),  # 12 chunks per time
    5: (4 ** 3, 5, 12 * 4 ** 4),
    # transition to 1 chunk per time.
    4: (4 ** 3, 5, 12 * 4 ** 4),
    3: (4 ** 4, 5, 12 * 4 ** 3),
    # increase number pressure levels.
    2: (4 ** 4, 25, 12 * 4 ** 2),
    1: (4 ** 5, 25, 12 * 4 ** 1),
    0: (4 ** 6, 25, 12 * 4 ** 0),
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
    'surface_altitude',
    'level_height',
    'sigma',
    'altitude',
]

time2d = pd.date_range('2020-01-20', '2021-04-01', freq='h')
time3d = pd.date_range('2020-01-20', '2021-04-01', freq='3h')

vn5kmRAL3 = 'v1'

processing_config = {
    '5km-RAL3': {
        'basedir': Path('/gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3'),
        'donedir': Path('/gws/nopw/j04/hrcm/mmuetz/slurm_done'),
        'donepath_tpl': f'5km-RAL3/{{task}}_{{date}}.{vn5kmRAL3}.done',
        'first_date': '20200120T00',
        'zarr_store_url_tpl': f's3://sim-data/5km-RAL3/dev/data.{{name}}.{vn5kmRAL3}.z{{zoom}}.zarr',
        'drop_vars': drop_vars,  # TODO: still needed?
        'regrid_method': 'easygems_delaunay',
        'groups': {
            '2d': {
                'time': time2d,
                'zarr_store': '2d',
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
                'extra_attrs': {
                    'stratiform_rainfall_flux': {'notes':
                                                 'hourly mean - time index shifted from half past the hour to the following hour'},
                    'stratiform_snowfall_flux': {'notes':
                                                 'hourly mean - time index shifted from half past the hour to the following hour'},
                },
                'chunks': chunks2d,
            },
            '3d': {
                'time': time3d,
                'zarr_store': '3d',
                'name_map': name_map_3d,
                'constraint': has_dimensions("time", "pressure", "latitude", "longitude"),
                'extra_constraints': {
                    'relative_humidity': iris.Constraint(name='relative_humidity') & iris.AttributeConstraint(
                        STASH='m01s16i256'),
                },
                'extra_attrs': {},
                'chunks': chunks3d,
            },
            '3d_ml': {
                'time': time3d,
                'zarr_store': '3d',
                'name_map': name_map_3d_ml,
                'constraint': has_dimensions("time", "model_level_number", "latitude", "longitude"),
                'extra_constraints': {},
                'extra_attrs': {n: {
                    'notes': 'interpolated from UM model levels to pressure levels using iris.experimental.stratify'}
                                for n in name_map_3d_ml.keys()},
                'chunks': chunks3d,
                'interpolate_model_levels_to_pressure': True,
            },
        }
    },
}

# processing_config['5km-RAL3-e2g'] = processing_config['5km-RAL3'].copy()
# processing_config['5km-RAL3-e2g']['regrid_method'] = 'earth2grid'
# processing_config['5km-RAL3-e2g']['groups']['2d']['stores'] = {
#     zoom: s3fs.S3Map(
#         root=f's3://sim-data/DYAMOND3_example_data/5km-RAL3/2d/data.2d.z{zoom}.earth2grid.zarr',
#         s3=jasmin_s3, check=False)
#     for zoom in range(11)}
#
