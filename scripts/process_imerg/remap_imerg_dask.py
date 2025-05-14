"""
Lat/Lon to Healpix Remapper

This script converts NetCDF data from lat/lon grid to healpix grid and saves it to zarr format.
"""
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import sys
import os
import argparse
from loguru import logger
from glob import glob

# Add path to modules containing regridding functions
sys.path.insert(0, '../process_um_data')
from um_latlon_pp_to_healpix_nc import gen_weights, UMLatLon2HealpixRegridder

import dask
import zarr
import asyncio
import concurrent.futures
from dask.distributed import Client, LocalCluster

# chunks2d = {
#     9: (1, 12 * 4**9),
#     8: (1, 12 * 4**8),
#     7: (4, 12 * 4**7),
#     6: (4**2, 12 * 4**6),
#     5: (4**3, 12 * 4**5),
#     4: (4**4, 12 * 4**4),
#     3: (4**5, 12 * 4**3),
#     2: (4**6, 12 * 4**2),
#     1: (4**7, 12 * 4),
#     0: (4**8, 12),
# }

def create_weights_file(da, zoom_level, output_dir=None, lonname='lon', latname='lat', 
                       add_cyclic=True, regional=True, regional_chunks=None):
    """
    Create a weight file for remapping from lat/lon grid to healpix.
    
    Parameters:
    -----------
    da : xarray.DataArray
        Input data array with lat/lon coordinates
    zoom_level : int
        Healpix zoom level
    output_dir : str or Path, optional
        Directory to save weight file
    lonname : str, optional
        Name of longitude dimension in da
    latname : str, optional
        Name of latitude dimension in da
    add_cyclic : bool, optional
        Whether to add cyclic point
    regional : bool, optional
        Whether this is a regional grid
    regional_chunks : int, optional
        Chunk size for regional grids
        
    Returns:
    --------
    Path: Path to the weight file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path(os.environ.get('WEIGHT_DIR', '.')) / 'weights'
    else:
        output_dir = Path(output_dir)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the weight filename
    lon0, lonN = da[lonname].values[[0, -1]]
    lat0, latN = da[latname].values[[0, -1]]
    lonstr = f'({lon0.item():.3f},{lonN.item():.3f},{len(da[lonname])})'
    latstr = f'({lat0.item():.3f},{latN.item():.3f},{len(da[latname])})'
    
    # Convert booleans to lowercase strings without quotes
    add_cyclic_str = str(add_cyclic).lower()
    regional_str = str(regional).lower()
    
    filename = f'regrid_weights.hpz{zoom_level}.cyclic_lon={add_cyclic_str}.regional={regional_str}.lon={lonstr}.lat={latstr}.nc'
    weights_path = output_dir / filename
    
    # Generate weights if they don't already exist
    if not weights_path.exists():
        logger.info(f'No weights file found at {weights_path}, generating new weights')
        gen_weights(da, zoom_level, lonname, latname, 
                   add_cyclic=add_cyclic, 
                   regional=regional,
                   regional_chunks=regional_chunks, 
                   weights_path=weights_path)
        logger.info(f'Weight file created: {weights_path}')
    else:
        logger.info(f'Weight file already exists: {weights_path}')
        
    return weights_path


def remap_to_healpix(da, zoom_level, weights_path=None, output_dir=None, 
                     lonname='lon', latname='lat', add_cyclic=True, 
                     regional=True, regional_chunks=None):
    """
    Remap a data array from lat/lon grid to healpix grid.
    
    Parameters:
    -----------
    da : xarray.DataArray
        Input data array with lat/lon coordinates
    zoom_level : int
        Healpix zoom level
    weights_path : str or Path, optional
        Path to weights file. If None, weights will be created.
    output_dir : str or Path, optional
        Directory to save weight file if weights_path is None
    lonname : str, optional
        Name of longitude dimension in da
    latname : str, optional
        Name of latitude dimension in da
    add_cyclic : bool, optional
        Whether to add cyclic point
    regional : bool, optional
        Whether this is a regional grid
    regional_chunks : int, optional
        Chunk size for regional grids
        
    Returns:
    --------
    xarray.DataArray: Remapped data array on healpix grid
    """
    # Create weights file if not provided
    if weights_path is None:
        weights_path = create_weights_file(
            da, zoom_level, output_dir, lonname, latname, 
            add_cyclic, regional, regional_chunks
        )

    # Create regridder
    logger.info(f'Using weights file: {weights_path}')
    regridder = UMLatLon2HealpixRegridder(
        method='easygems_delaunay', 
        zoom_level=zoom_level, 
        weights_path=weights_path,
        add_cyclic=add_cyclic, 
        regional=regional, 
        regional_chunks=regional_chunks
    )

    # Remap the data array
    logger.info(f'Remapping {da.name} to healpix zoom level {zoom_level}')
    da_hp = regridder.regrid(da, lonname, latname)
    
    return da_hp


async def async_write_to_zarr(da, zarr_store, region=None, retries=3, timeout=300):
    """
    Write a data array to zarr with retries and timeout.
    
    Parameters:
    -----------
    da : xarray.DataArray
        Data array to write to zarr
    zarr_store : str or zarr.storage
        Zarr store to write to
    region : dict, optional
        Region to write to in the zarr store
    retries : int, optional
        Number of retries
    timeout : int, optional
        Timeout in seconds
    """
    retry = 0
    while retry < retries:
        try:
            # Create a future to run the write operation with a timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(da.to_zarr, zarr_store, region=region)
                future.result(timeout=timeout)
            logger.info(f'Successfully wrote {da.name} to zarr store')
            return
        except (zarr.errors.ZarrError, concurrent.futures.TimeoutError) as e:
            retry += 1
            logger.warning(f'Failed to write to zarr (attempt {retry}/{retries}): {e}')
            await asyncio.sleep(2**retry)  # Exponential backoff
    
    logger.error(f'Failed to write {da.name} to zarr store after {retries} attempts')
    raise Exception(f'Failed to write {da.name} to zarr store')


def write_dataset_to_zarr(ds, store_path, chunks=None, parallel=False, n_workers=4, 
                          memory_limit='4GB'):
    """
    Write a dataset to zarr, optionally using dask for parallel processing.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset to write to zarr
    store_path : str
        Path to zarr store
    chunks : dict, optional
        Chunks for dask array
    parallel : bool, optional
        Whether to use dask for parallel processing
    n_workers : int, optional
        Number of workers for dask cluster
    memory_limit : str, optional
        Memory limit per worker
        
    Returns:
    --------
    str: Path to the zarr store
    """
    # Add healpix attributes for proper visualization
    if 'cell' in ds.dims:
        zoom_level = int(np.log2(np.sqrt(len(ds.cell) / 12)))
        crs = xr.DataArray(
            name="crs",
            attrs={
                "grid_mapping_name": "healpix",
                "healpix_nside": 2 ** zoom_level,
                "healpix_order": "nest",
            },
        )
        ds = ds.assign_coords(crs=crs)
        
        # Add standard metadata for healpix grid
        ds.attrs.update({
            'grid_mapping': 'healpix_nested',
            'healpix_zoom': zoom_level
        })
    
    if parallel:
        # Create dask client for parallel processing
        logger.info(f'Starting dask cluster with {n_workers} workers')
        cluster = LocalCluster(
            n_workers=n_workers, 
            threads_per_worker=1, 
            memory_limit=memory_limit
        )
        client = Client(cluster)
        logger.info(f'Dask dashboard available at: {client.dashboard_link}')
        
        try:
            # Apply chunking if specified
            if chunks:
                ds = ds.chunk(chunks)
            
            logger.info(f'Writing dataset to {store_path} using dask')
            ds.to_zarr(store_path, mode='w', consolidated=True)
            logger.info(f'Successfully wrote dataset to {store_path}')
        finally:
            # Close dask client
            client.close()
            cluster.close()
    else:
        # Write directly without dask
        logger.info(f'Writing dataset to {store_path}')
        ds.to_zarr(store_path, mode='w', consolidated=True)
        logger.info(f'Successfully wrote dataset to {store_path}')
    
    return store_path


def fix_lat_lon_coords(ds, lat_dim="lat", lon_dim="lon"):
    """
    Fix coordinates to ensure they are in correct format:
    - Longitude in 0-360 range
    - Coordinates in ascending order
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset
    lat_dim : str
        Name of latitude dimension
    lon_dim : str
        Name of longitude dimension
        
    Returns:
    --------
    xarray.Dataset: Dataset with fixed coordinates
    """
    # Find where longitude crosses from negative to positive (approx. where lon=0)
    lon_0_index = (ds[lon_dim] < 0).sum().item()

    if lon_0_index > 0:
        # Create indexers for the roll
        lon_indices = np.roll(np.arange(ds.sizes[lon_dim]), -lon_0_index)

        # Roll dataset and convert longitudes to 0-360 range
        ds = ds.isel({lon_dim: lon_indices})
        lon360 = xr.where(ds[lon_dim] < 0, ds[lon_dim] + 360, ds[lon_dim])
        ds = ds.assign_coords({lon_dim: lon360})

    # Ensure latitude and longitude are in ascending order if needed
    if np.all(np.diff(ds[lat_dim].values) < 0):
        ds = ds.isel({lat_dim: slice(None, None, -1)})
    if np.all(np.diff(ds[lon_dim].values) < 0):
        ds = ds.isel({lon_dim: slice(None, None, -1)})

    return ds


def process_latlon_to_healpix_zarr(input_path, output_path, zoom_level, 
                                  weights_path=None, chunks=None, parallel=False, 
                                  n_workers=4, memory_limit='4GB', fix_coords=True, 
                                  subset_extent=None, minute_filter=None):
    """
    Complete workflow to process lat/lon netCDF to healpix zarr.
    
    Parameters:
    -----------
    input_path : str or list
        Path to input netCDF file(s) with lat/lon grid. If a list, will use xr.open_mfdataset.
    output_path : str
        Path to output zarr store
    zoom_level : int
        Healpix zoom level
    weights_path : str or Path, optional
        Path to weights file. If None, weights will be created automatically.
    chunks : dict, optional
        Chunks for dask array
    parallel : bool, optional
        Whether to use dask for parallel processing
    n_workers : int, optional
        Number of workers for dask cluster
    memory_limit : str, optional
        Memory limit per dask worker
    fix_coords : bool, optional
        Whether to fix coordinates (convert longitudes to 0-360 range)
    subset_extent : list, optional
        Subset extent for the dataset in the format [lon_min, lon_max, lat_min, lat_max]
    minute_filter : int, optional
        Filter timestamps to select only those at specific minutes (0 or 30).
        None keeps all timestamps.
        
    Returns:
    --------
    str: Path to the output zarr store
    """
    # Setup logger
    logger.info(f"Processing to healpix zarr at zoom level {zoom_level}")
    
    # Open dataset - handle both single file and multiple files
    if isinstance(input_path, (list, tuple)):
        logger.info(f"Opening multiple input files")
        # ds = xr.open_mfdataset(input_path, combine='by_coords')
        ds = xr.open_mfdataset(
            input_path,
            # chunks=chunk_size,
            # parallel=True,            # Use parallel threads for loading
            combine='nested',         # Better concatenation along record dimension
            concat_dim='time',        # Explicitly specify dimension for concatenation
            compat='override',        # Handle minor metadata conflicts
            engine='netcdf4',         # Explicitly choose engine
            data_vars='minimal',      # Only load variables present in all files
            coords='minimal'          # Only load coordinates present in all files
        )
    else:
        logger.info(f"Opening input file: {input_path}")
        ds = xr.open_dataset(input_path)

    # Filter by minutes if requested
    if minute_filter is not None:
        # Extract minutes from each timestamp
        minutes = ds.time.dt.minute.values
        
        # Define tolerance (in minutes)
        tolerance = 1  # 1 minute tolerance
        
        # Create mask for timestamps near the specified minute
        minute_mask = np.abs(minutes - minute_filter) < tolerance
        
        # Apply mask and update dataset
        logger.info(f"Filtering timestamps to include only those at minute {minute_filter} (±{tolerance})")
        ds = ds.sel(time=ds.time[minute_mask])
        logger.info(f"After filtering: {len(ds.time)} timestamps remaining")

    # Subset dataset if extent is provided
    if subset_extent is not None:
        logger.info(f"Subsetting dataset to extent: {subset_extent}")
        ds = ds.sel(
            lon=slice(subset_extent[0], subset_extent[1]),
            lat=slice(subset_extent[2], subset_extent[3])
        )
        logger.info(f"Subset shape: {ds.sizes}")
    else:
        logger.info(f"Full dataset shape: {ds.sizes}")
    
    # Fix coordinates if needed
    if fix_coords:
        logger.info("Fixing coordinates to 0-360 range")
        ds = fix_lat_lon_coords(ds)
    
    # Create new dataset for healpix data
    ds_hp = xr.Dataset()
    
    # Define regional chunks based on zoom level
    regional_chunks = 12 * 4**zoom_level
    logger.info(f"Using regional chunks: {regional_chunks}")
    
    # Process each variable
    for name, da in ds.data_vars.items():
        logger.info(f"Processing variable: {name}")
        
        # Remap to healpix
        da_hp = remap_to_healpix(
            da, 
            zoom_level=zoom_level,
            weights_path=weights_path,
            regional_chunks=regional_chunks
        )
        
        # Add to dataset
        ds_hp[name] = da_hp
    
    # Write to zarr
    logger.info(f"Writing healpix data to zarr: {output_path}")
    write_dataset_to_zarr(
        ds_hp, 
        output_path, 
        chunks=chunks,
        parallel=parallel, 
        n_workers=n_workers,
        memory_limit=memory_limit
    )
    
    logger.info(f"Processing complete: {output_path}")
    return output_path


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Convert lat/lon NetCDF data to healpix zarr format')
    
    parser.add_argument('-i', '--input', required=True, nargs='+',
                        help='Input NetCDF file(s) with lat/lon grid')
    parser.add_argument('-o', '--output', required=True,
                        help='Output zarr store path')
    parser.add_argument('-z', '--zoom', type=int, default=6,
                        help='Healpix zoom level (default: 6)')
    parser.add_argument('--no-fix-coords', action='store_false', dest='fix_coords',
                        help='Do not fix coordinates to 0-360 range')
    parser.add_argument('-p', '--parallel', action='store_true',
                        help='Use dask for parallel processing')
    parser.add_argument('-w', '--workers', type=int, default=4,
                        help='Number of dask workers (default: 4)')
    parser.add_argument('-m', '--memory', default='4GB',
                        help='Memory limit per dask worker (default: 4GB)')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Specify parameters
    # version = 'V06B'
    version = 'V07B'
    zoom_level = 9
    
    # Minute filter - set to 0 for on-the-hour, 30 for half-hour, or None for all timestamps
    minute_filter = 0  # Change to 30 to get half-hour timestamps, or None to keep all
    
    # Subset IMERG data map extent [lon_min, lon_max, lat_min, lat_max]
    subset_extent = [-180, 180, -60, 60]

    fix_coords = True
    parallel = True
    n_workers = 32
    memory_limit = '15GB'
    
    # Input/output paths
    root_dir = '/pscratch/sd/w/wcmca1/GPM/'
    input_files = [
        f'{root_dir}IR_IMERG_Combined_{version}/IMERG_{version}_2019-2021/merg*.nc'
        # f'{root_dir}IR_IMERG_Combined_{version}/IMERG_{version}_2019-2021/merg_2020010*.nc'
    ]
    output_dir = f'/pscratch/sd/w/wcmca1/GPM/IR_IMERG_Combined_{version}/'
    zarr_path = f'{output_dir}/IMERG_{version}_hp{zoom_level}.zarr'
    
    # Specify weights file path
    os.environ['WEIGHT_DIR'] = f'/pscratch/sd/w/wcmca1/GPM/'
    
    # Expand any glob patterns in input files
    expanded_input_files = []
    for pattern in input_files:
        expanded_input_files.extend(glob(pattern))
    
    if not expanded_input_files:
        logger.error(f"No input files found matching the patterns: {input_files}")
        return 1
    
    logger.info(f"Found {len(expanded_input_files)} input files to process")
    
    # Process data
    out_dir = process_latlon_to_healpix_zarr(
        input_path=expanded_input_files, 
        output_path=zarr_path,
        zoom_level=zoom_level,
        subset_extent=subset_extent,
        minute_filter=minute_filter,
        parallel=parallel,
        n_workers=n_workers,
        memory_limit=memory_limit,
        fix_coords=fix_coords
    )
    
    print(f"Processing complete. Output saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())