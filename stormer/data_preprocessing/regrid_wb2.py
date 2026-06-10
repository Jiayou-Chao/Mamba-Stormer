import os
import argparse
import xarray as xr
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    from . import regridding
except ImportError:
    import regridding

# change as needed
VARS = [
    # constants
    "angle_of_sub_gridscale_orography.nc",
    "anisotropy_of_sub_gridscale_orography.nc",
    "geopotential_at_surface.nc",
    "high_vegetation_cover.nc",
    "lake_cover.nc",
    "lake_depth.nc",
    "land_sea_mask.nc",
    "low_vegetation_cover.nc",
    "slope_of_sub_gridscale_orography.nc",
    "soil_type.nc",
    "standard_deviation_of_filtered_subgrid_orography.nc",
    "standard_deviation_of_orography.nc",
    "type_of_high_vegetation.nc",
    "type_of_low_vegetation.nc",

    # surface variables
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "10m_wind_speed",
    "mean_sea_level_pressure",

    # pressure level variables
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

def regrid_file(file_path, save_path, new_lat, new_lon, chunk_size=None):
    """Regrid a single NetCDF file."""
    if chunk_size:
        try:
            ds_in = xr.open_dataset(file_path, chunks={'time': chunk_size})
        except (ValueError, ImportError) as exc:
            # xarray requires dask for chunked reads. Fall back to eager reads when
            # dask is unavailable so HPC jobs do not fail on environment differences.
            if "chunk manager 'dask' is not available" in str(exc):
                ds_in = xr.open_dataset(file_path)
            else:
                raise
    else:
        ds_in = xr.open_dataset(file_path)
    
    # Check if we need to transpose
    if 'latitude' in ds_in.dims and 'longitude' in ds_in.dims:
        ds_in = ds_in.transpose(..., 'latitude', 'longitude')
    
    old_lon = ds_in.coords['longitude'].data
    old_lat = ds_in.coords['latitude'].data
    source_grid = regridding.Grid.from_degrees(lon=old_lon, lat=np.sort(old_lat))
    target_grid = regridding.Grid.from_degrees(lon=new_lon, lat=new_lat)
    regridder = regridding.ConservativeRegridder(source_grid, target_grid)
    
    ds_out = regridder.regrid_dataset(ds_in)
    if 'latitude' in ds_out.dims and 'longitude' in ds_out.dims:
        ds_out = ds_out.transpose(..., 'latitude', 'longitude')
    
    ds_out.to_netcdf(save_path)
    return save_path

def parse_args():
    parser = argparse.ArgumentParser(description='Regridding NetCDF files.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save regridded files.')
    parser.add_argument('--ddeg_out', type=float, default=1.40625, help='Output grid spacing in degrees.')
    parser.add_argument('--start_year', type=int, default=1979, help='Start year for the data range.')
    parser.add_argument('--end_year', type=int, default=2021, help='End year for the data range.')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for reading datasets (default=100).')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers.')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    root_dir = args.root_dir
    save_dir = args.save_dir
    ddeg_out = args.ddeg_out
    start_year = args.start_year
    end_year = args.end_year
    chunk_size = args.chunk_size
    
    if args.num_workers is None:
        args.num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

    years = list(range(start_year, end_year + 1))
    os.makedirs(save_dir, exist_ok=True)

    lat_start = -90 + ddeg_out / 2
    lat_stop = 90 - ddeg_out / 2
    new_lat = np.linspace(lat_start, lat_stop, num=int(180/ddeg_out), endpoint=True)
    new_lon = np.linspace(0, 360, num=int(360//ddeg_out), endpoint=False)
    
    tasks = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for v in VARS:
            dir_path = os.path.join(root_dir, v)
            if not os.path.exists(dir_path):
                print(f"Warning: {dir_path} does not exist. Skipping.")
                continue
                
            if '.nc' in v:
                save_path = os.path.join(save_dir, v)
                tasks.append(executor.submit(regrid_file, dir_path, save_path, new_lat, new_lon))
            else:
                os.makedirs(os.path.join(save_dir, v), exist_ok=True)
                for year in years:
                    file_path = os.path.join(dir_path, f'{year}.nc')
                    if os.path.exists(file_path):
                        save_path = os.path.join(save_dir, v, f'{year}.nc')
                        tasks.append(executor.submit(regrid_file, file_path, save_path, new_lat, new_lon, chunk_size))
        
        failures = 0
        for future in tqdm(as_completed(tasks), total=len(tasks), desc='Regridding'):
            try:
                future.result()
            except Exception as exc:
                failures += 1
                print(f"Regridding task failed: {exc}")

    if failures:
        raise RuntimeError(f"Regridding failed for {failures} files")

if __name__ == "__main__":
    main()
