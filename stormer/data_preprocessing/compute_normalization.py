import os
import argparse
import numpy as np
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from stormer.utils.data_utils import (
    CONSTANTS,
    SINGLE_LEVEL_VARS,
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS
)

# change as needed
VARS = [
    "anisotropy_of_sub_gridscale_orography",
    "angle_of_sub_gridscale_orography",
    "geopotential_at_surface",
    "high_vegetation_cover",
    "lake_cover",
    # "lake_depth",
    "land_sea_mask",
    "low_vegetation_cover",
    "slope_of_sub_gridscale_orography",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
    
    "mean_surface_latent_heat_flux",
    "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_net_short_wave_radiation_flux",
    "mean_surface_sensible_heat_flux",
    "mean_top_downward_short_wave_radiation_flux",
    "mean_top_net_long_wave_radiation_flux",
    "mean_top_net_short_wave_radiation_flux",
    # "skin_temperature",
    "snow_depth",
    
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "10m_wind_speed",
    "mean_sea_level_pressure",
    "sea_ice_cover",
    "sea_surface_temperature",
    "surface_pressure",
    # "toa_incident_solar_radiation",
    # "toa_incident_solar_radiation_6hr",
    # "toa_incident_solar_radiation_12hr",
    # "toa_incident_solar_radiation_24hr",
    "total_cloud_cover",
    "total_precipitation_6hr",
    "total_precipitation_12hr",
    "total_precipitation_24hr",
    "total_column_water_vapour",
    
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "wind_speed"
]

def process_var_norm(var, root_dir, years, chunk_size, steps):
    """Compute mean and std for a single variable across all years."""
    var_mean_list = {}
    var_std_list = {}
    
    if var in SINGLE_LEVEL_VARS and var not in CONSTANTS:
        var_mean_list[var] = []
        var_std_list[var] = []
    elif var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            var_mean_list[f'{var}_{level}'] = []
            var_std_list[f'{var}_{level}'] = []
    else: # Constant or unknown
        return None

    for year in years:
        path = os.path.join(root_dir, var, f'{year}.nc')
        if not os.path.exists(path):
            continue
        ds = xr.open_dataset(path)
        
        if chunk_size is not None:
            n_chunks = len(ds.time) // chunk_size + (1 if len(ds.time) % chunk_size > 0 else 0)
        else:
            n_chunks = 1
            chunk_size = len(ds.time)
        
        for chunk_id in range(n_chunks):
            ds_small = ds.isel(time=slice(chunk_id*chunk_size, (chunk_id+1)*chunk_size))
            if var in SINGLE_LEVEL_VARS:
                ds_np = ds_small[var].values
                if steps is not None:
                    ds_np = ds_np[steps:] - ds_np[:-steps]
                if ds_np.size == 0: continue
                var_mean_list[var].append(np.nanmean(ds_np))
                var_std_list[var].append(np.nanstd(ds_np))
            else:
                ds_np = ds_small[var].values
                levels_in_ds = ds.level.values
                for i, level in enumerate(levels_in_ds):
                    if level not in DEFAULT_PRESSURE_LEVELS: continue
                    ds_np_lev = ds_np[:, i]
                    if steps is not None:
                        ds_np_lev = ds_np_lev[steps:] - ds_np_lev[:-steps]
                    if ds_np_lev.size == 0: continue
                    var_mean_list[f'{var}_{level}'].append(np.nanmean(ds_np_lev))
                    var_std_list[f'{var}_{level}'].append(np.nanstd(ds_np_lev))
    
    # Final aggregation for this variable
    results = {}
    for key in var_mean_list.keys():
        if not var_mean_list[key]: continue
        mean_vals = np.array(var_mean_list[key])
        std_vals = np.array(var_std_list[key])
        
        # Proper variance aggregation: var(X) = E[var(X|Y)] + var(E[X|Y])
        combined_mean = mean_vals.mean()
        combined_var = (std_vals**2).mean() + (mean_vals**2).mean() - combined_mean**2
        results[key] = (combined_mean, np.sqrt(combined_var))
        
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Compute normalization.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory.')
    parser.add_argument('--save_dir', type=str, required=True, help='Save directory.')
    parser.add_argument('--start_year', type=int, default=1979, help='Start year.')
    parser.add_argument('--end_year', type=int, default=2021, help='End year.')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size.')
    parser.add_argument('--lead_time', type=int, default=None, help='Lead time.')
    parser.add_argument('--data_frequency', type=int, default=6, help='Data frequency.')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of workers.')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.num_workers is None:
        args.num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
    
    years = list(range(args.start_year, args.end_year + 1))
    os.makedirs(args.save_dir, exist_ok=True)

    mean_file_name = f"normalize_diff_mean_{args.lead_time}.npz" if args.lead_time != -1 else "normalize_mean.npz"
    std_file_name = f"normalize_diff_std_{args.lead_time}.npz" if args.lead_time != -1 else "normalize_std.npz"

    steps = args.lead_time // args.data_frequency if args.lead_time is not None and args.lead_time != -1 else None

    # Load existing if available
    if os.path.exists(os.path.join(args.save_dir, mean_file_name)):
        normalize_mean = {k: v.tolist() for k, v in np.load(os.path.join(args.save_dir, mean_file_name)).items()}
        normalize_std = {k: v.tolist() for k, v in np.load(os.path.join(args.save_dir, std_file_name)).items()}
    else:
        normalize_mean = {}
        normalize_std = {}

    print(f"Computing normalization with {args.num_workers} workers...")
    
    tasks = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for var in VARS:
            if var not in CONSTANTS:
                tasks.append(executor.submit(process_var_norm, var, args.root_dir, years, args.chunk_size, steps))
        
        for future in tqdm(as_completed(tasks), total=len(tasks), desc='Computing normalization'):
            res = future.result()
            if res:
                for key, (m, s) in res.items():
                    normalize_mean[key] = np.array([m])
                    normalize_std[key] = np.array([s])

    # Handle constants
    for var in [v for v in VARS if v in CONSTANTS]:
        if steps is not None:
            normalize_mean[var] = np.array([0.0])
            normalize_std[var] = np.array([0.0])
        else:
            path = os.path.join(args.root_dir, f'{var}.nc')
            if os.path.exists(path):
                ds_np = xr.open_dataset(path)[var].values
                normalize_mean[var] = np.array([ds_np.mean()])
                normalize_std[var] = np.array([ds_np.std()])

    np.savez(os.path.join(args.save_dir, mean_file_name), **normalize_mean)
    np.savez(os.path.join(args.save_dir, std_file_name), **normalize_std)
    
if __name__ == "__main__":
    main()