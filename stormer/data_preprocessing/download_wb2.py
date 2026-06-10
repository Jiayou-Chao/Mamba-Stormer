import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import xarray as xr
from tqdm import tqdm

def download_var_year(file, var, year, save_dir_var, step, levels):
    """Worker function: Opens the Zarr and saves one year to NetCDF."""
    try:
        save_path = os.path.join(save_dir_var, f'{year}.nc')
        if os.path.exists(save_path):
            return "skipped"

        ds = xr.open_zarr('gs://weatherbench2/datasets/era5/' + file)

        # Select variable
        ds_var = ds[[var]]

        # Subsample levels
        if levels is not None:
            if 'level' in ds_var.coords:
                ds_var = ds_var.sel(level=levels)
            elif 'pressure_level' in ds_var.coords:
                ds_var = ds_var.sel(pressure_level=levels)

        # Select year
        ds_var_year = ds_var.sel(time=str(year))

        # Apply temporal subsampling
        if step > 1:
            ds_var_year = ds_var_year.isel(time=slice(None, None, step))

        # Use engine='h5netcdf' for better stability in parallel if available,
        # otherwise default netcdf4 is used.
        try:
            ds_var_year.to_netcdf(save_path, engine='h5netcdf')
        except Exception:
            ds_var_year.to_netcdf(save_path)
        return "downloaded"
    except Exception as e:
        print(f"Error processing {var} for {year}: {e}")
        return "failed"

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=1, help="Temporal step size")
    parser.add_argument("--levels", type=int, nargs='+', default=None, help="Specific pressure levels")
    parser.add_argument("--start_year", type=int, default=1959, help="Start year")
    parser.add_argument("--end_year", type=int, default=2023, help="End year")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers")

    args = parser.parse_args()
    
    # Auto-detect workers. For downloads, memory is the limit. 
    # 8-12 is usually a safe sweet spot for 350GB.
    if args.num_workers is None:
        args.num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initial open to get metadata
    ds = xr.open_zarr('gs://weatherbench2/datasets/era5/' + args.file)
    variables = list(ds.keys())
    years = list(range(args.start_year, args.end_year + 1))
    
    tasks = []
    completed_year_files = 0
    constant_files_written = 0
    pending_year_files = 0
    print(f"Submitting download tasks using {args.num_workers} workers...")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for var in variables:
            ds_var = ds[[var]]
            if len(ds_var.dims) < 3:  # constant variables
                save_path = os.path.join(args.save_dir, f'{var}.nc')
                if not os.path.exists(save_path):
                    ds_var.to_netcdf(save_path)
                    constant_files_written += 1
            else:
                save_dir_var = os.path.join(args.save_dir, var)
                os.makedirs(save_dir_var, exist_ok=True)
                for year in years:
                    save_path = os.path.join(save_dir_var, f'{year}.nc')
                    if os.path.exists(save_path):
                        completed_year_files += 1
                        continue
                    pending_year_files += 1
                    tasks.append(executor.submit(
                        download_var_year,
                        args.file, var, year, save_dir_var, args.step, args.levels
                    ))

        print(
            f"Existing yearly files: {completed_year_files} | "
            f"Pending yearly files: {pending_year_files} | "
            f"Constant files written this run: {constant_files_written}"
        )

        downloaded = 0
        skipped = 0
        failed = 0
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading data"):
            result = future.result()
            if result == "downloaded":
                downloaded += 1
            elif result == "skipped":
                skipped += 1
            else:
                failed += 1

    print(
        f"Download summary: downloaded={downloaded}, skipped={skipped}, "
        f"failed={failed}, existing={completed_year_files}"
    )
    if failed:
        raise RuntimeError(f"{failed} download tasks failed")

if __name__ == "__main__":
    main()
