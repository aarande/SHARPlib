#!/usr/bin/env python3
"""
HRRR to Zarr Pipeline using SHARPlib GPU Acceleration.

Fetches latest HRRR model sounding data from AWS S3 (noaa-hrrr-bdp-pds),
extracts required hybrid and surface vertical levels, computes comprehensive
sounding parameters via SHARPlib GPU (CUDA LUT mode) or CPU, and stores
the resulting grids into a chunked, CF-compliant Zarr dataset.
"""

import argparse
import datetime
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import eccodes
import numpy as np
import s3fs
import xarray as xr

# Try importing GPU acceleration
try:
    import cupy as cp
    import nwsspc.sharp.calc as sharp_calc
    HAS_GPU = hasattr(sharp_calc, 'gpu') and len(sharp_calc.gpu.SUPPORTED_PARAMS) > 0
except Exception:
    HAS_GPU = False

from nwsspc.sharp.calc import constants, thermo, parcel, winds, params, layer


# Parameter descriptions and standard units
PARAM_METADATA = {
    "sbcape": {"long_name": "Surface-Based Convective Available Potential Energy", "units": "J/kg"},
    "sbcin": {"long_name": "Surface-Based Convective Inhibition", "units": "J/kg"},
    "sblcl_hght_agl": {"long_name": "Surface-Based Lifted Condensation Level Height AGL", "units": "m"},
    "sblcl_pres": {"long_name": "Surface-Based Lifted Condensation Level Pressure", "units": "Pa"},
    "sblfc_hght_agl": {"long_name": "Surface-Based Level of Free Convection Height AGL", "units": "m"},
    "sblfc_pres": {"long_name": "Surface-Based Level of Free Convection Pressure", "units": "Pa"},
    "sbel_hght_agl": {"long_name": "Surface-Based Equilibrium Level Height AGL", "units": "m"},
    "sbel_pres": {"long_name": "Surface-Based Equilibrium Level Pressure", "units": "Pa"},
    "sb3cape": {"long_name": "Surface-Based 0-3 km CAPE", "units": "J/kg"},
    "mlcape": {"long_name": "Mixed-Layer Convective Available Potential Energy", "units": "J/kg"},
    "mlcin": {"long_name": "Mixed-Layer Convective Inhibition", "units": "J/kg"},
    "mllcl_hght_agl": {"long_name": "Mixed-Layer Lifted Condensation Level Height AGL", "units": "m"},
    "mllcl_pres": {"long_name": "Mixed-Layer Lifted Condensation Level Pressure", "units": "Pa"},
    "mllfc_hght_agl": {"long_name": "Mixed-Layer Level of Free Convection Height AGL", "units": "m"},
    "mllfc_pres": {"long_name": "Mixed-Layer Level of Free Convection Pressure", "units": "Pa"},
    "mlel_hght_agl": {"long_name": "Mixed-Layer Equilibrium Level Height AGL", "units": "m"},
    "mlel_pres": {"long_name": "Mixed-Layer Equilibrium Level Pressure", "units": "Pa"},
    "ml3cape": {"long_name": "Mixed-Layer 0-3 km CAPE", "units": "J/kg"},
    "mucape": {"long_name": "Most-Unstable Convective Available Potential Energy", "units": "J/kg"},
    "mucin": {"long_name": "Most-Unstable Convective Inhibition", "units": "J/kg"},
    "mulcl_hght_agl": {"long_name": "Most-Unstable Lifted Condensation Level Height AGL", "units": "m"},
    "mulcl_pres": {"long_name": "Most-Unstable Lifted Condensation Level Pressure", "units": "Pa"},
    "mulfc_hght_agl": {"long_name": "Most-Unstable Level of Free Convection Height AGL", "units": "m"},
    "mulfc_pres": {"long_name": "Most-Unstable Level of Free Convection Pressure", "units": "Pa"},
    "muel_hght_agl": {"long_name": "Most-Unstable Equilibrium Level Height AGL", "units": "m"},
    "muel_pres": {"long_name": "Most-Unstable Equilibrium Level Pressure", "units": "Pa"},
    "mu3cape": {"long_name": "Most-Unstable 0-3 km CAPE", "units": "J/kg"},
    "bunkers_rm_u": {"long_name": "Bunkers Right-Mover Storm Motion U-Component", "units": "m/s"},
    "bunkers_rm_v": {"long_name": "Bunkers Right-Mover Storm Motion V-Component", "units": "m/s"},
    "bunkers_lm_u": {"long_name": "Bunkers Left-Mover Storm Motion U-Component", "units": "m/s"},
    "bunkers_lm_v": {"long_name": "Bunkers Left-Mover Storm Motion V-Component", "units": "m/s"},
    "mean_wind_u": {"long_name": "0-6 km Mean Wind U-Component", "units": "m/s"},
    "mean_wind_v": {"long_name": "0-6 km Mean Wind V-Component", "units": "m/s"},
    "shr01": {"long_name": "0-1 km Bulk Wind Shear", "units": "m/s"},
    "shr03": {"long_name": "0-3 km Bulk Wind Shear", "units": "m/s"},
    "shr06": {"long_name": "0-6 km Bulk Wind Shear", "units": "m/s"},
    "srh01": {"long_name": "0-1 km Storm-Relative Helicity", "units": "m^2/s^2"},
    "srh03": {"long_name": "0-3 km Storm-Relative Helicity", "units": "m^2/s^2"},
    "srh05": {"long_name": "0-500 m Storm-Relative Helicity", "units": "m^2/s^2"},
    "crit_angle": {"long_name": "Critical Angle", "units": "degrees"},
    "pwat": {"long_name": "Precipitable Water", "units": "mm"},
    "lr03": {"long_name": "0-3 km Lapse Rate", "units": "C/km"},
    "lr36": {"long_name": "3-6 km Lapse Rate", "units": "C/km"},
    "lr75": {"long_name": "700-500 hPa Lapse Rate", "units": "C/km"},
    "fz_level_agl": {"long_name": "Freezing Level Height AGL", "units": "m"},
    "wbz_level_agl": {"long_name": "Wet-Bulb Zero Height AGL", "units": "m"},
    "stp": {"long_name": "Significant Tornado Parameter (Fixed Layer)", "units": "1"},
    "scp": {"long_name": "Supercell Composite Parameter", "units": "1"},
    "ehi01": {"long_name": "0-1 km Energy-Helicity Index", "units": "1"},
    "ehi03": {"long_name": "0-3 km Energy-Helicity Index", "units": "1"},
}


def find_latest_cycle(fs: s3fs.S3FileSystem, bucket: str = "noaa-hrrr-bdp-pds") -> Tuple[str, str]:
    """Finds the most recent complete cycle (YYYYMMDD and HHz) available on AWS S3."""
    now = datetime.datetime.now(datetime.timezone.utc)
    for day_offset in range(3):
        date_obj = now - datetime.timedelta(days=day_offset)
        datestr = date_obj.strftime("%Y%m%d")
        prefix = f"{bucket}/hrrr.{datestr}/conus"
        try:
            files = fs.ls(prefix)
        except Exception:
            continue

        nat_idx = [f for f in files if "wrfnatf00.grib2.idx" in f]
        if not nat_idx:
            continue

        latest_idx = sorted(nat_idx)[-1]
        cycle_str = latest_idx.split("hrrr.t")[1].split("z.")[0]
        return datestr, f"{cycle_str}z"

    raise RuntimeError("Could not find any recent HRRR cycles on AWS S3.")


def parse_grib_index(idx_text: str) -> List[Tuple[int, int, Optional[int], str, str]]:
    """
    Parses a GRIB2 .idx file into a list of tuples:
    (line_num, start_byte, end_byte, var_short, level_str)
    """
    lines = idx_text.strip().split("\n")
    parsed = []
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) >= 6:
            msg_num = int(parts[0])
            start_byte = int(parts[1])
            var = parts[3]
            lvl = parts[4]
            next_start = None
            if i + 1 < len(lines):
                next_parts = lines[i + 1].split(":")
                if len(next_parts) >= 2:
                    next_start = int(next_parts[1])
            parsed.append((msg_num, start_byte, next_start, var, lvl))
    return parsed


def fetch_hrrr_profile_data(
    fs: s3fs.S3FileSystem,
    bucket: str,
    datestr: str,
    cycle: str,
    fxx: int,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Downloads required surface and hybrid levels for soundings via targeted byte ranges.
    Returns: (pres, hght, tmpk, dwpk, u, v, lats, lons) with shape (51, NY, NX)
    """
    cycle_clean = cycle.replace("z", "").zfill(2)
    fxx_str = str(fxx).zfill(2)
    
    sfc_file = f"{bucket}/hrrr.{datestr}/conus/hrrr.t{cycle_clean}z.wrfsfcf{fxx_str}.grib2"
    nat_file = f"{bucket}/hrrr.{datestr}/conus/hrrr.t{cycle_clean}z.wrfnatf{fxx_str}.grib2"

    if verbose:
        print(f"Fetching indices for cycle {datestr} t{cycle_clean}z f{fxx_str}...")

    sfc_idx_text = fs.cat(sfc_file + ".idx").decode("utf-8")
    nat_idx_text = fs.cat(nat_file + ".idx").decode("utf-8")

    sfc_parsed = parse_grib_index(sfc_idx_text)
    nat_parsed = parse_grib_index(nat_idx_text)

    # 1. Surface variables mapping
    sfc_targets = {
        "sp": (":PRES:surface", "sp"),
        "orog": (":HGT:surface", "orog"),
        "t2m": (":TMP:2 m above ground", "t2m"),
        "q2m": (":SPFH:2 m above ground", "q2m"),
        "u10": (":UGRD:10 m above ground", "u10"),
        "v10": (":VGRD:10 m above ground", "v10"),
    }

    sfc_arrays = {}
    lats = None
    lons = None

    if verbose:
        print("Downloading 2D surface boundary fields...")

    for key, (pattern, _) in sfc_targets.items():
        found = False
        for msg_num, start_b, end_b, var, lvl in sfc_parsed:
            line_str = f":{var}:{lvl}"
            if pattern in line_str:
                data = fs.cat_file(sfc_file, start=start_b, end=end_b)
                gid = eccodes.codes_new_from_message(data)
                arr = eccodes.codes_get_values(gid).astype(np.float32)
                if lats is None:
                    Ny = eccodes.codes_get(gid, "Ny")
                    Nx = eccodes.codes_get(gid, "Nx")
                    lats = eccodes.codes_get_array(gid, "latitudes").reshape(Ny, Nx).astype(np.float32)
                    lons = eccodes.codes_get_array(gid, "longitudes").reshape(Ny, Nx).astype(np.float32)
                sfc_arrays[key] = arr
                eccodes.codes_release(gid)
                found = True
                break
        if not found:
            raise KeyError(f"Could not find surface field matching {pattern} in {sfc_file}")

    Ny, Nx = lats.shape
    total_profiles = Ny * Nx

    # 2. Hybrid variables (50 levels)
    # Order: level 1 (top of model) to level 50 (near-surface).
    # Surface level (level 0) is placed at index 0, followed by level 50, 49, ..., 1.
    if verbose:
        print("Parsing 50 hybrid atmospheric levels from wrfnat...")

    hybrid_vars = ["PRES", "HGT", "TMP", "SPFH", "UGRD", "VGRD"]
    hybrid_dict = {v: {} for v in hybrid_vars}

    for msg_num, start_b, end_b, var, lvl in nat_parsed:
        if var in hybrid_vars and "hybrid level" in lvl:
            lvl_num = int(lvl.split()[0])
            hybrid_dict[var][lvl_num] = (start_b, end_b)

    # Allocate full 3D arrays: 51 vertical levels x (Ny * Nx)
    # Level index 0 = surface.
    # Level index 1..50 = model hybrid levels from 50 (lowest) up to 1 (top).
    pres_3d = np.empty((51, total_profiles), dtype=np.float32)
    hght_3d = np.empty((51, total_profiles), dtype=np.float32)
    tmpk_3d = np.empty((51, total_profiles), dtype=np.float32)
    spfh_3d = np.empty((51, total_profiles), dtype=np.float32)
    u_3d    = np.empty((51, total_profiles), dtype=np.float32)
    v_3d    = np.empty((51, total_profiles), dtype=np.float32)

    pres_3d[0] = sfc_arrays["sp"]
    hght_3d[0] = sfc_arrays["orog"]
    tmpk_3d[0] = sfc_arrays["t2m"]
    spfh_3d[0] = sfc_arrays["q2m"]
    u_3d[0]    = sfc_arrays["u10"]
    v_3d[0]    = sfc_arrays["v10"]

    if verbose:
        print(f"Downloading 300 vertical slices for {Ny}x{Nx} grid columns in parallel...")

    import concurrent.futures

    tasks = []
    # (vname, model_lvl, target_k, start_b, end_b)
    for model_lvl in range(50, 0, -1):
        target_k = 51 - model_lvl  # level 50 -> 1, level 49 -> 2, ..., level 1 -> 50
        for vname in ["PRES", "HGT", "TMP", "SPFH", "UGRD", "VGRD"]:
            start_b, end_b = hybrid_dict[vname][model_lvl]
            tasks.append((vname, target_k, start_b, end_b))

    t0_fetch = time.perf_counter()

    def fetch_slice(item):
        vname, target_k, start_b, end_b = item
        data = fs.cat_file(nat_file, start=start_b, end=end_b)
        gid = eccodes.codes_new_from_message(data)
        arr = eccodes.codes_get_values(gid).astype(np.float32)
        eccodes.codes_release(gid)
        return vname, target_k, arr

    var_arr_map = {
        "PRES": pres_3d,
        "HGT": hght_3d,
        "TMP": tmpk_3d,
        "SPFH": spfh_3d,
        "UGRD": u_3d,
        "VGRD": v_3d,
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for vname, target_k, arr in executor.map(fetch_slice, tasks):
            var_arr_map[vname][target_k] = arr

    if verbose:
        print(f"Downloaded and unpacked all atmospheric soundings in {time.perf_counter() - t0_fetch:.1f}s.")

    # Calculate Dewpoint temperature from Specific Humidity
    # Follows SHARPlib standard formula
    mixr = thermo.mixratio(spfh_3d)
    mixr[mixr < constants.TOL] = constants.TOL
    dwpk_3d = thermo.temperature_at_mixratio(mixr, pres_3d).astype(np.float32)

    return pres_3d, hght_3d, tmpk_3d, dwpk_3d, u_3d, v_3d, lats, lons


def compute_parameters_gpu(
    pres: np.ndarray,
    hght: np.ndarray,
    tmpk: np.ndarray,
    dwpk: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    use_texture_lut: bool = True,
    batch_size: int = 500000,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Computes all convective parameters using SHARPlib CUDA kernels in chunked batches.
    Returns dictionary mapping parameter names to 1D numpy arrays of shape (N_profiles,).
    """
    n_levels, n_profiles = pres.shape
    results = {p: np.empty(n_profiles, dtype=np.float32) for p in sharp_calc.gpu.SUPPORTED_PARAMS}

    if verbose:
        mode_str = "Texture LUT (Fast)" if use_texture_lut else "Exact Wobus"
        print(f"Running SHARPlib GPU Acceleration [{mode_str}] across {n_profiles:,} soundings...")

    t0_gpu = time.perf_counter()
    for start_idx in range(0, n_profiles, batch_size):
        end_idx = min(start_idx + batch_size, n_profiles)
        
        # Transfer batch to GPU memory as level-major contiguous tensors
        p_dev = cp.ascontiguousarray(cp.asarray(pres[:, start_idx:end_idx]))
        h_dev = cp.ascontiguousarray(cp.asarray(hght[:, start_idx:end_idx]))
        t_dev = cp.ascontiguousarray(cp.asarray(tmpk[:, start_idx:end_idx]))
        d_dev = cp.ascontiguousarray(cp.asarray(dwpk[:, start_idx:end_idx]))
        u_dev = cp.ascontiguousarray(cp.asarray(u[:, start_idx:end_idx]))
        v_dev = cp.ascontiguousarray(cp.asarray(v[:, start_idx:end_idx]))

        batch_out = sharp_calc.compute_sounding_grid(
            p_dev, h_dev, t_dev, d_dev, u_dev, v_dev,
            params="all",
            use_texture_lut=use_texture_lut,
            fast_mu_thetae=True
        )
        cp.cuda.Stream.null.synchronize()

        for k, v_arr in batch_out.items():
            results[k][start_idx:end_idx] = v_arr.get()

    elapsed = time.perf_counter() - t0_gpu
    if verbose:
        rate = n_profiles / elapsed
        print(f"GPU execution completed in {elapsed:.2f}s ({rate:,.0f} profiles/sec).")

    return results


def write_to_zarr(
    results: Dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    output_zarr: str,
    datestr: str,
    cycle: str,
    fxx: int,
    use_texture_lut: bool = True,
    verbose: bool = True
):
    """
    Writes computed parameter grids to a chunked, CF-compliant Zarr dataset.
    """
    Ny, Nx = lats.shape
    cycle_int = int(cycle.replace("z", ""))
    init_time = np.datetime64(datetime.datetime.strptime(f"{datestr}{cycle_int:02d}", "%Y%m%d%H"), "ns")
    valid_time = init_time + np.timedelta64(fxx, "h")

    # Reshape 1D results to (step, y, x)
    data_vars = {}
    for p_name, p_data in results.items():
        arr_2d = p_data.reshape(1, Ny, Nx)
        # Convert SHARPlib MISSING sentinel to NaN
        arr_2d = np.where(arr_2d == constants.MISSING, np.nan, arr_2d)
        
        meta = PARAM_METADATA.get(p_name, {"long_name": p_name, "units": "1"})
        data_vars[p_name] = (
            ("step", "y", "x"),
            arr_2d,
            {"long_name": meta["long_name"], "units": meta["units"]}
        )

    coords = {
        "step": [fxx],
        "init_time": (("step",), [init_time]),
        "valid_time": (("step",), [valid_time]),
        "latitude": (("y", "x"), lats, {"units": "degrees_north", "standard_name": "latitude"}),
        "longitude": (("y", "x"), lons, {"units": "degrees_east", "standard_name": "longitude"}),
    }

    attrs = {
        "title": "High-Resolution Rapid Refresh (HRRR) Convective Sounding Parameters",
        "institution": "NOAA / NWS / SPC (Computed via SHARPlib GPU)",
        "source": f"AWS S3 noaa-hrrr-bdp-pds cycle {datestr} t{cycle}z f{fxx:02d}",
        "sharplib_gpu_mode": "LUT (Texture Cache)" if use_texture_lut else "Exact Wobus",
        "Conventions": "CF-1.8",
        "history": f"Created on {datetime.datetime.now(datetime.timezone.utc).isoformat()} via hrrr_to_zarr.py",
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    # Chunking: 1 step, 256 x 256 spatial
    ds = ds.chunk({"step": 1, "y": 256, "x": 256})

    if verbose:
        print(f"Writing dataset to Zarr store: {output_zarr}...")

    t0_write = time.perf_counter()
    if os.path.exists(output_zarr) and fxx > 0:
        # Append along step dimension if store already exists
        ds.to_zarr(output_zarr, mode="a", append_dim="step")
    else:
        ds.to_zarr(output_zarr, mode="w")

    if verbose:
        print(f"Zarr store successfully written in {time.perf_counter() - t0_write:.2f}s.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process NOAA HRRR soundings on AWS S3, compute parameters via SHARPlib GPU, and write to Zarr."
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        default="hrrr_soundings.zarr",
        help="Path to output Zarr store (default: hrrr_soundings.zarr)"
    )
    parser.add_argument(
        "--cycle", "-c",
        type=str,
        default="auto",
        help="Model run cycle: 'auto' for latest available, or YYYYMMDDHH / YYYYMMDD tHHz (default: auto)"
    )
    parser.add_argument(
        "--fxx",
        type=str,
        default="0",
        help="Forecast hour(s) to process: single integer (e.g. 0), list (e.g. '0,1,2'), or range (e.g. '0-3') (default: 0)"
    )
    parser.add_argument(
        "--lut",
        action="store_true",
        default=True,
        help="Use GPU texture lookup table mode for moist parcel ascent (default: True)"
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Force exact Wobus polynomial lifter mode instead of LUT mode"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500000,
        help="Batch size (number of soundings) for GPU memory management (default: 500,000)"
    )
    return parser.parse_args()


def parse_forecast_hours(fxx_str: str) -> List[int]:
    """Parses forecast hours from string formatted as '0', '0,1,2', or '0-3'."""
    fxx_str = fxx_str.strip()
    if "-" in fxx_str:
        start_h, end_h = map(int, fxx_str.split("-"))
        return list(range(start_h, end_h + 1))
    elif "," in fxx_str:
        return [int(x.strip()) for x in fxx_str.split(",")]
    else:
        return [int(fxx_str)]


def main():
    args = parse_args()
    use_lut = not args.exact

    print("=" * 70)
    print(" SHARPlib Operational HRRR Sounding Pipeline (AWS S3 -> Zarr)")
    print("=" * 70)
    
    if not HAS_GPU:
        print("ERROR: SHARPlib GPU acceleration is not available in the current environment.")
        sys.exit(1)

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"CUDA GPU Device:     {props['name'].decode()}")
    print(f"Compute Mode:        {'LUT (Hardware Texture Cache)' if use_lut else 'Exact (Wobus Solver)'}")
    print(f"Output Zarr Store:   {args.out}")

    fs = s3fs.S3FileSystem(anon=True)
    bucket = "noaa-hrrr-bdp-pds"

    # Determine cycle
    if args.cycle.lower() == "auto":
        print("Scanning AWS S3 for the most recent complete HRRR model cycle...")
        datestr, cycle = find_latest_cycle(fs, bucket=bucket)
        print(f"Found latest cycle:  Date={datestr}, Cycle={cycle}")
    else:
        clean = args.cycle.strip().replace("t", "").replace("z", "")
        if len(clean) == 10:
            datestr = clean[:8]
            cycle = f"{clean[8:]}z"
        else:
            raise ValueError(f"Invalid cycle format: {args.cycle}. Expected 'auto' or 'YYYYMMDDHH'.")
        print(f"Selected cycle:      Date={datestr}, Cycle={cycle}")

    forecast_hours = parse_forecast_hours(args.fxx)
    print(f"Forecast hours:      {forecast_hours}")
    print("=" * 70)

    for fxx in forecast_hours:
        print(f"\n---> Processing Forecast Hour f{fxx:02d}...")
        t0 = time.perf_counter()
        
        pres, hght, tmpk, dwpk, u, v, lats, lons = fetch_hrrr_profile_data(
            fs, bucket, datestr, cycle, fxx=fxx, verbose=True
        )

        results = compute_parameters_gpu(
            pres, hght, tmpk, dwpk, u, v,
            use_texture_lut=use_lut,
            batch_size=args.batch_size,
            verbose=True
        )

        write_to_zarr(
            results, lats, lons,
            output_zarr=args.out,
            datestr=datestr,
            cycle=cycle,
            fxx=fxx,
            use_texture_lut=use_lut,
            verbose=True
        )

        total_time = time.perf_counter() - t0
        print(f"---> Forecast Hour f{fxx:02d} completed in {total_time:.2f}s ({total_time/60.0:.2f} min).")

    print("\nPipeline execution finished successfully.")


if __name__ == "__main__":
    main()
