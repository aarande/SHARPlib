## Examples Overview

- **`hrrr-cpu-vs-gpu.ipynb`**: End-to-end CPU vs. GPU acceleration benchmark and accuracy verification on the real CONUS HRRR dataset (1.9M profiles), comparing traditional Dask CPU execution against GPU Exact (Wobus) and Hardware Texture LUT modes with detailed numerical tolerance validation and Cartopy visualizations.
- **`hrrr_to_zarr.py`**: Production CLI pipeline script that fetches the latest HRRR cycle from AWS S3 (`noaa-hrrr-bdp-pds`), extracts hybrid and surface soundings via parallel byte-range requests, computes all convective parameters via SHARPlib GPU, and exports a chunked, CF-compliant Zarr dataset.
- **`gpu-acceleration-benchmark.ipynb`**: High-throughput GPU batch acceleration and benchmarking across 2D/3D atmospheric grids (HRRR, RAP, GFS), demonstrating Native CUDA batched execution, 2D hardware texture lookup tables, selective parameter execution, and up to **6,600x speedup** over CPU loops.
- **`hrrr-example.ipynb`**: Working with 3D gridded HRRR data via remote kerchunk references and Dask chunking.
- **`lifting-parcels.ipynb`**: In-depth parcel lifting, thermodynamic adiabats (Wobus vs. CM1), and skew-T calculations.

## Install Prerequisites

```bash
conda create -n SHARPlib python=3.12
pip install -r requirements.txt
# Optional for GPU acceleration:
# pip install cupy-cuda12x zarr s3fs
```
