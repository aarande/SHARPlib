## Examples Overview

- **`gpu-acceleration-benchmark.ipynb`**: High-throughput GPU batch acceleration and benchmarking across 2D/3D atmospheric grids (HRRR, RAP, GFS), demonstrating Native CUDA batched execution, 2D hardware texture lookup tables, selective parameter execution, and up to **6,600x speedup** over CPU loops.
- **`hrrr-example.ipynb`**: Working with 3D gridded HRRR data via remote kerchunk references and Dask chunking.
- **`lifting-parcels.ipynb`**: In-depth parcel lifting, thermodynamic adiabats (Wobus vs. CM1), and skew-T calculations.

## Install Prerequisites

```bash
conda create -n SHARPlib python=3.12
pip install -r requirements.txt
# Optional for GPU acceleration:
# pip install cupy-cuda12x  # or matching CUDA version
```
