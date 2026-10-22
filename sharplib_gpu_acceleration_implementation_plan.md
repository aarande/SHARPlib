# SHARPlib GPU Acceleration Implementation Plan (Approach 1: Native CUDA + `nanobind` DLPack)

## 1. Objective & Architectural Overview
Upgrade `SHARPlib` (`keltonhalbert/SHARPlib`) to support high-throughput, batched GPU execution across 2D/3D atmospheric grids (e.g., HRRR, RAP, GFS, or sounding ensembles) using **Native CUDA**, **`__host__ __device__` (`SHARP_HD`) header annotations**, **2D Hardware Texture Lookup Tables (LUTs)** for moist adiabatic ascent, and **zero-copy `nanobind` GPU tensor bindings** (`CuPy` / `PyTorch` via DLPack).

### Core Design Rules for the Coding Agent
1. **Preserve 100% Backwards Compatibility:** All existing CPU C++ APIs, unit tests, and 1D Python bindings must continue to compile and pass when `SHARP_ENABLE_CUDA=OFF`.
2. **Parallelism Axis:** Map **1 GPU thread = 1 vertical sounding profile** (`profile_idx = blockIdx.x * blockDim.x + threadIdx.x`). **Never** parallelize across the vertical dimension ($N_z$) within a single profile.
3. **Coalesced Memory Layout:** All batched GPU kernels must operate on **Level-Major (`[N_levels, N_profiles]`)** contiguous arrays so that 32 threads in a warp reading vertical level $k$ (`arr[k * N_profiles + profile_idx]`) execute a single coalesced 128-byte global memory transaction.
4. **Zero Per-Thread Local Memory Spills:** GPU kernels must **never** allocate dynamic or large stack arrays (e.g., `float buoy[N_z]`) per thread. Parcel lifting and CAPE/CIN/LFC/EL integration must be fused into a single bottom-to-top scalar register pass.

---

## 2. Target Repository File Changes

```text
SHARPlib/
├── CMakeLists.txt                              # [MODIFY] Add optional CUDA language & nvcc flags
├── include/SHARPlib/
│   ├── constants.h                             # [MODIFY] Define SHARP_HD & SHARP_INLINE macros
│   ├── layer.h                                 # [MODIFY] Annotate structs/methods with SHARP_HD + stride support
│   ├── interp.h                                # [MODIFY] Annotate interpolation routines with SHARP_HD + stride support
│   ├── thermo.h                                # [MODIFY] Annotate thermodynamic functions with SHARP_HD
│   ├── winds.h                                 # [MODIFY] Annotate wind shear/helicity routines with SHARP_HD
│   ├── parcel.h                                # [MODIFY] Annotate lifters & parcel structs with SHARP_HD
│   └── gpu/                                    # [NEW DIRECTORY]
│       ├── strided_ptr.h                       # [NEW] Lightweight strided pointer wrapper for [Z, N] indexing
│       ├── lut_texture.cuh                     # [NEW] 2D CUDA Texture Object manager for moist adiabats
│       ├── fused_parcel.cuh                    # [NEW] Single-pass register-fused lift + CAPE/CIN/LFC/EL
│       └── batch_kernels.h                     # [NEW] Host-callable C++ declarations for GPU batch launchers
├── src/SHARPlib/
│   ├── CMakeLists.txt                          # [MODIFY] Conditionally compile gpu/*.cu sources
│   └── gpu/                                    # [NEW DIRECTORY]
│       ├── lut_texture.cu                      # [NEW] Host LUT generation (via lifter_cm1/wobus) & cudaTextureObject_t binding
│       └── batch_kernels.cu                    # [NEW] Batched CUDA kernels (SB/ML/MU CAPE, CIN, LCL, LFC, EL, SRH, Shear)
├── src/nanobind/
│   ├── pySHARP.cpp                             # [MODIFY] Register `batch_gpu` submodule if compiled with CUDA
│   └── gpu_bindings.cpp                        # [NEW] Zero-copy nb::ndarray<float, nb::cupy> & nb::pytorch bindings
└── tests/
    └── python/
        └── test_gpu_batch.py                   # [NEW] Accuracy & benchmark test comparing CPU 1D vs GPU Batch
```

---

## 3. Step-by-Step Implementation Tasks

### Phase 1: Build System & Portability Macros

#### Task 1.1: Update Root `CMakeLists.txt` for Optional CUDA Support
1. Add a CMake option:
   ```cmake
   option(SHARP_ENABLE_CUDA "Enable CUDA GPU batch acceleration" OFF)
   ```
2. When `SHARP_ENABLE_CUDA` is `ON`:
   - Call `enable_language(CUDA)` and `find_package(CUDAToolkit REQUIRED)`.
   - Set `CMAKE_CUDA_STANDARD 17` and `CMAKE_CUDA_STANDARD_REQUIRED ON`.
   - Set `CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90"` (covering Turing, Ampere, Ada Lovelace, Hopper).
   - Add compile definition `-DSHARP_HAS_CUDA=1` and NVCC flags `--expt-relaxed-constexpr --use_fast_math` (optional toggle for strict IEEE vs fast math).

#### Task 1.2: Define `SHARP_HD` and `StridedPtr`
1. In `include/SHARPlib/constants.h`, add:
   ```cpp
   #if defined(__CUDACC__) || defined(__HIPCC__)
       #define SHARP_HD __host__ __device__
       #define SHARP_DEVICE __device__
       #define SHARP_FORCEINLINE __forceinline__
   #else
       #define SHARP_HD
       #define SHARP_DEVICE
       #define SHARP_FORCEINLINE inline
   #endif
   ```
2. Create `include/SHARPlib/gpu/strided_ptr.h`:
   - Implement a zero-overhead template wrapper `sharp::StridedPtr<T>` that holds `T* ptr` and `std::ptrdiff_t stride`.
   - Overload `SHARP_HD SHARP_FORCEINLINE T operator[](std::ptrdiff_t k) const noexcept { return ptr[k * stride]; }`.
   - This allows existing 1D algorithms written for `arr[k]` to work transparently on `[N_levels, N_profiles]` GPU arrays where `ptr = base_ptr + profile_idx` and `stride = N_profiles`.

---

### Phase 2: Dual-Target (`SHARP_HD`) Header Refactoring

> **Agent Note:** `SHARPlib` Currently splits declarations in `include/SHARPlib/*.h` and definitions in `src/SHARPlib/*.cpp`. For CUDA `__device__` functions to be inlined into GPU kernels across translation units without expensive CUDA Relocatable Device Code (`-rdc=true`, which degrades register allocation and complicates shared library linking), move core scalar/1D mathematical definitions into `inline SHARP_HD` functions inside headers (or `.inl` files included by headers).

#### Task 2.1: Refactor `interp.h` and `layer.h`
1. Annotate `PressureLayer`, `HeightLayer`, `LayerIndex`, and `get_layer_index` with `SHARP_HD`.
2. Annotate `interp_height`, `interp_pressure`, `generic_mean`, `layer_min`, and `layer_max` with `SHARP_HD`.
3. Overload or template array pointer inputs (`template <typename Ptr = const float*>`) so both `const float*` (CPU, `stride=1`) and `sharp::StridedPtr<const float>` (GPU, `stride=N_profiles`) work seamlessly.

#### Task 2.2: Refactor `thermo.h`
1. Move scalar thermodynamic implementations into `include/SHARPlib/thermo.h` with `SHARP_HD SHARP_FORCEINLINE`:
   - `vapor_pressure`, `mixratio`, `specific_humidity`, `virtual_temperature`
   - `theta`, `theta_level`, `thetae`, `wetbulb`, `wetlift`, `wobf`
   - `drylift`, `lcl_temperature`, `buoyancy`
2. Ensure all `std::log`, `std::exp`, `std::pow`, `std::isnan`, and `std::clamp` calls remain standard `<cmath>` / `<algorithm>` functions (NVCC automatically maps `std::` math functions on `float` to fast CUDA device intrinsics when `--expt-relaxed-constexpr` is enabled).

#### Task 2.3: Refactor `winds.h`
1. Annotate `WindComponents`, `WindVector`, `u_comp`, `v_comp`, `wind_speed`, `wind_direction`, `helicity`, `wind_shear`, and `mean_wind` with `SHARP_HD`.
2. Support `StridedPtr<const float>` for `pres`, `hght`, `u_wind`, and `v_wind` arrays.

---

### Phase 3: Single-Pass Register-Fused Parcel & CAPE/CIN Engine

#### Task 3.1: Create `include/SHARPlib/gpu/fused_parcel.cuh`
CPU `SHARPlib` computes buoyancy across the entire column into a `float buoy_arr[N]` buffer and then calls `find_lfc_el` and `cape_cin`. On the GPU, this causes `.local` memory spills to global VRAM.

Implement `sharp::gpu::lift_and_integrate_column` as a single bottom-to-top pass using **only scalar registers**:

```cpp
namespace sharp::gpu {

struct ConvectiveOutput {
    float cape = 0.0f;
    float cin = 0.0f;
    float lcl_hght_agl = 0.0f;
    float lfc_hght_agl = 0.0f;
    float el_hght_agl = 0.0f;
};

template <typename Lifter, typename Ptr>
SHARP_HD ConvectiveOutput lift_and_integrate_column(
    const Lifter& lifter,
    Ptr pres, Ptr hght, Ptr tmpk, Ptr mixr, Ptr vtmpk,
    int N_z,
    float pcl_pres, float pcl_tmpk, float pcl_mixr) noexcept
{
    ConvectiveOutput out{};
    const float sfc_hght = hght[0];

    // 1. Lift dry adiabatically to LCL
    float lcl_pres = 0.0f, lcl_tmpk = 0.0f;
    sharp::drylift(pcl_pres, pcl_tmpk, pcl_mixr, lcl_pres, lcl_tmpk);
    if (lcl_pres <= pres[N_z - 1] || std::isnan(lcl_pres)) {
        return out;
    }

    // 2. Interpolate environmental height and virtual temp at exact LCL pressure
    float lcl_hght = sharp::interp_pressure(lcl_pres, pres, hght, N_z);
    float env_vtmpk_lcl = sharp::interp_pressure(lcl_pres, pres, vtmpk, N_z);
    float pcl_vtmpk_lcl = sharp::virtual_temperature(
        lcl_tmpk, sharp::mixratio(lcl_pres, lcl_tmpk)
    );
    out.lcl_hght_agl = lcl_hght - sfc_hght;

    // 3. Initialize state at LCL (incorporate sub-LCL CIN if parcel started below LCL)
    float p_prev = lcl_pres;
    float z_prev = lcl_hght;
    float t_pcl_prev = lcl_tmpk;
    float b_prev = sharp::buoyancy(pcl_vtmpk_lcl, env_vtmpk_lcl);

    float pos_buoy_Running = 0.0f;
    float neg_buoy_Running = 0.0f;
    float total_cape = 0.0f;
    float total_cin = 0.0f;
    bool has_lfc = false;

    // 4. Single upward pass from first level above LCL to top of profile
    for (int k = 0; k < N_z; ++k) {
        float p_curr = pres[k];
        if (p_curr >= lcl_pres) continue;

        float z_curr = hght[k];
        float t_pcl_curr = lifter(p_prev, t_pcl_prev, p_curr);
        float q_pcl_curr = sharp::mixratio(p_curr, t_pcl_curr);
        float tv_pcl_curr = sharp::virtual_temperature(t_pcl_curr, q_pcl_curr);
        float b_curr = sharp::buoyancy(tv_pcl_curr, vtmpk[k]);

        float dz = z_curr - z_prev;
        // Handle zero-crossings (LFC / EL) via linear interpolation within [z_prev, z_curr]
        if (b_prev <= 0.0f && b_curr > 0.0f) {
            // Crossing from negative to positive buoyancy (LFC candidate)
            float frac = -b_prev / (b_curr - b_prev);
            float z_cross = z_prev + frac * dz;
            neg_buoy_Running += 0.5f * b_prev * (z_cross - z_prev);
            pos_buoy_Running += 0.5f * b_curr * (z_curr - z_cross);
            out.lfc_hght_agl = z_cross - sfc_hght;
            has_lfc = true;
        } else if (b_prev > 0.0f && b_curr <= 0.0f) {
            // Crossing from positive to negative buoyancy (EL candidate)
            float frac = b_prev / (b_prev - b_curr);
            float z_cross = z_prev + frac * dz;
            pos_buoy_Running += 0.5f * b_prev * (z_cross - z_prev);
            neg_buoy_Running += 0.5f * b_curr * (z_curr - z_cross);

            // Commit accumulated positive area & lock in CIN below this EL
            total_cape += pos_buoy_Running;
            pos_buoy_Running = 0.0f;
            total_cin += neg_buoy_Running;
            neg_buoy_Running = 0.0f;
            out.el_hght_agl = z_cross - sfc_hght;
        } else {
            float area = 0.5f * (b_prev + b_curr) * dz;
            if (b_curr > 0.0f) {
                pos_buoy_Running += area;
            } else {
                neg_buoy_Running += area;
            }
        }

        p_prev = p_curr;
        z_prev = z_curr;
        t_pcl_prev = t_pcl_curr;
        b_prev = b_curr;
    }

    // If still positively buoyant at top of sounding, commit remaining CAPE
    if (pos_buoy_Running > 0.0f) {
        total_cape += pos_buoy_Running;
        total_cin += neg_buoy_Running;
        out.el_hght_agl = z_prev - sfc_hght;
    }

    if (total_cape > 0.0f) {
        out.cape = total_cape;
        out.cin = total_cin;
    }
    return out;
}

} // namespace sharp::gpu
```

> **Verification Requirement:** Ensure sub-LCL negative buoyancy between the parcel origination level and LCL is also integrated in registers using `drylift` temperature so `total_cin` matches `SHARPlib`'s `parcel::cape_cin` within $\pm 0.5\text{ J/kg}$.

---

### Phase 4: 2D CUDA Texture Lookup Table (`lifter_texture_gpu`)

Iterative root-finding lifters (`lifter_wobus` and `lifter_cm1`) cause warp divergence on GPUs because moist tropical profiles require more solver iterations than dry profiles.

#### Task 4.1: Build `lut_texture.cuh` and `lut_texture.cu`
1. Create a host-side RAII manager class `sharp::gpu::MoistAdiabatTextureLUT`:
   - Grid dimensions: `N_PRES = 256` (log-pressure space from $110000\text{ Pa}$ to $5000\text{ Pa}$) $\times$ `N_THETAE = 256` ($\theta_{e,\text{sat}}$ space from $220\text{ K}$ to $420\text{ K}$).
   - On construction on the CPU, use `sharp::lifter_cm1` (or `sharp::lifter_wobus`) to precompute the 2D table of parcel temperature $T(p, \theta_e)$.
   - Allocate a 2D `cudaArray_t` using `cudaMallocArray` with `cudaCreateChannelDesc<float>()`.
   - Copy the $256 \times 256$ float grid to the device via `cudaMemcpy2DToArray`.
   - Bind a `cudaTextureObject_t` with:
     - ` filterMode = cudaFilterModeLinear` (enables single-cycle hardware bilinear interpolation)
     - `addressMode[0] = cudaAddressModeClamp`
     - `addressMode[1] = cudaAddressModeClamp`
     - `normalizedCoords = 0` (use unnormalized pixel coordinates $+ 0.5\text{f}$ offset).
2. Create the lightweight device functor `sharp::gpu::lifter_texture_gpu`:
   ```cpp
   struct lifter_texture_gpu {
       cudaTextureObject_t tex;
       float log_p_min, inv_dlog_p;
       float thetae_min, inv_dthetae;

       SHARP_DEVICE SHARP_FORCEINLINE
       float operator()(float pres_start, float tmpk_start, float pres_end) const noexcept {
           // Compute conserved pseudo-equivalent potential temperature or moist static index
           // Or use 2-step lookup: (pres_start, tmpk_start) -> curve_id -> T(pres_end, curve_id)
           ...
           return tex2D<float>(tex, u + 0.5f, v + 0.5f);
       }
   };
   ```
   - **Important Tip:** Simply reuse `SHARPlib`'s existing `sharp::lifter_lut` logic from `include/SHARPlib/parcel.h`, replacing its software `interp_table_2d` bilinear interpolation with hardware `tex2D<float>(tex, u, v)`.
   - Keep `sharp::lifter_wobus` (annotated with `SHARP_HD`) available as a template parameter fallback so users can choose between `LifterMode::WOBUS_EXACT`, `LifterMode::CM1_EXACT`, and `LifterMode::LUT_TEXTURE_FAST`.

---

### Phase 5: Batched CUDA Kernels (`batch_kernels.cu`)

#### Task 5.1: Implement `convective_batch_kernel`
Create `src/SHARPlib/gpu/batch_kernels.cu` with a configurable batch kernel that computes all primary convective and kinematic parameters in one kernel launch:

1. **Thread Indexing & Guard:**
   ```cpp
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= n_profiles) return;
   ```
2. **Wrap Strided Pointers:**
   ```cpp
   sharp::StridedPtr<const float> pres{d_pres + idx, n_profiles};
   sharp::StridedPtr<const float> hght{d_hght + idx, n_profiles};
   sharp::StridedPtr<const float> tmpk{d_tmpk + idx, n_profiles};
   sharp::StridedPtr<const float> dwpk{d_dwpk + idx, n_profiles};
   sharp::StridedPtr<const float> u_wind{d_u + idx, n_profiles};
   sharp::StridedPtr<const float> v_wind{d_v + idx, n_profiles};
   ```
3. **Compute Derived Thermodynamic Quantities On-The-Fly:**
   - Rather than requiring the caller to pre-allocate 3D GPU arrays for `mixr` and `vtmpk`, either accept them optionally or provide an overload that computes `mixr(k)` and `vtmpk(k)` inline via a functor wrapper `VirtualTempAccessor`.
   - **Optimization:** Computing `mixr` and `vtmpk` once into a small pre-allocated `[N_levels, N_profiles]` device buffer (or computing inline) saves memory bandwidth. Benchmark both; for $N_z \le 60$, pre-computing `vtmpk` and `mixr` in a fused setup pass or on-the-fly functor avoids redundant `std::exp` calls during MU-CAPE loops.
4. **Compute Parcels per Thread:**
   - **Surface-Based Parcel (`SB`):** `pcl_pres = pres[0]`, `pcl_tmpk = tmpk[0]`, `pcl_mixr = mixr[0]`. Call `lift_and_integrate_column`.
   - **100-hPa Mixed-Layer Parcel (`ML`):** Average $\theta$ and mixing ratio over `[pres[0], pres[0] - 10000.0f]` in registers, then call `lift_and_integrate_column`.
   - **Most-Unstable Parcel (`MU`):** Support a `bool fast_mu_by_thetae` flag:
     - When `true` (default fast GPU mode): Scan levels within $300\text{ hPa}$ of the surface, find the level $k_{\text{max}}$ with maximum $\theta_e$, and lift **only** $k_{\text{max}}$ via `lift_and_integrate_column`.
     - When `false` (exact NSHARP mode): Loop over levels within $300\text{ hPa}$ of the surface and retain the parcel with maximum integrated CAPE.
5. **Compute Kinematic Parameters in the Same Kernel:**
   - Bunkers Storm Motion (`storm_motion_bunkers`)
   - $0\text{--}1\text{ km}$ and $0\text{--}3\text{ km}$ Storm-Relative Helicity (`helicity`)
   - $0\text{--}6\text{ km}$ Bulk Wind Difference (`wind_shear`)
   - Supercell Composite Parameter (`scp`) and Significant Tornado Parameter (`stp_fixed` / `stp_cin`).

---

### Phase 6: Zero-Copy Python Bindings (`nanobind` + DLPack)

#### Task 6.1: Create `src/nanobind/gpu_bindings.cpp`
Use `nanobind/ndarray.h` to accept **CuPy** (`nb::cupy`) or **PyTorch** (`nb::pytorch`) GPU arrays directly without copying data to host RAM:

```cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <SHARPlib/gpu/batch_kernels.h>

namespace nb = nanobind;

// Type alias for 2D Level-Major [N_levels, N_profiles] contiguous GPU float arrays
using GpuGrid2D = nb::ndarray<const float, nb::ndim<2>, nb::device::cuda, nb::c_contig>;
using GpuOut1D  = nb::ndarray<float, nb::ndim<1>, nb::device::cuda, nb::c_contig>;

void bind_gpu_ submodule(nb::module_& m) {
    auto gpu_mod = m.def_submodule("gpu", "GPU-accelerated batch sounding routines (CUDA)");

    gpu_mod.def("convective_batch", [](
        GpuGrid2D pres, GpuGrid2D hght, GpuGrid2D tmpk, GpuGrid2D dwpk,
        GpuGrid2D u_wind, GpuGrid2D v_wind,
        GpuOut1D out_sbcape, GpuOut1D out_sbcin,
        GpuOut1D out_mlcape, GpuOut1D out_mlcin,
        GpuOut1D out_mucape, GpuOut1D out_mucin,
        GpuOut1D out_srh01,  GpuOut1D out_srh03,
        GpuOut1D out_shr06,  GpuOut1D out_stp,
        bool use_texture_lut,
        bool fast_mu_thetae,
        std::uintptr_t cuda_stream_ptr) 
    {
        std::size_t n_levels   = pres.shape(0);
        std::size_t n_profiles = pres.shape(1);

        // Validate shapes of all inputs and outputs
        if (hght.shape(0) != n_levels || hght.shape(1) != n_profiles) {
            throw nb::buffer_error("All input 2D grids must have shape [N_levels, N_profiles]");
        }
        if (out_sbcape.shape(0) != n_profiles) {
            throw nb::buffer_error("All output 1D arrays must have shape [N_profiles]");
        }

        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_ptr);

        sharp::gpu::launch_convective_batch(
            pres.data(), hght.data(), tmpk.data(), dwpk.data(),
            u_wind.data(), v_wind.data(),
            out_sbcape.data(), out_sbcin.data(),
            out_mlcape.data(), out_mlcin.data(),
            out_mucape.data(), out_mucin.data(),
            out_srh01.data(),  out_srh03.data(),
            out_shr06.data(),  out_stp.data(),
            static_cast<int>(n_levels),
            static_cast<int>(n_profiles),
            use_texture_lut,
            fast_mu_thetae,
            stream
        );
    },
    nb::arg("pres"), nb::arg("hght"), nb::arg("tmpk"), nb::arg("dwpk"),
    nb::arg("u_wind"), nb::arg("v_wind"),
    nb::arg("out_sbcape"), nb::arg("out_sbcin"),
    nb::arg("out_mlcape"), nb::arg("out_mlcin"),
    nb::arg("out_mucape"), nb::arg("out_mucin"),
    nb::arg("out_srh01"),  nb::arg("out_srh03"),
    nb::arg("out_shr06"),  nb::arg("out_stp"),
    nb::arg("use_texture_lut") = true,
    nb::arg("fast_mu_thetae") = true,
    nb::arg("cuda_stream_ptr") = 0);
}
```
* Note: Using `nb::device::cuda` in `nanobind` makes the function accept **both** `cupy.ndarray` and `torch.Tensor(..., device='cuda')` interchangeably!

---

## 4. Verification, Profiling & Acceptance Criteria

### 4.1 Unit & Numerical Accuracy Tests (`tests/python/test_gpu_batch.py`)
The coding agent must create a pytest suite that:
1. Loads a reference sounding (or generates a synthetic 50-level atmospheric grid of $N_{\text{profiles}} = 10,000$ perturbed soundings covering dry, capped, tropical, and supercell regimes).
2. Runs the existing CPU `SHARPlib` 1D functions across a sample of profiles.
3. Runs `sharplib.gpu.convective_batch` on the GPU using CuPy/PyTorch tensors shaped `[N_levels, N_profiles]`.
4. Verifies numerical tolerances:
   - **Exact Wobus Mode (`use_texture_lut=False, fast_mu_thetae=False`):**
     - `SBCAPE`, `MLCAPE`, `MUCAPE` match CPU `SHARPlib` within **$\le 1.0\text{ J/kg}$** (or $0.1\%$ relative error).
     - `SRH` and `Bulk Shear` match CPU `SHARPlib` within **$\le 10^{-3}\text{ m}^2/\text{s}^2$**.
   - **Fast Texture LUT Mode (`use_texture_lut=True`):**
     - Parcel temperatures match exact lifter within **$\le 0.03\text{ K}$** and CAPE matches within **$\le 1.5\%$**.

### 4.2 Register Spill & Memory Coalescing Audit
The coding agent must verify compilation metrics by passing `-Xptxas=-v` to `nvcc`:
- **Acceptance Criterion 1 (Zero Spills):** `ptxas info` must report **`0 bytes spill stores, 0 bytes spill loads`** for `convective_batch_kernel`.
- **Acceptance Criterion 2 (Register Count):** Per-thread register usage should stay at or below **64 registers/thread** (`--maxrregcount=64`) to maintain $\ge 50\%$ theoretical SM occupancy at `blockDim = 256`.
- **Acceptance Criterion 3 (Throughput Benchmark):** On a full CONUS HRRR grid ($1799 \times 1059 \approx 1,905,141$ profiles $\times 50$ vertical levels), end-to-end kernel execution time (`SBCAPE`, `MLCAPE`, `MUCAPE`, `SRH01`, `SRH03`, `SHR06`, `STP`) with `use_texture_lut=True` must complete in **$< 50\text{ ms}$** on an modern mid-to-high-tier GPU (e.g., RTX 4070 / A100).

---

## 5. Recommended Agent Execution Order
1. **Commit 1:** Add `SHARP_HD` macros (`constants.h`) and `StridedPtr` (`include/SHARPlib/gpu/strided_ptr.h`). Ensure existing CPU CMake build and tests pass.
2. **Commit 2:** Annotate `interp.h`, `layer.h`, `thermo.h`, and `winds.h` with `SHARP_HD` and move required device functions inline into headers. Verify CPU tests still pass.
3. **Commit 3:** Implement `fused_parcel.cuh` (`lift_and_integrate_column`) and test it on the **CPU** first against `SHARPlib`'s existing two-pass `lifter` + `cape_cin` to prove numerical equivalence before touching CUDA.
4. **Commit 4:** Enable `SHARP_ENABLE_CUDA` in `CMakeLists.txt`, implement `lut_texture.cu` and `batch_kernels.cu`, and verify `0 bytes spill` via `-Xptxas=-v`.
5. **Commit 5:** Add `gpu_bindings.cpp` using `nb::device::cuda` and run `tests/python/test_gpu_batch.py` to validate accuracy and benchmark throughput.