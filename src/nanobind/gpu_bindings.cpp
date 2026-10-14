/**
 * \file
 * \brief Nanobind DLPack bindings for GPU-accelerated batch sounding routines
 * \author Kelton Halbert, Aaron Anderson
 * \date 2026-10-13
 */

#include <cstdint>
#include <string>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eval.h>
#include <nanobind/stl/string.h>
#include <SHARPlib/gpu/batch_kernels.h>


namespace nb = nanobind;

// Type alias for 2D Level-Major [N_levels, N_profiles] contiguous GPU float arrays
using GpuGrid2D = nb::ndarray<const float, nb::ndim<2>, nb::device::cuda, nb::c_contig>;
using GpuOut1D  = nb::ndarray<float, nb::ndim<1>, nb::device::cuda, nb::c_contig>;

static const char* const SUPPORTED_PARAMS_LIST[] = {
    // Surface-Based (SB)
    "sbcape",
    "sbcin",
    "sblcl_hght_agl",
    "sblcl_pres",
    "sblfc_hght_agl",
    "sblfc_pres",
    "sbel_hght_agl",
    "sbel_pres",
    "sb3cape",

    // Mixed-Layer (ML - 100 hPa)
    "mlcape",
    "mlcin",
    "mllcl_hght_agl",
    "mllcl_pres",
    "mllfc_hght_agl",
    "mllfc_pres",
    "mlel_hght_agl",
    "mlel_pres",
    "ml3cape",

    // Most-Unstable (MU - within 300 hPa)
    "mucape",
    "mucin",
    "mulcl_hght_agl",
    "mulcl_pres",
    "mulfc_hght_agl",
    "mulfc_pres",
    "muel_hght_agl",
    "muel_pres",
    "mu3cape",

    // Kinematics & Storm Motion
    "bunkers_rm_u",
    "bunkers_rm_v",
    "bunkers_lm_u",
    "bunkers_lm_v",
    "mean_wind_u",
    "mean_wind_v",
    "shr01",
    "shr03",
    "shr06",
    "srh01",
    "srh03",
    "srh05",
    "crit_angle",

    // Thermodynamics & Moisture
    "pwat",
    "lr03",
    "lr36",
    "lr75",
    "fz_level_agl",
    "wbz_level_agl",

    // Composite Severe Indices
    "stp",
    "scp",
    "ehi01",
    "ehi03"
};

void bind_gpu_submodule(nb::module_& m) {
    auto gpu_mod = m.def_submodule("gpu", "GPU-accelerated batch sounding routines (CUDA)");

    // Expose SUPPORTED_PARAMS tuple
    nb::list params_list;
    for (const char* param_name : SUPPORTED_PARAMS_LIST) {
        params_list.append(param_name);
    }
    gpu_mod.attr("SUPPORTED_PARAMS") = nb::tuple(params_list);

    gpu_mod.def(
        "convective_batch",
        [](GpuGrid2D pres, GpuGrid2D hght, GpuGrid2D tmpk, GpuGrid2D dwpk,
           GpuGrid2D u_wind, GpuGrid2D v_wind,
           GpuOut1D out_sbcape, GpuOut1D out_sbcin,
           GpuOut1D out_mlcape, GpuOut1D out_mlcin,
           GpuOut1D out_mucape, GpuOut1D out_mucin,
           GpuOut1D out_srh01,  GpuOut1D out_srh03,
           GpuOut1D out_shr06,  GpuOut1D out_stp,
           bool use_texture_lut,
           bool fast_mu_thetae,
           std::uintptr_t cuda_stream_ptr) {
            std::size_t n_levels   = pres.shape(0);
            std::size_t n_profiles = pres.shape(1);

            // Validate shapes of all inputs
            if (hght.shape(0) != n_levels || hght.shape(1) != n_profiles ||
                tmpk.shape(0) != n_levels || tmpk.shape(1) != n_profiles ||
                dwpk.shape(0) != n_levels || dwpk.shape(1) != n_profiles ||
                u_wind.shape(0) != n_levels || u_wind.shape(1) != n_profiles ||
                v_wind.shape(0) != n_levels || v_wind.shape(1) != n_profiles) {
                throw nb::buffer_error("All input 2D grids must have shape [N_levels, N_profiles]");
            }

            // Validate shapes of all outputs
            if (out_sbcape.shape(0) != n_profiles ||
                out_sbcin.shape(0) != n_profiles ||
                out_mlcape.shape(0) != n_profiles ||
                out_mlcin.shape(0) != n_profiles ||
                out_mucape.shape(0) != n_profiles ||
                out_mucin.shape(0) != n_profiles ||
                out_srh01.shape(0) != n_profiles ||
                out_srh03.shape(0) != n_profiles ||
                out_shr06.shape(0) != n_profiles ||
                out_stp.shape(0) != n_profiles) {
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
        nb::arg("cuda_stream_ptr") = 0,
        "Compute convective and kinematic sounding parameters across batched 2D GPU grids."
    );

    gpu_mod.def(
        "convective_batch_flexible",
        [](GpuGrid2D pres, GpuGrid2D hght, GpuGrid2D tmpk, GpuGrid2D dwpk,
           GpuGrid2D u_wind, GpuGrid2D v_wind,
           nb::dict outputs,
           bool use_texture_lut,
           bool fast_mu_thetae,
           std::uintptr_t cuda_stream_ptr) {
            std::size_t n_levels   = pres.shape(0);
            std::size_t n_profiles = pres.shape(1);

            // Validate shapes of all inputs
            if (hght.shape(0) != n_levels || hght.shape(1) != n_profiles ||
                tmpk.shape(0) != n_levels || tmpk.shape(1) != n_profiles ||
                dwpk.shape(0) != n_levels || dwpk.shape(1) != n_profiles ||
                u_wind.shape(0) != n_levels || u_wind.shape(1) != n_profiles ||
                v_wind.shape(0) != n_levels || v_wind.shape(1) != n_profiles) {
                throw nb::buffer_error("All input 2D grids must have shape [N_levels, N_profiles]");
            }

            sharp::gpu::ConvectiveBatchOutputs batch_outputs{};
            std::vector<GpuOut1D> keep_alive;
            keep_alive.reserve(outputs.size());

            for (auto item : outputs) {
                std::string key = nb::cast<std::string>(item.first);
                GpuOut1D arr;
                try {
                    arr = nb::cast<GpuOut1D>(item.second);
                } catch (const std::exception&) {
                    throw nb::type_error(("Output for '" + key + "' must be a 1D C-contiguous CUDA float array").c_str());
                }

                if (arr.shape(0) != n_profiles) {
                    throw nb::buffer_error(("Output 1D array for '" + key + "' must have shape [N_profiles]").c_str());
                }

                keep_alive.push_back(arr);
                float* ptr = arr.data();

                // Surface-Based (SB)
                if (key == "sbcape") batch_outputs.sbcape = ptr;
                else if (key == "sbcin") batch_outputs.sbcin = ptr;
                else if (key == "sblcl_hght_agl") batch_outputs.sblcl_hght_agl = ptr;
                else if (key == "sblcl_pres") batch_outputs.sblcl_pres = ptr;
                else if (key == "sblfc_hght_agl") batch_outputs.sblfc_hght_agl = ptr;
                else if (key == "sblfc_pres") batch_outputs.sblfc_pres = ptr;
                else if (key == "sbel_hght_agl") batch_outputs.sbel_hght_agl = ptr;
                else if (key == "sbel_pres") batch_outputs.sbel_pres = ptr;
                else if (key == "sb3cape") batch_outputs.sb3cape = ptr;

                // Mixed-Layer (ML)
                else if (key == "mlcape") batch_outputs.mlcape = ptr;
                else if (key == "mlcin") batch_outputs.mlcin = ptr;
                else if (key == "mllcl_hght_agl") batch_outputs.mllcl_hght_agl = ptr;
                else if (key == "mllcl_pres") batch_outputs.mllcl_pres = ptr;
                else if (key == "mllfc_hght_agl") batch_outputs.mllfc_hght_agl = ptr;
                else if (key == "mllfc_pres") batch_outputs.mllfc_pres = ptr;
                else if (key == "mlel_hght_agl") batch_outputs.mlel_hght_agl = ptr;
                else if (key == "mlel_pres") batch_outputs.mlel_pres = ptr;
                else if (key == "ml3cape") batch_outputs.ml3cape = ptr;

                // Most-Unstable (MU)
                else if (key == "mucape") batch_outputs.mucape = ptr;
                else if (key == "mucin") batch_outputs.mucin = ptr;
                else if (key == "mulcl_hght_agl") batch_outputs.mulcl_hght_agl = ptr;
                else if (key == "mulcl_pres") batch_outputs.mulcl_pres = ptr;
                else if (key == "mulfc_hght_agl") batch_outputs.mulfc_hght_agl = ptr;
                else if (key == "mulfc_pres") batch_outputs.mulfc_pres = ptr;
                else if (key == "muel_hght_agl") batch_outputs.muel_hght_agl = ptr;
                else if (key == "muel_pres") batch_outputs.muel_pres = ptr;
                else if (key == "mu3cape") batch_outputs.mu3cape = ptr;

                // Kinematics & Storm Motion
                else if (key == "bunkers_rm_u") batch_outputs.bunkers_rm_u = ptr;
                else if (key == "bunkers_rm_v") batch_outputs.bunkers_rm_v = ptr;
                else if (key == "bunkers_lm_u") batch_outputs.bunkers_lm_u = ptr;
                else if (key == "bunkers_lm_v") batch_outputs.bunkers_lm_v = ptr;
                else if (key == "mean_wind_u") batch_outputs.mean_wind_u = ptr;
                else if (key == "mean_wind_v") batch_outputs.mean_wind_v = ptr;
                else if (key == "shr01") batch_outputs.shr01 = ptr;
                else if (key == "shr03") batch_outputs.shr03 = ptr;
                else if (key == "shr06") batch_outputs.shr06 = ptr;
                else if (key == "srh01") batch_outputs.srh01 = ptr;
                else if (key == "srh03") batch_outputs.srh03 = ptr;
                else if (key == "srh05") batch_outputs.srh05 = ptr;
                else if (key == "crit_angle") batch_outputs.crit_angle = ptr;

                // Thermodynamics & Moisture
                else if (key == "pwat") batch_outputs.pwat = ptr;
                else if (key == "lr03") batch_outputs.lr03 = ptr;
                else if (key == "lr36") batch_outputs.lr36 = ptr;
                else if (key == "lr75") batch_outputs.lr75 = ptr;
                else if (key == "fz_level_agl") batch_outputs.fz_level_agl = ptr;
                else if (key == "wbz_level_agl") batch_outputs.wbz_level_agl = ptr;

                // Composite Severe Indices
                else if (key == "stp") batch_outputs.stp = ptr;
                else if (key == "scp") batch_outputs.scp = ptr;
                else if (key == "ehi01") batch_outputs.ehi01 = ptr;
                else if (key == "ehi03") batch_outputs.ehi03 = ptr;
                else {
                    throw nb::value_error(("Unknown parameter name: " + key).c_str());
                }
            }

            cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_ptr);

            sharp::gpu::launch_convective_batch_flexible(
                pres.data(), hght.data(), tmpk.data(), dwpk.data(),
                u_wind.data(), v_wind.data(),
                batch_outputs,
                static_cast<int>(n_levels),
                static_cast<int>(n_profiles),
                use_texture_lut,
                fast_mu_thetae,
                stream
            );
        },
        nb::arg("pres"), nb::arg("hght"), nb::arg("tmpk"), nb::arg("dwpk"),
        nb::arg("u_wind"), nb::arg("v_wind"),
        nb::arg("outputs"),
        nb::arg("use_texture_lut") = true,
        nb::arg("fast_mu_thetae") = true,
        nb::arg("cuda_stream_ptr") = 0,
        "Compute selected convective and kinematic sounding parameters across batched 2D GPU grids."
    );

    // High-level Python API compute_sounding_grid
    nb::exec(R"python(
def compute_sounding_grid(
    pres,
    hght,
    tmpk,
    dwpk,
    u_wind,
    v_wind,
    params="all",
    use_texture_lut=True,
    fast_mu_thetae=True,
    stream=None
):
    """
    Compute convective, kinematic, thermodynamic, and composite parameters across
    a batched 2D grid of vertical atmospheric soundings on the GPU.

    Parameters
    ----------
    pres : CuPy ndarray or PyTorch CUDA Tensor of shape [N_levels, N_profiles]
        Pressure grid (Pa). Must be C-contiguous.
    hght : CuPy ndarray or PyTorch CUDA Tensor of shape [N_levels, N_profiles]
        Height grid (m MSL). Must be C-contiguous.
    tmpk : CuPy ndarray or PyTorch CUDA Tensor of shape [N_levels, N_profiles]
        Temperature grid (K). Must be C-contiguous.
    dwpk : CuPy ndarray or PyTorch CUDA Tensor of shape [N_levels, N_profiles]
        Dewpoint temperature grid (K). Must be C-contiguous.
    u_wind : CuPy ndarray or PyTorch CUDA Tensor of shape [N_levels, N_profiles]
        U-wind component (m/s). Must be C-contiguous.
    v_wind : CuPy ndarray or PyTorch CUDA Tensor of shape [N_levels, N_profiles]
        V-wind component (m/s). Must be C-contiguous.
    params : str or Iterable[str] or None, default="all"
        Parameters to compute. If "all" or None, all supported parameters are computed.
        Otherwise, an iterable of parameter names selected from SUPPORTED_PARAMS.
    use_texture_lut : bool, default=True
        If True, use hardware 2D texture cache for moist pseudoadiabatic ascent.
        If False, use exact Wobus polynomial lifter.
    fast_mu_thetae : bool, default=True
        If True, choose most unstable parcel by maximum theta-e in lowest 300 hPa.
        If False, lift parcels from all levels in lowest 300 hPa.
    stream : CuPy Stream or PyTorch CUDA Stream or int, optional
        CUDA stream to use for asynchronous kernel execution.

    Returns
    -------
    dict[str, Array/Tensor]
        Dictionary mapping requested parameter name strings to 1D device arrays
        of shape [N_profiles]. The returned arrays match the input tensor framework
        (CuPy or PyTorch).
    """
    supported_set = set(SUPPORTED_PARAMS)
    if params == "all" or params is None:
        target_params = list(SUPPORTED_PARAMS)
    elif isinstance(params, str):
        if params not in supported_set:
            raise ValueError(f"Unknown parameter name: {params!r}. Must be one of {SUPPORTED_PARAMS}")
        target_params = [params]
    else:
        target_params = list(params)
        for p in target_params:
            if p not in supported_set:
                raise ValueError(f"Unknown parameter name: {p!r}. Must be one of {SUPPORTED_PARAMS}")

    if not hasattr(pres, "shape") or len(pres.shape) != 2:
        raise ValueError("Input 'pres' must be a 2D grid of shape [N_levels, N_profiles]")

    n_levels, n_profiles = pres.shape

    # Framework auto-detection
    is_torch = False
    if hasattr(pres, "is_cuda") and pres.is_cuda:
        is_torch = True
    else:
        try:
            import torch
            if torch.is_tensor(pres) and pres.is_cuda:
                is_torch = True
        except ImportError:
            pass

    is_cupy = False
    if not is_torch:
        if hasattr(pres, "__cuda_array_interface__"):
            is_cupy = True
        else:
            try:
                import cupy as cp
                if isinstance(pres, cp.ndarray):
                    is_cupy = True
            except ImportError:
                pass

    if not is_torch and not is_cupy:
        raise TypeError(
            "Inputs must be either CuPy ndarrays or PyTorch CUDA tensors."
        )

    output_dict = {}
    if is_torch:
        import torch
        for name in target_params:
            output_dict[name] = torch.empty(n_profiles, dtype=torch.float32, device=pres.device)
    else:
        import cupy as cp
        for name in target_params:
            output_dict[name] = cp.empty(n_profiles, dtype=cp.float32)

    if stream is None:
        stream_ptr = 0
    elif hasattr(stream, "ptr"):
        stream_ptr = int(stream.ptr)
    elif hasattr(stream, "cuda_stream"):
        stream_ptr = int(stream.cuda_stream)
    elif isinstance(stream, int):
        stream_ptr = stream
    else:
        stream_ptr = 0

    convective_batch_flexible(
        pres, hght, tmpk, dwpk, u_wind, v_wind,
        output_dict,
        use_texture_lut=use_texture_lut,
        fast_mu_thetae=fast_mu_thetae,
        cuda_stream_ptr=stream_ptr
    )
    return output_dict
)python", gpu_mod.attr("__dict__"));

    m.attr("compute_sounding_grid") = gpu_mod.attr("compute_sounding_grid");
    m.attr("SUPPORTED_PARAMS") = gpu_mod.attr("SUPPORTED_PARAMS");
}

