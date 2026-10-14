/**
 * \file
 * \brief Nanobind DLPack bindings for GPU-accelerated batch sounding routines
 * \author Kelton Halbert, Aaron Anderson
 * \date 2026-10-13
 */

#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <SHARPlib/gpu/batch_kernels.h>

namespace nb = nanobind;

// Type alias for 2D Level-Major [N_levels, N_profiles] contiguous GPU float arrays
using GpuGrid2D = nb::ndarray<const float, nb::ndim<2>, nb::device::cuda, nb::c_contig>;
using GpuOut1D  = nb::ndarray<float, nb::ndim<1>, nb::device::cuda, nb::c_contig>;

void bind_gpu_submodule(nb::module_& m) {
    auto gpu_mod = m.def_submodule("gpu", "GPU-accelerated batch sounding routines (CUDA)");

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
}
