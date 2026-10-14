/**
 * \file
 * \brief Host-callable C++ declarations for GPU batch sounding kernels
 * \author Kelton Halbert, Aaron Anderson
 * \date 2026-10-13
 */

#ifndef SHARP_GPU_BATCH_KERNELS_H
#define SHARP_GPU_BATCH_KERNELS_H

#include <cuda_runtime.h>

namespace sharp::gpu {

/**
 * \brief Launch a batched convective analysis kernel across 2D [n_levels, n_profiles] soundings.
 *
 * All pointer inputs are assumed to point to level-major contiguous device memory
 * of shape [n_levels, n_profiles] where the value for level k of profile idx is at
 * ptr[k * n_profiles + idx].
 *
 * All pointer outputs point to 1D device arrays of size [n_profiles].
 *
 * \param d_pres          Pressure grid (Pa) [n_levels, n_profiles]
 * \param d_hght          Height grid (m MSL) [n_levels, n_profiles]
 * \param d_tmpk          Temperature grid (K) [n_levels, n_profiles]
 * \param d_dwpk          Dewpoint temperature grid (K) [n_levels, n_profiles]
 * \param d_u             U-wind component (m/s) [n_levels, n_profiles]
 * \param d_v             V-wind component (m/s) [n_levels, n_profiles]
 * \param d_out_sbcape    Surface-Based CAPE (J/kg) [n_profiles]
 * \param d_out_sbcin     Surface-Based CIN (J/kg) [n_profiles]
 * \param d_out_mlcape    100-hPa Mixed-Layer CAPE (J/kg) [n_profiles]
 * \param d_out_mlcin     100-hPa Mixed-Layer CIN (J/kg) [n_profiles]
 * \param d_out_mucape    Most-Unstable CAPE (J/kg) [n_profiles]
 * \param d_out_mucin     Most-Unstable CIN (J/kg) [n_profiles]
 * \param d_out_srh01     0-1 km Storm-Relative Helicity (m^2/s^2) [n_profiles]
 * \param d_out_srh03     0-3 km Storm-Relative Helicity (m^2/s^2) [n_profiles]
 * \param d_out_shr06     0-6 km Bulk Wind Shear magnitude (m/s) [n_profiles]
 * \param d_out_stp       Fixed-layer Significant Tornado Parameter [n_profiles]
 * \param n_levels        Number of vertical sounding levels
 * \param n_profiles      Number of horizontal sounding profiles
 * \param use_texture_lut If true, use hardware 2D texture LUT for moist ascent; otherwise Wobus
 * \param fast_mu_thetae  If true, pick MU parcel via max theta-e; otherwise lift all parcels in lowest 300 hPa
 * \param stream          CUDA stream for kernel launch
 */
void launch_convective_batch(
    const float* d_pres, const float* d_hght, const float* d_tmpk, const float* d_dwpk,
    const float* d_u, const float* d_v,
    float* d_out_sbcape, float* d_out_sbcin,
    float* d_out_mlcape, float* d_out_mlcin,
    float* d_out_mucape, float* d_out_mucin,
    float* d_out_srh01,  float* d_out_srh03,
    float* d_out_shr06,  float* d_out_stp,
    int n_levels, int n_profiles,
    bool use_texture_lut,
    bool fast_mu_thetae,
    cudaStream_t stream = 0
);

} // namespace sharp::gpu

#endif // SHARP_GPU_BATCH_KERNELS_H
