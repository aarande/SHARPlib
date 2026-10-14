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

struct ConvectiveBatchOutputs {
    // Surface-Based (SB)
    float* sbcape = nullptr;
    float* sbcin = nullptr;
    float* sblcl_hght_agl = nullptr;
    float* sblcl_pres = nullptr;
    float* sblfc_hght_agl = nullptr;
    float* sblfc_pres = nullptr;
    float* sbel_hght_agl = nullptr;
    float* sbel_pres = nullptr;
    float* sb3cape = nullptr;

    // Mixed-Layer (ML - 100 hPa)
    float* mlcape = nullptr;
    float* mlcin = nullptr;
    float* mllcl_hght_agl = nullptr;
    float* mllcl_pres = nullptr;
    float* mllfc_hght_agl = nullptr;
    float* mllfc_pres = nullptr;
    float* mlel_hght_agl = nullptr;
    float* mlel_pres = nullptr;
    float* ml3cape = nullptr;

    // Most-Unstable (MU - within 300 hPa)
    float* mucape = nullptr;
    float* mucin = nullptr;
    float* mulcl_hght_agl = nullptr;
    float* mulcl_pres = nullptr;
    float* mulfc_hght_agl = nullptr;
    float* mulfc_pres = nullptr;
    float* muel_hght_agl = nullptr;
    float* muel_pres = nullptr;
    float* mu3cape = nullptr;

    // Kinematics & Storm Motion
    float* bunkers_rm_u = nullptr;
    float* bunkers_rm_v = nullptr;
    float* bunkers_lm_u = nullptr;
    float* bunkers_lm_v = nullptr;
    float* mean_wind_u = nullptr;
    float* mean_wind_v = nullptr;
    float* shr01 = nullptr;
    float* shr03 = nullptr;
    float* shr06 = nullptr;
    float* srh01 = nullptr;
    float* srh03 = nullptr;
    float* srh05 = nullptr;
    float* crit_angle = nullptr;

    // Thermodynamics & Moisture
    float* pwat = nullptr;
    float* lr03 = nullptr;
    float* lr36 = nullptr;
    float* lr75 = nullptr;
    float* fz_level_agl = nullptr;
    float* wbz_level_agl = nullptr;

    // Composite Severe Indices
    float* stp = nullptr;
    float* scp = nullptr;
    float* ehi01 = nullptr;
    float* ehi03 = nullptr;
};

/**
 * \brief Launch a batched convective analysis kernel across 2D [n_levels, n_profiles] soundings with flexible outputs.
 *
 * All pointer inputs are assumed to point to level-major contiguous device memory
 * of shape [n_levels, n_profiles] where the value for level k of profile idx is at
 * ptr[k * n_profiles + idx].
 *
 * All non-null pointers in outputs point to 1D device arrays of size [n_profiles].
 * Null pointers are skipped with selective execution pruning.
 */
void launch_convective_batch_flexible(
    const float* d_pres, const float* d_hght, const float* d_tmpk, const float* d_dwpk,
    const float* d_u, const float* d_v,
    const ConvectiveBatchOutputs& outputs,
    int n_levels, int n_profiles,
    bool use_texture_lut,
    bool fast_mu_thetae,
    cudaStream_t stream = 0
);

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
