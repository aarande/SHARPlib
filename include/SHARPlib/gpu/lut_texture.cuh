/**
 * \file
 * \brief 2D CUDA Texture Object manager for moist adiabat lookup
 * \author Kelton Halbert, Aaron Anderson
 * \date 2026-10-13
 */

#ifndef SHARP_GPU_LUT_TEXTURE_CUH
#define SHARP_GPU_LUT_TEXTURE_CUH

#include <cuda_runtime.h>
#include <SHARPlib/constants.h>
#include <SHARPlib/thermo.h>

#include <cmath>

namespace sharp::gpu {

/**
 * \brief Lightweight device-side functor for hardware texture-based parcel lifting.
 */
struct lifter_texture_gpu {
    cudaTextureObject_t tex = 0;
    float log_p_max = 0.0f;
    float inv_dlog_p = 0.0f;
    float thetae_min = 0.0f;
    float inv_dthetae = 0.0f;
    float cached_v = -1.0f;

    static constexpr bool lift_from_lcl = true;

    SHARP_HD inline void setup(const float lcl_pres, const float lcl_tmpk) noexcept {
#if defined(__CUDA_ARCH__)
        float u = (std::log(lcl_pres) - log_p_max) * inv_dlog_p;
        float v_lo = 0.0f;
        float v_hi = 255.0f;
        for (int step = 0; step < 16; ++step) {
            float v_mid = 0.5f * (v_lo + v_hi);
            float t_val = tex2D<float>(tex, u + 0.5f, v_mid + 0.5f);
            if (t_val > lcl_tmpk) {
                v_hi = v_mid;
            } else {
                v_lo = v_mid;
            }
        }
        cached_v = 0.5f * (v_lo + v_hi);
#else
        (void)lcl_pres;
        (void)lcl_tmpk;
#endif
    }

    SHARP_HD inline float operator()(
        [[maybe_unused]] float pres_start,
        [[maybe_unused]] float tmpk_start,
        float pres_end) const noexcept {
#if defined(__CUDA_ARCH__)
        float v = cached_v;
        if (v < 0.0f) {
            float u_start = (std::log(pres_start) - log_p_max) * inv_dlog_p;
            float v_lo = 0.0f, v_hi = 255.0f;
            for (int step = 0; step < 16; ++step) {
                float v_mid = 0.5f * (v_lo + v_hi);
                float t_val = tex2D<float>(tex, u_start + 0.5f, v_mid + 0.5f);
                if (t_val > tmpk_start) v_hi = v_mid; else v_lo = v_mid;
            }
            v = 0.5f * (v_lo + v_hi);
        }
        float u = (std::log(pres_end) - log_p_max) * inv_dlog_p;
        return tex2D<float>(tex, u + 0.5f, v + 0.5f);
#else
        (void)pres_start;
        (void)tmpk_start;
        (void)pres_end;
        return MISSING;
#endif
    }

    [[nodiscard]] SHARP_HD inline float parcel_virtual_temperature(
        const float pres, const float tmpk) const noexcept {
        return sharp::virtual_temperature(tmpk, sharp::mixratio(pres, tmpk));
    }
};

/**
 * \brief Host-side RAII manager for 2D moist adiabat hardware texture lookup table.
 */
class MoistAdiabatTextureLUT {
public:
    static constexpr int N_PRES = 256;
    static constexpr int N_THETAE = 256;
    static constexpr float PRES_MAX = 110000.0f;  // 1100 hPa
    static constexpr float PRES_MIN = 5000.0f;    // 50 hPa
    static constexpr float THETAE_MIN = 220.0f;   // 220 K
    static constexpr float THETAE_MAX = 420.0f;   // 420 K

    MoistAdiabatTextureLUT();
    ~MoistAdiabatTextureLUT();

    MoistAdiabatTextureLUT(const MoistAdiabatTextureLUT&) = delete;
    MoistAdiabatTextureLUT& operator=(const MoistAdiabatTextureLUT&) = delete;

    MoistAdiabatTextureLUT(MoistAdiabatTextureLUT&& other) noexcept;
    MoistAdiabatTextureLUT& operator=(MoistAdiabatTextureLUT&& other) noexcept;

    [[nodiscard]] lifter_texture_gpu get_device_lifter() const noexcept;
    [[nodiscard]] cudaTextureObject_t get_texture_object() const noexcept { return m_texObj; }

private:
    cudaArray_t m_cuArray = nullptr;
    cudaTextureObject_t m_texObj = 0;
    float m_log_p_max = 0.0f;
    float m_inv_dlog_p = 0.0f;
    float m_thetae_min = 0.0f;
    float m_inv_dthetae = 0.0f;
};

/**
 * \brief Get or lazily create the process-wide singleton MoistAdiabatTextureLUT.
 */
MoistAdiabatTextureLUT& get_global_texture_lut();

} // namespace sharp::gpu

#endif // SHARP_GPU_LUT_TEXTURE_CUH
