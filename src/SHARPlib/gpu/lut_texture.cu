/**
 * \file
 * \brief Host LUT generation and cudaTextureObject_t management for moist adiabat lookup
 * \author Kelton Halbert, Aaron Anderson
 * \date 2026-10-13
 */

#include <SHARPlib/gpu/lut_texture.cuh>
#include <SHARPlib/parcel.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace sharp::gpu {

namespace {

static float solve_tmpk_for_thetae(float pres, float target_thetae) {
    float lo = 180.0f;
    float hi = 360.0f;
    float mid = 273.15f;
    for (int iter = 0; iter < 60; ++iter) {
        mid = 0.5f * (lo + hi);
        float th_e = sharp::thetae(pres, mid, mid);
        if (std::abs(th_e - target_thetae) < 0.001f) break;
        if (th_e < target_thetae) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return mid;
}

} // anonymous namespace

MoistAdiabatTextureLUT::MoistAdiabatTextureLUT() {
    m_log_p_max = std::log(PRES_MAX);
    const float log_p_min = std::log(PRES_MIN);
    const float delta_log_p = (log_p_min - m_log_p_max) / static_cast<float>(N_PRES - 1);
    m_inv_dlog_p = 1.0f / delta_log_p;

    m_thetae_min = THETAE_MIN;
    const float delta_thetae = (THETAE_MAX - THETAE_MIN) / static_cast<float>(N_THETAE - 1);
    m_inv_dthetae = 1.0f / delta_thetae;

    std::vector<float> host_grid(static_cast<std::size_t>(N_PRES) * N_THETAE);
    sharp::lifter_wobus wobus;

    for (int i = 0; i < N_THETAE; ++i) {
        const float thetae_i = THETAE_MIN + static_cast<float>(i) * delta_thetae;
        const float tmpk_start = solve_tmpk_for_thetae(PRES_MAX, thetae_i);

        float pres_curr = PRES_MAX;
        float tmpk_curr = tmpk_start;
        host_grid[static_cast<std::size_t>(i) * N_PRES + 0] = tmpk_curr;

        for (int k = 1; k < N_PRES; ++k) {
            const float pres_next = std::exp(m_log_p_max + static_cast<float>(k) * delta_log_p);
            tmpk_curr = wobus(pres_curr, tmpk_curr, pres_next);
            host_grid[static_cast<std::size_t>(i) * N_PRES + k] = tmpk_curr;
            pres_curr = pres_next;
        }
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaError_t err = cudaMallocArray(&m_cuArray, &channelDesc, N_PRES, N_THETAE);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMallocArray failed: ") + cudaGetErrorString(err));
    }

    err = cudaMemcpy2DToArray(m_cuArray, 0, 0, host_grid.data(),
                              N_PRES * sizeof(float), N_PRES * sizeof(float),
                              N_THETAE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFreeArray(m_cuArray);
        m_cuArray = nullptr;
        throw std::runtime_error(std::string("cudaMemcpy2DToArray failed: ") + cudaGetErrorString(err));
    }

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_cuArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    err = cudaCreateTextureObject(&m_texObj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        cudaFreeArray(m_cuArray);
        m_cuArray = nullptr;
        throw std::runtime_error(std::string("cudaCreateTextureObject failed: ") + cudaGetErrorString(err));
    }
}

MoistAdiabatTextureLUT::~MoistAdiabatTextureLUT() {
    if (m_texObj) {
        cudaDestroyTextureObject(m_texObj);
        m_texObj = 0;
    }
    if (m_cuArray) {
        cudaFreeArray(m_cuArray);
        m_cuArray = nullptr;
    }
}

MoistAdiabatTextureLUT::MoistAdiabatTextureLUT(MoistAdiabatTextureLUT&& other) noexcept
    : m_cuArray(other.m_cuArray),
      m_texObj(other.m_texObj),
      m_log_p_max(other.m_log_p_max),
      m_inv_dlog_p(other.m_inv_dlog_p),
      m_thetae_min(other.m_thetae_min),
      m_inv_dthetae(other.m_inv_dthetae) {
    other.m_cuArray = nullptr;
    other.m_texObj = 0;
}

MoistAdiabatTextureLUT& MoistAdiabatTextureLUT::operator=(MoistAdiabatTextureLUT&& other) noexcept {
    if (this != &other) {
        if (m_texObj) cudaDestroyTextureObject(m_texObj);
        if (m_cuArray) cudaFreeArray(m_cuArray);

        m_cuArray = other.m_cuArray;
        m_texObj = other.m_texObj;
        m_log_p_max = other.m_log_p_max;
        m_inv_dlog_p = other.m_inv_dlog_p;
        m_thetae_min = other.m_thetae_min;
        m_inv_dthetae = other.m_inv_dthetae;

        other.m_cuArray = nullptr;
        other.m_texObj = 0;
    }
    return *this;
}

lifter_texture_gpu MoistAdiabatTextureLUT::get_device_lifter() const noexcept {
    lifter_texture_gpu lifter;
    lifter.tex = m_texObj;
    lifter.log_p_max = m_log_p_max;
    lifter.inv_dlog_p = m_inv_dlog_p;
    lifter.thetae_min = m_thetae_min;
    lifter.inv_dthetae = m_inv_dthetae;
    lifter.cached_thetae = 0.0f;
    return lifter;
}

MoistAdiabatTextureLUT& get_global_texture_lut() {
    static MoistAdiabatTextureLUT s_lut;
    return s_lut;
}

} // namespace sharp::gpu
