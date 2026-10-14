/**
 * \file
 * \brief Batched CUDA convective sounding kernels
 * \author Kelton Halbert, Aaron Anderson
 * \date 2026-10-13
 */

#include <SHARPlib/gpu/batch_kernels.h>
#include <SHARPlib/gpu/fused_parcel.cuh>
#include <SHARPlib/gpu/lut_texture.cuh>
#include <SHARPlib/gpu/strided_ptr.h>
#include <SHARPlib/layer.h>
#include <SHARPlib/parcel.h>
#include <SHARPlib/thermo.h>
#include <SHARPlib/winds.h>

#include <cmath>

namespace sharp::gpu {

namespace {

struct EnvVtmpkAccessor {
    sharp::StridedPtr<const float> pres;
    sharp::StridedPtr<const float> tmpk;
    sharp::StridedPtr<const float> dwpk;
    SHARP_HD inline float operator[](std::ptrdiff_t k) const {
        float q = sharp::mixratio(pres[k], dwpk[k]);
        return sharp::virtual_temperature(tmpk[k], q);
    }
};

struct MixrAccessor {
    sharp::StridedPtr<const float> pres;
    sharp::StridedPtr<const float> dwpk;
    SHARP_HD inline float operator[](std::ptrdiff_t k) const {
        return sharp::mixratio(pres[k], dwpk[k]);
    }
};

struct ThetaAccessor {
    sharp::StridedPtr<const float> pres;
    sharp::StridedPtr<const float> tmpk;
    SHARP_HD inline float operator[](std::ptrdiff_t k) const {
        return sharp::theta(pres[k], tmpk[k], sharp::THETA_REF_PRESSURE);
    }
};

SHARP_HD SHARP_FORCEINLINE float compute_stp(
    float cape, float cinh, float lcl_hght_agl, float srh, float bulk_wind_diff) noexcept
{
    if ((cinh != cinh) || (cape != cape) || sharp::is_missing(cape)) return 0.0f;

    float cinh_term;
    if (cinh >= -50.0f)
        cinh_term = 1.0f;
    else if (cinh < -200.0f)
        cinh_term = 0.0f;
    else
        cinh_term = ((200.0f + cinh) / 150.0f);

    float lcl_term;
    if (lcl_hght_agl < 1000.0f)
        lcl_term = 1.0f;
    else if (lcl_hght_agl > 2000.0f)
        lcl_term = 0.0f;
    else
        lcl_term = ((2000.0f - lcl_hght_agl) / 1000.0f);

    float shear_term;
    if (bulk_wind_diff > 30.0f)
        shear_term = 1.5f;
    else if (bulk_wind_diff < 12.5f)
        shear_term = 0.0f;
    else
        shear_term = bulk_wind_diff / 20.0f;

    float srh_term;
    if (sharp::is_missing(srh) || (srh != srh))
        srh_term = 0.0f;
    else
        srh_term = srh / 150.0f;

    float cape_term = cape / 1500.0f;
    float stp = cape_term * cinh_term * lcl_term * srh_term * shear_term;
    if (stp < 0.0f || (stp != stp)) stp = 0.0f;
    return stp;
}

template <typename Lifter>
__device__ __noinline__ void run_sb_parcel(
    int idx, int n_levels, Lifter base_lifter,
    sharp::StridedPtr<const float> pres, sharp::StridedPtr<const float> hght,
    sharp::StridedPtr<const float> tmpk, sharp::StridedPtr<const float> dwpk,
    float* d_out_sbcape, float* d_out_sbcin,
    float& sb_cape, float& sb_cin, float& sb_lcl_agl)
{
    float pcl_pres = pres[0];
    float pcl_tmpk = tmpk[0];
    float pcl_mixr = sharp::mixratio(pcl_pres, dwpk[0]);
    EnvVtmpkAccessor vtmpk_acc{pres, tmpk, dwpk};
    MixrAccessor mixr_acc{pres, dwpk};
    Lifter lft = base_lifter;
    sharp::gpu::ConvectiveOutput sb_out = sharp::gpu::lift_and_integrate_column(
        lft, pcl_pres, pcl_tmpk, pcl_mixr,
        pres, hght, tmpk, mixr_acc, vtmpk_acc, n_levels);
    sb_cape = sb_out.cape;
    sb_cin = sb_out.cin;
    sb_lcl_agl = sb_out.lcl_hght_agl;
    if (d_out_sbcape) d_out_sbcape[idx] = sb_cape;
    if (d_out_sbcin) d_out_sbcin[idx] = sb_cin;
}

template <typename PresPtr, typename ThtaPtr, typename MixrPtr>
__device__ inline void compute_ml_mean(
    PresPtr pres, ThtaPtr theta_acc, MixrPtr mixr_acc, int n_levels,
    float& mean_thta, float& mean_mixr)
{
    float p_bot = pres[0];
    float p_top = p_bot - 10000.0f;
    if (p_top < pres[n_levels - 1]) p_top = pres[n_levels - 1];

    float th_integ = 0.0f;
    float w_integ = 0.0f;
    float dp_total = 0.0f;

    float p_prev = p_bot;
    float th_prev = theta_acc[0];
    float w_prev = mixr_acc[0];

    for (int k = 1; k < n_levels; ++k) {
        float p_curr = pres[k];
        if (p_curr < p_top) {
            float frac = (p_top - p_prev) / (p_curr - p_prev);
            float th_top = th_prev + frac * (theta_acc[k] - th_prev);
            float w_top = w_prev + frac * (mixr_acc[k] - w_prev);
            float dp = p_prev - p_top;
            th_integ += 0.5f * (th_prev + th_top) * dp;
            w_integ += 0.5f * (w_prev + w_top) * dp;
            dp_total += dp;
            break;
        }

        float th_curr = theta_acc[k];
        float w_curr = mixr_acc[k];
        float dp = p_prev - p_curr;
        th_integ += 0.5f * (th_prev + th_curr) * dp;
        w_integ += 0.5f * (w_prev + w_curr) * dp;
        dp_total += dp;

        p_prev = p_curr;
        th_prev = th_curr;
        w_prev = w_curr;

        if (p_curr == p_top) break;
    }

    if (dp_total > 0.0f) {
        mean_thta = th_integ / dp_total;
        mean_mixr = w_integ / dp_total;
    } else {
        mean_thta = theta_acc[0];
        mean_mixr = mixr_acc[0];
    }
}

template <typename Lifter>
__device__ __noinline__ void run_ml_parcel(
    int idx, int n_levels, Lifter base_lifter,
    sharp::StridedPtr<const float> pres, sharp::StridedPtr<const float> hght,
    sharp::StridedPtr<const float> tmpk, sharp::StridedPtr<const float> dwpk,
    float* d_out_mlcape, float* d_out_mlcin)
{
    ThetaAccessor theta_acc{pres, tmpk};
    MixrAccessor mixr_acc{pres, dwpk};
    EnvVtmpkAccessor vtmpk_acc{pres, tmpk, dwpk};

    float mean_thta = 0.0f;
    float mean_mixr = 0.0f;
    compute_ml_mean(pres, theta_acc, mixr_acc, n_levels, mean_thta, mean_mixr);

    float ml_pcl_pres = pres[0];
    float ml_pcl_tmpk = sharp::theta(sharp::THETA_REF_PRESSURE, mean_thta, ml_pcl_pres);
    float ml_pcl_mixr = mean_mixr;
    Lifter lft = base_lifter;
    sharp::gpu::ConvectiveOutput ml_out = sharp::gpu::lift_and_integrate_column(
        lft, ml_pcl_pres, ml_pcl_tmpk, ml_pcl_mixr,
        pres, hght, tmpk, mixr_acc, vtmpk_acc, n_levels);
    if (d_out_mlcape) d_out_mlcape[idx] = ml_out.cape;
    if (d_out_mlcin) d_out_mlcin[idx] = ml_out.cin;
}

template <typename Lifter, bool FastMuThetae>
__device__ __noinline__ void run_mu_parcel(
    int idx, int n_levels, Lifter base_lifter,
    sharp::StridedPtr<const float> pres, sharp::StridedPtr<const float> hght,
    sharp::StridedPtr<const float> tmpk, sharp::StridedPtr<const float> dwpk,
    float* d_out_mucape, float* d_out_mucin)
{
    float p_limit = pres[0] - 30000.0f;
    EnvVtmpkAccessor vtmpk_acc{pres, tmpk, dwpk};
    MixrAccessor mixr_acc{pres, dwpk};

    if constexpr (FastMuThetae) {
        int k_max = 0;
        float max_thetae = -1e9f;
        for (int k = 0; k < n_levels; ++k) {
            if (pres[k] < p_limit) break;
            float th_e = sharp::thetae(pres[k], tmpk[k], dwpk[k]);
            if (th_e > max_thetae) {
                max_thetae = th_e;
                k_max = k;
            }
        }
        float mu_pres = pres[k_max];
        float mu_tmpk = tmpk[k_max];
        float mu_mixr = sharp::mixratio(mu_pres, dwpk[k_max]);
        Lifter lft = base_lifter;
        sharp::gpu::ConvectiveOutput mu_out = sharp::gpu::lift_and_integrate_column(
            lft, mu_pres, mu_tmpk, mu_mixr,
            pres, hght, tmpk, mixr_acc, vtmpk_acc, n_levels);
        if (d_out_mucape) d_out_mucape[idx] = mu_out.cape;
        if (d_out_mucin) d_out_mucin[idx] = mu_out.cin;
    } else {
        float max_cape = 0.0f;
        float best_cin = 0.0f;
        bool found = false;
        for (int k = 0; k < n_levels; ++k) {
            if (pres[k] < p_limit) break;
            float p_k = pres[k];
            float t_k = tmpk[k];
            float q_k = sharp::mixratio(p_k, dwpk[k]);
            Lifter lft = base_lifter;
            sharp::gpu::ConvectiveOutput out = sharp::gpu::lift_and_integrate_column(
                lft, p_k, t_k, q_k,
                pres, hght, tmpk, mixr_acc, vtmpk_acc, n_levels);
            if (!found || out.cape > max_cape) {
                max_cape = out.cape;
                best_cin = out.cin;
                found = true;
            }
        }
        if (d_out_mucape) d_out_mucape[idx] = max_cape;
        if (d_out_mucin) d_out_mucin[idx] = best_cin;
    }
}

__device__ __noinline__ void run_kinematics(
    int idx, int n_levels,
    sharp::StridedPtr<const float> pres, sharp::StridedPtr<const float> hght,
    sharp::StridedPtr<const float> u_wind, sharp::StridedPtr<const float> v_wind,
    float sb_cape, float sb_cin, float sb_lcl_agl,
    float* d_out_srh01, float* d_out_srh03, float* d_out_shr06, float* d_out_stp)
{
    sharp::HeightLayer mw_lyr_agl{0.0f, 6000.0f};
    sharp::HeightLayer h_layer_lo_agl{0.0f, 500.0f};
    sharp::HeightLayer h_layer_hi_agl{5500.0f, 6000.0f};

    sharp::PressureLayer mw_lyr = sharp::height_layer_to_pressure(mw_lyr_agl, pres, hght, n_levels, true);
    sharp::PressureLayer p_lo   = sharp::height_layer_to_pressure(h_layer_lo_agl, pres, hght, n_levels, true);
    sharp::PressureLayer p_hi   = sharp::height_layer_to_pressure(h_layer_hi_agl, pres, hght, n_levels, true);

    sharp::WindComponents layer_mean_wind = sharp::mean_wind(mw_lyr, pres, u_wind, v_wind, n_levels, false);
    sharp::WindComponents winds_lo = sharp::mean_wind(p_lo, pres, u_wind, v_wind, n_levels, false);
    sharp::WindComponents winds_hi = sharp::mean_wind(p_hi, pres, u_wind, v_wind, n_levels, false);

    float shear_u = winds_hi.u - winds_lo.u;
    float shear_v = winds_hi.v - winds_lo.v;
    float mag = sharp::vector_magnitude(shear_u, shear_v);

    sharp::WindComponents storm_motion{0.0f, 0.0f};
    constexpr float deviation = 7.5f;
    if (mag > 0.0f && !sharp::is_missing(layer_mean_wind.u) && !sharp::is_missing(layer_mean_wind.v)) {
        storm_motion.u = layer_mean_wind.u + ((deviation / mag) * shear_v);
        storm_motion.v = layer_mean_wind.v - ((deviation / mag) * shear_u);
    } else {
        storm_motion = layer_mean_wind;
    }

    sharp::HeightLayer lyr01{0.0f, 1000.0f};
    sharp::HeightLayer lyr03{0.0f, 3000.0f};
    float srh01 = sharp::helicity(lyr01, storm_motion, hght, u_wind, v_wind, n_levels);
    float srh03 = sharp::helicity(lyr03, storm_motion, hght, u_wind, v_wind, n_levels);
    if (d_out_srh01) d_out_srh01[idx] = srh01;
    if (d_out_srh03) d_out_srh03[idx] = srh03;

    sharp::HeightLayer lyr06{0.0f, 6000.0f};
    sharp::WindComponents shr06_cmp = sharp::wind_shear(lyr06, hght, u_wind, v_wind, n_levels);
    float shr06 = sharp::vector_magnitude(shr06_cmp.u, shr06_cmp.v);
    if (d_out_shr06) d_out_shr06[idx] = shr06;

    float stp = compute_stp(sb_cape, sb_cin, sb_lcl_agl, srh01, shr06);
    if (d_out_stp) d_out_stp[idx] = stp;
}

} // anonymous namespace

template <typename Lifter, bool FastMuThetae>
__global__ void convective_batch_kernel(
    const float* __restrict__ d_pres,
    const float* __restrict__ d_hght,
    const float* __restrict__ d_tmpk,
    const float* __restrict__ d_dwpk,
    const float* __restrict__ d_u,
    const float* __restrict__ d_v,
    float* __restrict__ d_out_sbcape,
    float* __restrict__ d_out_sbcin,
    float* __restrict__ d_out_mlcape,
    float* __restrict__ d_out_mlcin,
    float* __restrict__ d_out_mucape,
    float* __restrict__ d_out_mucin,
    float* __restrict__ d_out_srh01,
    float* __restrict__ d_out_srh03,
    float* __restrict__ d_out_shr06,
    float* __restrict__ d_out_stp,
    int n_levels,
    int n_profiles,
    Lifter base_lifter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_profiles) return;

    sharp::StridedPtr<const float> pres{d_pres + idx, n_profiles};
    sharp::StridedPtr<const float> hght{d_hght + idx, n_profiles};
    sharp::StridedPtr<const float> tmpk{d_tmpk + idx, n_profiles};
    sharp::StridedPtr<const float> dwpk{d_dwpk + idx, n_profiles};
    sharp::StridedPtr<const float> u_wind{d_u + idx, n_profiles};
    sharp::StridedPtr<const float> v_wind{d_v + idx, n_profiles};

    float sb_cape = 0.0f;
    float sb_cin = 0.0f;
    float sb_lcl_agl = 0.0f;

    run_sb_parcel<Lifter>(idx, n_levels, base_lifter, pres, hght, tmpk, dwpk,
                          d_out_sbcape, d_out_sbcin, sb_cape, sb_cin, sb_lcl_agl);

    run_ml_parcel<Lifter>(idx, n_levels, base_lifter, pres, hght, tmpk, dwpk,
                          d_out_mlcape, d_out_mlcin);

    run_mu_parcel<Lifter, FastMuThetae>(idx, n_levels, base_lifter, pres, hght, tmpk, dwpk,
                                        d_out_mucape, d_out_mucin);

    run_kinematics(idx, n_levels, pres, hght, u_wind, v_wind,
                   sb_cape, sb_cin, sb_lcl_agl,
                   d_out_srh01, d_out_srh03, d_out_shr06, d_out_stp);
}

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
    cudaStream_t stream)
{
    if (n_profiles <= 0 || n_levels <= 0) return;

    constexpr int blockSize = 256;
    const int numBlocks = (n_profiles + blockSize - 1) / blockSize;

    if (use_texture_lut) {
        lifter_texture_gpu lifter = get_global_texture_lut().get_device_lifter();
        if (fast_mu_thetae) {
            convective_batch_kernel<lifter_texture_gpu, true><<<numBlocks, blockSize, 0, stream>>>(
                d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
                d_out_sbcape, d_out_sbcin,
                d_out_mlcape, d_out_mlcin,
                d_out_mucape, d_out_mucin,
                d_out_srh01,  d_out_srh03,
                d_out_shr06,  d_out_stp,
                n_levels, n_profiles, lifter);
        } else {
            convective_batch_kernel<lifter_texture_gpu, false><<<numBlocks, blockSize, 0, stream>>>(
                d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
                d_out_sbcape, d_out_sbcin,
                d_out_mlcape, d_out_mlcin,
                d_out_mucape, d_out_mucin,
                d_out_srh01,  d_out_srh03,
                d_out_shr06,  d_out_stp,
                n_levels, n_profiles, lifter);
        }
    } else {
        sharp::lifter_wobus lifter{};
        if (fast_mu_thetae) {
            convective_batch_kernel<sharp::lifter_wobus, true><<<numBlocks, blockSize, 0, stream>>>(
                d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
                d_out_sbcape, d_out_sbcin,
                d_out_mlcape, d_out_mlcin,
                d_out_mucape, d_out_mucin,
                d_out_srh01,  d_out_srh03,
                d_out_shr06,  d_out_stp,
                n_levels, n_profiles, lifter);
        } else {
            convective_batch_kernel<sharp::lifter_wobus, false><<<numBlocks, blockSize, 0, stream>>>(
                d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
                d_out_sbcape, d_out_sbcin,
                d_out_mlcape, d_out_mlcin,
                d_out_mucape, d_out_mucin,
                d_out_srh01,  d_out_srh03,
                d_out_shr06,  d_out_stp,
                n_levels, n_profiles, lifter);
        }
    }
}

} // namespace sharp::gpu
