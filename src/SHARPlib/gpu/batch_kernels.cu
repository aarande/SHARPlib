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

SHARP_HD SHARP_FORCEINLINE float compute_scp(
    float mu_cape, float eff_srh, float eff_shear) noexcept
{
    if ((mu_cape != mu_cape) || sharp::is_missing(mu_cape) ||
        (eff_srh != eff_srh) || sharp::is_missing(eff_srh) ||
        (eff_shear != eff_shear) || sharp::is_missing(eff_shear)) {
        return 0.0f;
    }

    if (eff_shear > 20.0f) {
        eff_shear = 20.0f;
    } else if (eff_shear < 10.0f) {
        eff_shear = 0.0f;
    }

    const float mu_cape_term = mu_cape / 1000.0f;
    const float eff_srh_term = eff_srh / 50.0f;
    const float eff_shear_term = eff_shear / 20.0f;

    float scp = mu_cape_term * eff_srh_term * eff_shear_term;
    if (scp < 0.0f || (scp != scp)) scp = 0.0f;
    return scp;
}

template <typename Lifter>
struct WetbulbAccessor {
    Lifter lifter;
    sharp::StridedPtr<const float> pres;
    sharp::StridedPtr<const float> tmpk;
    sharp::StridedPtr<const float> dwpk;
    SHARP_HD inline float operator[](std::ptrdiff_t k) const {
        Lifter lft = lifter;
        return sharp::wetbulb(lft, pres[k], tmpk[k], dwpk[k]);
    }
};

template <typename Lifter>
__device__ __noinline__ sharp::gpu::ConvectiveOutput run_sb_parcel(
    int n_levels, Lifter base_lifter,
    sharp::StridedPtr<const float> pres, sharp::StridedPtr<const float> hght,
    sharp::StridedPtr<const float> tmpk, sharp::StridedPtr<const float> dwpk)
{
    float pcl_pres = pres[0];
    float pcl_tmpk = tmpk[0];
    float pcl_mixr = sharp::mixratio(pcl_pres, dwpk[0]);
    EnvVtmpkAccessor vtmpk_acc{pres, tmpk, dwpk};
    MixrAccessor mixr_acc{pres, dwpk};
    Lifter lft = base_lifter;
    return sharp::gpu::lift_and_integrate_column(
        lft, pcl_pres, pcl_tmpk, pcl_mixr,
        pres, hght, tmpk, mixr_acc, vtmpk_acc, n_levels);
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
__device__ __noinline__ sharp::gpu::ConvectiveOutput run_ml_parcel(
    int n_levels, Lifter base_lifter,
    sharp::StridedPtr<const float> pres, sharp::StridedPtr<const float> hght,
    sharp::StridedPtr<const float> tmpk, sharp::StridedPtr<const float> dwpk)
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
    return sharp::gpu::lift_and_integrate_column(
        lft, ml_pcl_pres, ml_pcl_tmpk, ml_pcl_mixr,
        pres, hght, tmpk, mixr_acc, vtmpk_acc, n_levels);
}

template <typename Lifter, bool FastMuThetae>
__device__ __noinline__ sharp::gpu::ConvectiveOutput run_mu_parcel(
    int n_levels, Lifter base_lifter,
    sharp::StridedPtr<const float> pres, sharp::StridedPtr<const float> hght,
    sharp::StridedPtr<const float> tmpk, sharp::StridedPtr<const float> dwpk)
{
    float p_limit = pres[0] - 30000.0f;
    EnvVtmpkAccessor vtmpk_acc{pres, tmpk, dwpk};
    MixrAccessor mixr_acc{pres, dwpk};
    sharp::gpu::ConvectiveOutput mu_out{};

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
        mu_out = sharp::gpu::lift_and_integrate_column(
            lft, mu_pres, mu_tmpk, mu_mixr,
            pres, hght, tmpk, mixr_acc, vtmpk_acc, n_levels);
    } else {
        bool found = false;
        for (int k = 0; k < n_levels; ++k) {
            if (pres[k] < p_limit) break;
            float p_k = pres[k];
            float t_k = tmpk[k];
            float q_k = sharp::mixratio(p_k, dwpk[k]);
            Lifter lft = base_lifter;
            sharp::gpu::ConvectiveOutput cur = sharp::gpu::lift_and_integrate_column(
                lft, p_k, t_k, q_k,
                pres, hght, tmpk, mixr_acc, vtmpk_acc, n_levels);
            if (!found || cur.cape > mu_out.cape) {
                mu_out = cur;
                found = true;
            }
        }
    }
    return mu_out;
}

struct ThermoOutputs {
    float pwat = sharp::MISSING;
    float lr03 = sharp::MISSING;
    float lr36 = sharp::MISSING;
    float lr75 = sharp::MISSING;
    float fz_level_agl = sharp::MISSING;
    float wbz_level_agl = sharp::MISSING;
};

template <typename Lifter>
__device__ __noinline__ ThermoOutputs run_thermo(
    int n_levels, Lifter base_lifter,
    sharp::StridedPtr<const float> pres, sharp::StridedPtr<const float> hght,
    sharp::StridedPtr<const float> tmpk, sharp::StridedPtr<const float> dwpk,
    bool do_pwat, bool do_lr03, bool do_lr36, bool do_lr75, bool do_fz, bool do_wbz)
{
    ThermoOutputs tout{};
    if (do_pwat) {
        float pwat_sum = 0.0f;
        float w_prev = sharp::mixratio(pres[0], dwpk[0]);
        float p_prev = pres[0];
        for (int k = 1; k < n_levels; ++k) {
            float p_curr = pres[k];
            float w_curr = sharp::mixratio(p_curr, dwpk[k]);
            pwat_sum += 0.5f * (w_prev + w_curr) * (p_prev - p_curr);
            w_prev = w_curr;
            p_prev = p_curr;
        }
        tout.pwat = (1000.0f / (sharp::RHO_LWAT * sharp::GRAVITY)) * pwat_sum;
    }

    if (do_lr03) {
        float t_3k = sharp::interp_height(hght[0] + 3000.0f, hght, tmpk, n_levels);
        if (!sharp::is_missing(t_3k) && !sharp::is_missing(tmpk[0])) {
            tout.lr03 = -(t_3k - tmpk[0]) / 3.0f;
        }
    }

    if (do_lr36) {
        float t_3k = sharp::interp_height(hght[0] + 3000.0f, hght, tmpk, n_levels);
        float t_6k = sharp::interp_height(hght[0] + 6000.0f, hght, tmpk, n_levels);
        if (!sharp::is_missing(t_3k) && !sharp::is_missing(t_6k)) {
            tout.lr36 = -(t_6k - t_3k) / 3.0f;
        }
    }

    if (do_lr75) {
        float t_700 = sharp::interp_pressure(70000.0f, pres, tmpk, n_levels);
        float t_500 = sharp::interp_pressure(50000.0f, pres, tmpk, n_levels);
        float z_700 = sharp::interp_pressure(70000.0f, pres, hght, n_levels);
        float z_500 = sharp::interp_pressure(50000.0f, pres, hght, n_levels);
        if (!sharp::is_missing(t_700) && !sharp::is_missing(t_500) &&
            !sharp::is_missing(z_700) && !sharp::is_missing(z_500) &&
            std::abs(z_500 - z_700) > 1.0f) {
            tout.lr75 = -((t_500 - t_700) / (z_500 - z_700)) * 1000.0f;
        }
    }

    if (do_fz) {
        float h_fz = sharp::find_first_height(sharp::ZEROCNK, hght, tmpk, n_levels);
        if (!sharp::is_missing(h_fz)) {
            tout.fz_level_agl = h_fz - hght[0];
        }
    }

    if (do_wbz) {
        WetbulbAccessor<Lifter> wb_acc{base_lifter, pres, tmpk, dwpk};
        float h_wbz = sharp::find_first_height(sharp::ZEROCNK, hght, wb_acc, n_levels);
        if (!sharp::is_missing(h_wbz)) {
            tout.wbz_level_agl = h_wbz - hght[0];
        }
    }
    return tout;
}

struct KinematicsOutputs {
    float bunkers_rm_u = sharp::MISSING;
    float bunkers_rm_v = sharp::MISSING;
    float bunkers_lm_u = sharp::MISSING;
    float bunkers_lm_v = sharp::MISSING;
    float mean_wind_u  = sharp::MISSING;
    float mean_wind_v  = sharp::MISSING;
    float shr01 = sharp::MISSING;
    float shr03 = sharp::MISSING;
    float shr06 = sharp::MISSING;
    float srh01 = sharp::MISSING;
    float srh03 = sharp::MISSING;
    float srh05 = sharp::MISSING;
    float crit_angle = sharp::MISSING;
    float stp = 0.0f;
    float scp = 0.0f;
    float ehi01 = 0.0f;
    float ehi03 = 0.0f;
};

__device__ __noinline__ KinematicsOutputs run_kinematics(
    int n_levels,
    sharp::StridedPtr<const float> pres, sharp::StridedPtr<const float> hght,
    sharp::StridedPtr<const float> u_wind, sharp::StridedPtr<const float> v_wind,
    float sb_cape, float sb_cin, float sb_lcl_agl, float mu_cape,
    bool do_srh05, bool do_srh01, bool do_srh03,
    bool do_shr01, bool do_shr03, bool do_shr06,
    bool do_crit_angle, bool do_stp, bool do_scp, bool do_ehi01, bool do_ehi03)
{
    KinematicsOutputs kout{};

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
    sharp::WindComponents storm_motion_lm{0.0f, 0.0f};
    constexpr float deviation = 7.5f;
    if (mag > 0.0f && !sharp::is_missing(layer_mean_wind.u) && !sharp::is_missing(layer_mean_wind.v)) {
        storm_motion.u = layer_mean_wind.u + ((deviation / mag) * shear_v);
        storm_motion.v = layer_mean_wind.v - ((deviation / mag) * shear_u);
        storm_motion_lm.u = layer_mean_wind.u - ((deviation / mag) * shear_v);
        storm_motion_lm.v = layer_mean_wind.v + ((deviation / mag) * shear_u);
    } else {
        storm_motion = layer_mean_wind;
        storm_motion_lm = layer_mean_wind;
    }

    kout.bunkers_rm_u = storm_motion.u;
    kout.bunkers_rm_v = storm_motion.v;
    kout.bunkers_lm_u = storm_motion_lm.u;
    kout.bunkers_lm_v = storm_motion_lm.v;
    kout.mean_wind_u  = layer_mean_wind.u;
    kout.mean_wind_v  = layer_mean_wind.v;

    sharp::HeightLayer lyr005{0.0f, 500.0f};
    sharp::HeightLayer lyr01{0.0f, 1000.0f};
    sharp::HeightLayer lyr03{0.0f, 3000.0f};
    sharp::HeightLayer lyr06{0.0f, 6000.0f};

    float srh05 = 0.0f;
    float srh01 = 0.0f;
    float srh03 = 0.0f;
    if (do_srh05) {
        srh05 = sharp::helicity(lyr005, storm_motion, hght, u_wind, v_wind, n_levels);
        kout.srh05 = srh05;
    }
    if (do_srh01) {
        srh01 = sharp::helicity(lyr01, storm_motion, hght, u_wind, v_wind, n_levels);
        kout.srh01 = srh01;
    }
    if (do_srh03) {
        srh03 = sharp::helicity(lyr03, storm_motion, hght, u_wind, v_wind, n_levels);
        kout.srh03 = srh03;
    }

    if (do_shr01) {
        sharp::WindComponents shr01_cmp = sharp::wind_shear(lyr01, hght, u_wind, v_wind, n_levels);
        kout.shr01 = sharp::vector_magnitude(shr01_cmp.u, shr01_cmp.v);
    }
    if (do_shr03) {
        sharp::WindComponents shr03_cmp = sharp::wind_shear(lyr03, hght, u_wind, v_wind, n_levels);
        kout.shr03 = sharp::vector_magnitude(shr03_cmp.u, shr03_cmp.v);
    }

    float shr06 = 0.0f;
    if (do_shr06) {
        sharp::WindComponents shr06_cmp = sharp::wind_shear(lyr06, hght, u_wind, v_wind, n_levels);
        shr06 = sharp::vector_magnitude(shr06_cmp.u, shr06_cmp.v);
        kout.shr06 = shr06;
    }

    if (do_crit_angle) {
        float u_sfc = u_wind[0];
        float v_sfc = v_wind[0];
        float u_500 = sharp::interp_height(hght[0] + 500.0f, hght, u_wind, n_levels);
        float v_500 = sharp::interp_height(hght[0] + 500.0f, hght, v_wind, n_levels);

        float shear_05_u = u_500 - u_sfc;
        float shear_05_v = v_500 - v_sfc;
        float sr_u = u_sfc - storm_motion.u;
        float sr_v = v_sfc - storm_motion.v;

        float mag_shear = sharp::vector_magnitude(shear_05_u, shear_05_v);
        float mag_sr    = sharp::vector_magnitude(sr_u, sr_v);

        float crit_ang = sharp::MISSING;
        if (mag_shear > 1e-4f && mag_sr > 1e-4f) {
            float dot = shear_05_u * sr_u + shear_05_v * sr_v;
            float cos_angle = dot / (mag_shear * mag_sr);
            cos_angle = fmaxf(-1.0f, fminf(1.0f, cos_angle));
            crit_ang = acosf(cos_angle) * (180.0f / sharp::PI);
        }
        kout.crit_angle = crit_ang;
    }

    if (do_stp) {
        kout.stp = compute_stp(sb_cape, sb_cin, sb_lcl_agl, srh01, shr06);
    }
    if (do_scp) {
        kout.scp = compute_scp(mu_cape, srh03, shr06);
    }
    if (do_ehi01) {
        if (sb_cape != sb_cape || sharp::is_missing(sb_cape) || srh01 != srh01 || sharp::is_missing(srh01)) {
            kout.ehi01 = 0.0f;
        } else {
            kout.ehi01 = (sb_cape * srh01) / 160000.0f;
        }
    }
    if (do_ehi03) {
        if (sb_cape != sb_cape || sharp::is_missing(sb_cape) || srh03 != srh03 || sharp::is_missing(srh03)) {
            kout.ehi03 = 0.0f;
        } else {
            kout.ehi03 = (sb_cape * srh03) / 160000.0f;
        }
    }

    return kout;
}

} // anonymous namespace

template <typename Lifter, bool FastMuThetae>
__global__ void convective_batch_flexible_kernel(
    const float* __restrict__ d_pres,
    const float* __restrict__ d_hght,
    const float* __restrict__ d_tmpk,
    const float* __restrict__ d_dwpk,
    const float* __restrict__ d_u,
    const float* __restrict__ d_v,
    ConvectiveBatchOutputs out,
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

    bool need_sb = (out.sbcape || out.sbcin || out.sblcl_hght_agl || out.sblcl_pres ||
                    out.sblfc_hght_agl || out.sblfc_pres || out.sbel_hght_agl || out.sbel_pres ||
                    out.sb3cape || out.stp || out.ehi01 || out.ehi03);
    bool need_ml = (out.mlcape || out.mlcin || out.mllcl_hght_agl || out.mllcl_pres ||
                    out.mllfc_hght_agl || out.mllfc_pres || out.mlel_hght_agl || out.mlel_pres ||
                    out.ml3cape);
    bool need_mu = (out.mucape || out.mucin || out.mulcl_hght_agl || out.mulcl_pres ||
                    out.mulfc_hght_agl || out.mulfc_pres || out.muel_hght_agl || out.muel_pres ||
                    out.mu3cape || out.scp);
    bool need_kin = (out.bunkers_rm_u || out.bunkers_rm_v || out.bunkers_lm_u || out.bunkers_lm_v ||
                     out.mean_wind_u || out.mean_wind_v || out.shr01 || out.shr03 || out.shr06 ||
                     out.srh01 || out.srh03 || out.srh05 || out.crit_angle || out.stp || out.scp ||
                     out.ehi01 || out.ehi03);
    bool need_thermo = (out.pwat || out.lr03 || out.lr36 || out.lr75 || out.fz_level_agl ||
                        out.wbz_level_agl);

    float sb_cape = 0.0f;
    float sb_cin = 0.0f;
    float sb_lcl_agl = 0.0f;

    if (need_sb) {
        sharp::gpu::ConvectiveOutput sb_out = run_sb_parcel<Lifter>(
            n_levels, base_lifter, pres, hght, tmpk, dwpk);
        sb_cape = sb_out.cape;
        sb_cin = sb_out.cin;
        sb_lcl_agl = sb_out.lcl_hght_agl;

        if (out.sbcape) out.sbcape[idx] = sb_out.cape;
        if (out.sbcin) out.sbcin[idx] = sb_out.cin;
        if (out.sblcl_hght_agl) out.sblcl_hght_agl[idx] = sb_out.lcl_hght_agl;
        if (out.sblcl_pres) out.sblcl_pres[idx] = sb_out.lcl_pres;
        if (out.sblfc_hght_agl) out.sblfc_hght_agl[idx] = sb_out.lfc_hght_agl;
        if (out.sblfc_pres) out.sblfc_pres[idx] = sb_out.lfc_pres;
        if (out.sbel_hght_agl) out.sbel_hght_agl[idx] = sb_out.el_hght_agl;
        if (out.sbel_pres) out.sbel_pres[idx] = sb_out.el_pres;
        if (out.sb3cape) out.sb3cape[idx] = sb_out.cape_3km;
    }

    if (need_ml) {
        sharp::gpu::ConvectiveOutput ml_out = run_ml_parcel<Lifter>(
            n_levels, base_lifter, pres, hght, tmpk, dwpk);
        if (out.mlcape) out.mlcape[idx] = ml_out.cape;
        if (out.mlcin) out.mlcin[idx] = ml_out.cin;
        if (out.mllcl_hght_agl) out.mllcl_hght_agl[idx] = ml_out.lcl_hght_agl;
        if (out.mllcl_pres) out.mllcl_pres[idx] = ml_out.lcl_pres;
        if (out.mllfc_hght_agl) out.mllfc_hght_agl[idx] = ml_out.lfc_hght_agl;
        if (out.mllfc_pres) out.mllfc_pres[idx] = ml_out.lfc_pres;
        if (out.mlel_hght_agl) out.mlel_hght_agl[idx] = ml_out.el_hght_agl;
        if (out.mlel_pres) out.mlel_pres[idx] = ml_out.el_pres;
        if (out.ml3cape) out.ml3cape[idx] = ml_out.cape_3km;
    }

    float mu_cape = 0.0f;
    if (need_mu) {
        sharp::gpu::ConvectiveOutput mu_out = run_mu_parcel<Lifter, FastMuThetae>(
            n_levels, base_lifter, pres, hght, tmpk, dwpk);
        mu_cape = mu_out.cape;
        if (out.mucape) out.mucape[idx] = mu_out.cape;
        if (out.mucin) out.mucin[idx] = mu_out.cin;
        if (out.mulcl_hght_agl) out.mulcl_hght_agl[idx] = mu_out.lcl_hght_agl;
        if (out.mulcl_pres) out.mulcl_pres[idx] = mu_out.lcl_pres;
        if (out.mulfc_hght_agl) out.mulfc_hght_agl[idx] = mu_out.lfc_hght_agl;
        if (out.mulfc_pres) out.mulfc_pres[idx] = mu_out.lfc_pres;
        if (out.muel_hght_agl) out.muel_hght_agl[idx] = mu_out.el_hght_agl;
        if (out.muel_pres) out.muel_pres[idx] = mu_out.el_pres;
        if (out.mu3cape) out.mu3cape[idx] = mu_out.cape_3km;
    }

    if (need_kin) {
        bool do_srh05 = (out.srh05 != nullptr);
        bool do_srh01 = (out.srh01 || out.stp || out.ehi01);
        bool do_srh03 = (out.srh03 || out.scp || out.ehi03);
        bool do_shr01 = (out.shr01 != nullptr);
        bool do_shr03 = (out.shr03 != nullptr);
        bool do_shr06 = (out.shr06 || out.stp || out.scp);
        bool do_crit_angle = (out.crit_angle != nullptr);
        bool do_stp = (out.stp != nullptr);
        bool do_scp = (out.scp != nullptr);
        bool do_ehi01 = (out.ehi01 != nullptr);
        bool do_ehi03 = (out.ehi03 != nullptr);

        KinematicsOutputs kout = run_kinematics(
            n_levels, pres, hght, u_wind, v_wind,
            sb_cape, sb_cin, sb_lcl_agl, mu_cape,
            do_srh05, do_srh01, do_srh03,
            do_shr01, do_shr03, do_shr06,
            do_crit_angle, do_stp, do_scp, do_ehi01, do_ehi03);

        if (out.bunkers_rm_u) out.bunkers_rm_u[idx] = kout.bunkers_rm_u;
        if (out.bunkers_rm_v) out.bunkers_rm_v[idx] = kout.bunkers_rm_v;
        if (out.bunkers_lm_u) out.bunkers_lm_u[idx] = kout.bunkers_lm_u;
        if (out.bunkers_lm_v) out.bunkers_lm_v[idx] = kout.bunkers_lm_v;
        if (out.mean_wind_u)  out.mean_wind_u[idx]  = kout.mean_wind_u;
        if (out.mean_wind_v)  out.mean_wind_v[idx]  = kout.mean_wind_v;
        if (out.shr01) out.shr01[idx] = kout.shr01;
        if (out.shr03) out.shr03[idx] = kout.shr03;
        if (out.shr06) out.shr06[idx] = kout.shr06;
        if (out.srh01) out.srh01[idx] = kout.srh01;
        if (out.srh03) out.srh03[idx] = kout.srh03;
        if (out.srh05) out.srh05[idx] = kout.srh05;
        if (out.crit_angle) out.crit_angle[idx] = kout.crit_angle;
        if (out.stp) out.stp[idx] = kout.stp;
        if (out.scp) out.scp[idx] = kout.scp;
        if (out.ehi01) out.ehi01[idx] = kout.ehi01;
        if (out.ehi03) out.ehi03[idx] = kout.ehi03;
    }

    if (need_thermo) {
        bool do_pwat = (out.pwat != nullptr);
        bool do_lr03 = (out.lr03 != nullptr);
        bool do_lr36 = (out.lr36 != nullptr);
        bool do_lr75 = (out.lr75 != nullptr);
        bool do_fz   = (out.fz_level_agl != nullptr);
        bool do_wbz  = (out.wbz_level_agl != nullptr);

        ThermoOutputs tout = run_thermo<Lifter>(
            n_levels, base_lifter, pres, hght, tmpk, dwpk,
            do_pwat, do_lr03, do_lr36, do_lr75, do_fz, do_wbz);

        if (out.pwat) out.pwat[idx] = tout.pwat;
        if (out.lr03) out.lr03[idx] = tout.lr03;
        if (out.lr36) out.lr36[idx] = tout.lr36;
        if (out.lr75) out.lr75[idx] = tout.lr75;
        if (out.fz_level_agl) out.fz_level_agl[idx] = tout.fz_level_agl;
        if (out.wbz_level_agl) out.wbz_level_agl[idx] = tout.wbz_level_agl;
    }
}

void launch_convective_batch_flexible(
    const float* d_pres, const float* d_hght, const float* d_tmpk, const float* d_dwpk,
    const float* d_u, const float* d_v,
    const ConvectiveBatchOutputs& outputs,
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
            convective_batch_flexible_kernel<lifter_texture_gpu, true><<<numBlocks, blockSize, 0, stream>>>(
                d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
                outputs,
                n_levels, n_profiles, lifter);
        } else {
            convective_batch_flexible_kernel<lifter_texture_gpu, false><<<numBlocks, blockSize, 0, stream>>>(
                d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
                outputs,
                n_levels, n_profiles, lifter);
        }
    } else {
        sharp::lifter_wobus lifter{};
        if (fast_mu_thetae) {
            convective_batch_flexible_kernel<sharp::lifter_wobus, true><<<numBlocks, blockSize, 0, stream>>>(
                d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
                outputs,
                n_levels, n_profiles, lifter);
        } else {
            convective_batch_flexible_kernel<sharp::lifter_wobus, false><<<numBlocks, blockSize, 0, stream>>>(
                d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
                outputs,
                n_levels, n_profiles, lifter);
        }
    }
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
    ConvectiveBatchOutputs outputs;
    outputs.sbcape = d_out_sbcape;
    outputs.sbcin  = d_out_sbcin;
    outputs.mlcape = d_out_mlcape;
    outputs.mlcin  = d_out_mlcin;
    outputs.mucape = d_out_mucape;
    outputs.mucin  = d_out_mucin;
    outputs.srh01  = d_out_srh01;
    outputs.srh03  = d_out_srh03;
    outputs.shr06  = d_out_shr06;
    outputs.stp    = d_out_stp;

    launch_convective_batch_flexible(
        d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
        outputs,
        n_levels, n_profiles,
        use_texture_lut,
        fast_mu_thetae,
        stream);
}

} // namespace sharp::gpu
