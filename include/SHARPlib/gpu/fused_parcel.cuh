/**
 * \file
 * \brief Register-fused parcel lifting and convective integration engine
 * \author Kelton Halbert, Aaron Anderson
 * \date 2026-10-13
 */

#ifndef SHARP_GPU_FUSED_PARCEL_CUH
#define SHARP_GPU_FUSED_PARCEL_CUH

#include <SHARPlib/constants.h>
#include <SHARPlib/interp.h>
#include <SHARPlib/layer.h>
#include <SHARPlib/parcel.h>
#include <SHARPlib/thermo.h>

#include <cmath>
#include <cstddef>

namespace sharp::gpu {

struct ConvectiveOutput {
    float cape = 0.0f;
    float cin = 0.0f;
    float lcl_hght_agl = 0.0f;
    float lfc_hght_agl = 0.0f;
    float el_hght_agl = 0.0f;
    float lcl_pres = MISSING;
    float lfc_pres = MISSING;
    float el_pres = MISSING;
    float cape_3km = 0.0f;
};

namespace detail {

template <typename Lifter, typename PtrPres, typename PtrVtmpk>
struct ParcelBuoyancyAccessor {
    Lifter& lifter;
    float lcl_pres;
    float lcl_tmpk;
    float thetav_lcl;
    std::ptrdiff_t k_dry_bot;
    std::ptrdiff_t k_dry_top;
    std::ptrdiff_t k_sat_bot;
    PtrPres pres;
    PtrVtmpk vtmpk;

    SHARP_HD inline float operator[](std::ptrdiff_t k) const {
        if (k < k_dry_bot) return MISSING;
        if (k <= k_dry_top) {
            float tv = sharp::theta(THETA_REF_PRESSURE, thetav_lcl, pres[k]);
            return sharp::buoyancy(tv, vtmpk[k]);
        }
        if (k >= k_sat_bot) {
            float tp = lifter(lcl_pres, lcl_tmpk, pres[k]);
            float tv = lifter.parcel_virtual_temperature(pres[k], tp);
            return sharp::buoyancy(tv, vtmpk[k]);
        }
        return MISSING;
    }
};

} // namespace detail

template <typename Lifter, typename PtrPres, typename PtrHght, typename PtrTmpk,
          typename PtrMixr, typename PtrVtmpk>
SHARP_HD inline ConvectiveOutput lift_and_integrate_column(
    Lifter& lifter,
    float pcl_pres, float pcl_tmpk, float pcl_mixr,
    const PtrPres pres, const PtrHght hght, const PtrTmpk tmpk,
    const PtrMixr mixr, const PtrVtmpk vtmpk,
    int N_z)
{
    ConvectiveOutput out{};

    // Dry ascent to LCL
    float lcl_pres = MISSING;
    float lcl_tmpk = MISSING;
    sharp::drylift(pcl_pres, pcl_tmpk, pcl_mixr, lcl_pres, lcl_tmpk);

    if (sharp::is_missing(lcl_pres) || (lcl_pres != lcl_pres) || lcl_pres <= pres[N_z - 1]) {
        return out;
    }

    out.lcl_pres = lcl_pres;
    float lcl_hght = sharp::interp_pressure(lcl_pres, pres, hght, N_z);
    out.lcl_hght_agl = lcl_hght - hght[0];

    float qv_lcl = sharp::mixratio(lcl_pres, lcl_tmpk);
    float tv_lcl = sharp::virtual_temperature(lcl_tmpk, qv_lcl);
    float thetav_lcl = sharp::theta(lcl_pres, tv_lcl, THETA_REF_PRESSURE);

    PressureLayer dry_lyr = {pcl_pres, lcl_pres};
    PressureLayer sat_lyr = {lcl_pres, pres[N_z - 1]};
    LayerIndex dry_idx = sharp::get_layer_index(dry_lyr, pres, N_z);
    LayerIndex sat_idx = sharp::get_layer_index(sat_lyr, pres, N_z);

    lifter.setup(lcl_pres, lcl_tmpk);

    detail::ParcelBuoyancyAccessor<Lifter, PtrPres, PtrVtmpk> buoy_accessor{
        lifter, lcl_pres, lcl_tmpk, thetav_lcl,
        dry_idx.kbot, dry_idx.ktop, sat_idx.kbot,
        pres, vtmpk
    };

    // Find LFC and EL (matching sharp::Parcel::find_lfc_el)
    float pos_buoy = 0.0f;
    float pos_buoy_3km = 0.0f;
    float pos_buoy_max = 0.0f;
    float pos_buoy_3km_final = 0.0f;
    float pbot = sat_lyr.bottom;
    float buoy_bot = sharp::interp_pressure(sat_lyr.bottom, pres, buoy_accessor, N_z);
    float hbot = lcl_hght;

    float lfc_pres = (buoy_bot > 0.0f) ? sat_lyr.bottom : MISSING;
    float eql_pres = MISSING;
    float lfc_pres_final = MISSING;
    float eql_pres_final = MISSING;
    bool in_pos_area = (buoy_bot > 0.0f);
    const float z_3km = hght[0] + 3000.0f;

    for (std::ptrdiff_t k = sat_idx.kbot; k < sat_idx.ktop + 1; ++k) {
#ifndef NO_QC
        if (sharp::is_missing(vtmpk[k])) continue;
#endif
        const float ptop = pres[k];
        const float htop = hght[k];
        const float buoy_top = buoy_accessor[k];
        const float lyr_top = (buoy_top + buoy_bot) / 2.0f;

        if (!in_pos_area && (buoy_bot <= 0.0f) && (buoy_top > 0.0f)) {
            pos_buoy = 0.0f;
            pos_buoy_3km = 0.0f;
            eql_pres = MISSING;
            in_pos_area = true;

            const float log_pbot = std::log10(pbot);
            const float log_ptop = std::log10(ptop);
            const float buoy_diff = buoy_top - buoy_bot;
            const float log_pres_diff = log_ptop - log_pbot;
            const float log_lfc_pres =
                log_pbot - buoy_bot * (log_pres_diff / buoy_diff);
            lfc_pres =
                std::min(std::pow(10.0f, log_lfc_pres), lcl_pres);
        }

        const float condition = (in_pos_area && (lyr_top > 0.0f)) ? 1.0f : 0.0f;
        const float dz = htop - hbot;
        pos_buoy += condition * lyr_top * dz;

        if (condition > 0.0f) {
            if (htop <= z_3km) {
                pos_buoy_3km += lyr_top * dz;
            } else if (hbot < z_3km) {
                float frac = (z_3km - hbot) / dz;
                float buoy_3k = buoy_bot + frac * (buoy_top - buoy_bot);
                pos_buoy_3km += 0.5f * (buoy_bot + buoy_3k) * (z_3km - hbot);
            }
        }

        if (in_pos_area && ((buoy_bot >= 0.0f) && (buoy_top < 0.0f))) {
            in_pos_area = false;
            const float log_pbot = std::log10(pbot);
            const float log_ptop = std::log10(ptop);
            const float buoy_diff = buoy_top - buoy_bot;
            const float log_pres_diff = log_ptop - log_pbot;
            const float log_eql_pres =
                log_pbot - buoy_bot * (log_pres_diff / buoy_diff);
            eql_pres = std::max(std::pow(10.0f, log_eql_pres), pres[N_z - 1]);
            if (pos_buoy > pos_buoy_max) {
                pos_buoy_max = pos_buoy;
                lfc_pres_final = lfc_pres;
                eql_pres_final = eql_pres;
                pos_buoy_3km_final = pos_buoy_3km;
            }
        }
        pbot = ptop;
        hbot = htop;
        buoy_bot = buoy_top;
    }

    if (in_pos_area) {
        eql_pres = MISSING;
        if (pos_buoy > pos_buoy_max) {
            pos_buoy_max = pos_buoy;
            lfc_pres_final = lfc_pres;
            eql_pres_final = eql_pres;
            pos_buoy_3km_final = pos_buoy_3km;
        }
    }

    if (pos_buoy_max > 0.0f) {
        out.lfc_pres = lfc_pres_final;
        out.el_pres = eql_pres_final;
        out.cape_3km = pos_buoy_3km_final;
    }

    if (out.lfc_pres != MISSING) {
        PressureLayer lfc_el = {MISSING, MISSING};
        if (out.el_pres == MISSING) {
            lfc_el = {out.lfc_pres, pres[N_z - 1]};
        } else {
            lfc_el = {out.lfc_pres, out.el_pres};
        }
        PressureLayer lpl_lfc = {pcl_pres, out.lfc_pres};

        HeightLayer lfc_el_ht =
            sharp::pressure_layer_to_height(lfc_el, pres, hght, N_z);
        HeightLayer lpl_lfc_ht =
            sharp::pressure_layer_to_height(lpl_lfc, pres, hght, N_z);

        out.cin = sharp::integrate_layer_trapz(lpl_lfc_ht, buoy_accessor, hght, N_z, -1);
        out.cape = sharp::integrate_layer_trapz(lfc_el_ht, buoy_accessor, hght, N_z, 1);

        out.lfc_hght_agl = lfc_el_ht.bottom - hght[0];
        out.el_hght_agl = lfc_el_ht.top - hght[0];

        if (pos_buoy_max > 0.0f && out.cape > 0.0f) {
            out.cape_3km = out.cape * (pos_buoy_3km_final / pos_buoy_max);
        }
    }

    return out;
}

// Overload with alternate parameter ordering (from implementation plan)
template <typename Lifter, typename PtrPres, typename PtrHght, typename PtrTmpk,
          typename PtrMixr, typename PtrVtmpk>
SHARP_HD inline ConvectiveOutput lift_and_integrate_column(
    Lifter& lifter,
    const PtrPres pres, const PtrHght hght, const PtrTmpk tmpk,
    const PtrMixr mixr, const PtrVtmpk vtmpk,
    int N_z,
    float pcl_pres, float pcl_tmpk, float pcl_mixr)
{
    return lift_and_integrate_column(lifter, pcl_pres, pcl_tmpk, pcl_mixr,
                                     pres, hght, tmpk, mixr, vtmpk, N_z);
}

// Const lifter overloads (for lifters with const operator() such as lifter_wobus)
template <typename Lifter, typename PtrPres, typename PtrHght, typename PtrTmpk,
          typename PtrMixr, typename PtrVtmpk>
SHARP_HD inline ConvectiveOutput lift_and_integrate_column(
    const Lifter& lifter,
    float pcl_pres, float pcl_tmpk, float pcl_mixr,
    const PtrPres pres, const PtrHght hght, const PtrTmpk tmpk,
    const PtrMixr mixr, const PtrVtmpk vtmpk,
    int N_z)
{
    Lifter lifter_copy = lifter;
    return lift_and_integrate_column(lifter_copy, pcl_pres, pcl_tmpk, pcl_mixr,
                                     pres, hght, tmpk, mixr, vtmpk, N_z);
}

template <typename Lifter, typename PtrPres, typename PtrHght, typename PtrTmpk,
          typename PtrMixr, typename PtrVtmpk>
SHARP_HD inline ConvectiveOutput lift_and_integrate_column(
    const Lifter& lifter,
    const PtrPres pres, const PtrHght hght, const PtrTmpk tmpk,
    const PtrMixr mixr, const PtrVtmpk vtmpk,
    int N_z,
    float pcl_pres, float pcl_tmpk, float pcl_mixr)
{
    Lifter lifter_copy = lifter;
    return lift_and_integrate_column(lifter_copy, pcl_pres, pcl_tmpk, pcl_mixr,
                                     pres, hght, tmpk, mixr, vtmpk, N_z);
}

} // namespace sharp::gpu

#endif // SHARP_GPU_FUSED_PARCEL_CUH
