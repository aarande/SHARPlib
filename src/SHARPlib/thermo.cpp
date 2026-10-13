/**
 * \file
 * \brief Thermodynamic routines that do <!--
 * -->not directly involve parcel based ascent
 * \author
 *   Kelton Halbert                  \n
 *   Email: kelton.halbert@noaa.gov  \n
 * \date   2022-10-13
 *
 * Written for the NWS Storm Predidiction Center \n
 * Based on NSHARP routines originally written by
 * John Hart and Rich Thompson at SPC.
 */
#include <SHARPlib/constants.h>
#include <SHARPlib/interp.h>
#include <SHARPlib/layer.h>
#include <SHARPlib/parcel.h>
#include <SHARPlib/thermo.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>

namespace sharp {

struct lifter_wobus;
struct lifter_cm1;



float _solve_cm1(float& pcl_pres_next, float& pcl_pi_next, float& pcl_t_next,
                 float& pcl_rv_next, float& pcl_rl_next, float& pcl_ri_next,
                 float pcl_pres_prev, float pcl_t_prev, float pcl_theta_prev,
                 float pcl_rv_prev, float pcl_rl_prev, float pcl_ri_prev,
                 float rv_total, bool ascending, bool ice = false,
                 float converge = 0.0002f, const int max_iters = 100) {
    // first guess - use the theta of the parcel
    // before lifting, and update the first guess
    // accordingly
    float pcl_theta_last = pcl_theta_prev;
    float pcl_theta_next = pcl_theta_prev;
    bool not_converged = true;
    int iter_count = 0;

    while ((not_converged) && (iter_count < max_iters)) {
        pcl_t_next = pcl_theta_last * pcl_pi_next;

        float fliq = 1.0;
        float fice = 0.0;
        if (ice) {
            fliq = std::max(
                std::min((pcl_t_next - 233.15f) / (ZEROCNK - 233.15f), 1.0f),
                0.0f);
            fice = 1.0f - fliq;
        }

        const float rv_term = fliq * mixratio(pcl_pres_next, pcl_t_next) +
                              fice * mixratio_ice(pcl_pres_next, pcl_t_next);
        if (ascending) {
            pcl_rv_next = std::min(rv_total, rv_term);
            pcl_ri_next = std::max(fice * (rv_total - pcl_rv_next), 0.0f);
            pcl_rl_next = std::max(rv_total - pcl_rv_next - pcl_ri_next, 0.0f);
        } else {
            pcl_rv_next = rv_term;
            pcl_rl_next = fliq * (rv_total - pcl_rv_next);
            pcl_ri_next = fice * (rv_total - pcl_rv_next);
        }

        const float tbar = 0.5f * (pcl_t_prev + pcl_t_next);
        const float rvbar = 0.5f * (pcl_rv_prev + pcl_rv_next);
        const float rlbar = 0.5f * (pcl_rl_prev + pcl_rl_next);
        const float ribar = 0.5f * (pcl_ri_prev + pcl_ri_next);

        const float LHV = LV1 - LV2 * tbar;
        const float LHS = LS1 - LS2 * tbar;

        const float RM = RDGAS + RVGAS * rvbar;
        const float CPM =
            CP_DRYAIR + CP_VAPOR * rvbar + CP_LIQUID * rlbar + CP_ICE * ribar;
        const float term =
            LHV * (pcl_rl_next - pcl_rl_prev) / (CPM * tbar) +
            LHS * (pcl_ri_next - pcl_ri_prev) / (CPM * tbar) +
            (RM / CPM - ROCP) * std::log(pcl_pres_next / pcl_pres_prev);

        pcl_theta_next = pcl_theta_prev * std::exp(term);

        float relax = (iter_count > 20) ? 0.05f : 0.3f;
        if (std::abs(pcl_theta_next - pcl_theta_last) > converge) {
            pcl_theta_last =
                pcl_theta_last + relax * (pcl_theta_next - pcl_theta_last);
        } else {
            not_converged = false;
        }
        iter_count += 1;
    }
    if (iter_count >= max_iters) {
        printf(
            "WARNING: iterative solver failed to converge. "
            "Returning best guess value.");
    }
    return pcl_theta_next;
}

float moist_adiabat_cm1(float pressure, float temperature, float new_pressure,
                        float& rv_total, float& rv, float& rl, float& ri,
                        float pres_incr, float converge, adiabat ma_type) {
#ifndef NO_QC
    if ((temperature == MISSING) || (pressure == MISSING) ||
        (new_pressure == MISSING)) {
        return MISSING;
    }
    if (std::isnan(temperature) || std::isnan(pressure) ||
        std::isnan(new_pressure)) {
        return std::nanf("");
    }
#endif

    // set up solver variables
    const bool ice = (ma_type >= adiabat::pseudo_ice) ? true : false;
    float dp = new_pressure - pressure;
    const bool ascending = std::signbit(dp);
    const int n_iters =
        (std::abs(dp) < pres_incr) ? 1 : 1 + (int)(std::abs(dp) / pres_incr);
    dp = (n_iters == 1) ? dp : dp / (float)n_iters;

    // Start by setting the "top" variables.
    float pcl_theta_next = theta(pressure, temperature);
    float pcl_pres_next = pressure;
    float pcl_pi_next = std::pow(pressure / THETA_REF_PRESSURE, ROCP);
    float pcl_t_next = pcl_theta_next * pcl_pi_next;
    float pcl_rv_next = rv;
    float pcl_rl_next = rl;
    float pcl_ri_next = ri;

    // Iterate the required number of times to reach the new pressure
    // level from the old one in increments of dp
    for (int i = 0; i < n_iters; ++i) {
        float pcl_theta_prev = pcl_theta_next;
        float pcl_pres_prev = pcl_pres_next;
        float pcl_rv_prev = pcl_rv_next;
        float pcl_rl_prev = pcl_rl_next;
        float pcl_ri_prev = pcl_ri_next;
        float pcl_t_prev = pcl_t_next;

        pcl_pres_next += dp;
        pcl_pi_next = std::pow(pcl_pres_next / THETA_REF_PRESSURE, ROCP);

        // call the iterative solver to get the new parcel
        // theta accounting for liquid (and ice if enabled)
        pcl_theta_next = _solve_cm1(
            pcl_pres_next, pcl_pi_next, pcl_t_next, pcl_rv_next, pcl_rl_next,
            pcl_ri_next, pcl_pres_prev, pcl_t_prev, pcl_theta_prev, pcl_rv_prev,
            pcl_rl_prev, pcl_ri_prev, rv_total, ascending, ice, converge);

        if ((ma_type == adiabat::pseudo_liq) ||
            (ma_type == adiabat::pseudo_ice)) {
            rv_total = pcl_rv_next;
            pcl_rl_next = 0.0f;
            pcl_ri_next = 0.0f;
        }
    }
    pcl_t_next = pcl_theta_next * pcl_pi_next;
    rv = pcl_rv_next;
    rl = pcl_rl_next;
    ri = pcl_ri_next;
    return pcl_t_next;
}



float lapse_rate(HeightLayer layer_agl, const float height[],
                 const float temperature[], const std::ptrdiff_t N) {
#ifndef NO_QC
    if ((layer_agl.bottom == MISSING) || (layer_agl.top == MISSING)) {
        return MISSING;
    }
#endif

    // convert from agl to msl
    layer_agl.bottom += height[0];
    layer_agl.top += height[0];

    // bounds check the height layer
    if (layer_agl.bottom < height[0]) {
        layer_agl.bottom = height[0];
    }
    if (layer_agl.top > height[N - 1]) {
        layer_agl.top = height[N - 1];
    }

    // lower and upper temperature
    const float tmpc_l =
        interp_height(layer_agl.bottom, height, temperature, N);
    const float tmpc_u = interp_height(layer_agl.top, height, temperature, N);
#ifndef NO_QC
    if ((tmpc_l == MISSING) || (tmpc_u == MISSING)) {
        return MISSING;
    }
#endif

    // dT/dz, positive (definition of lapse rate), in km
    const float dz = layer_agl.top - layer_agl.bottom;
    if (dz < 1e-6) return MISSING;
    return ((tmpc_u - tmpc_l) / dz) * -1000.0f;
}

float lapse_rate(PressureLayer layer, const float pressure[],
                 const float height[], const float temperature[],
                 const std::ptrdiff_t N) {
#ifndef NO_QC
    if ((layer.bottom == MISSING) || (layer.top == MISSING)) {
        return MISSING;
    }
#endif

    // bounds check the pressure layer
    if (layer.bottom > pressure[0]) {
        layer.bottom = pressure[0];
    }
    if (layer.top < pressure[N - 1]) {
        layer.top = pressure[N - 1];
    }

    // Each end is clipped only against its own edge of the profile, so a
    // layer wholly outside it comes out inverted. Not a QC check.
    if (layer.bottom < layer.top) return MISSING;

    HeightLayer h_layer =
        pressure_layer_to_height(layer, pressure, height, N, true);

    return lapse_rate(h_layer, height, temperature, N);
}

float lapse_rate_max(HeightLayer layer_agl, const float depth,
                     const float height[], const float temperature[],
                     const std::ptrdiff_t N, HeightLayer* max_lyr) {
    if (max_lyr) {
        max_lyr->bottom = MISSING;
        max_lyr->top = MISSING;
    }
    // Not a QC check: a zero or wrong-signed delta never ends the loop.
    if (!(layer_agl.delta > 0.0f)) return MISSING;
    float max_lr = MISSING;
    for (float z = layer_agl.bottom; z <= (layer_agl.top - depth);
         z += layer_agl.delta) {
        HeightLayer lyr = {z, z + depth};
        const float bottom_msl = lyr.bottom + height[0];
        const float top_msl = lyr.top + height[0];
        if ((bottom_msl < height[0]) || (top_msl > height[N - 1])) continue;
        float lr = lapse_rate(lyr, height, temperature, N);
        if (lr > max_lr) {
            max_lr = lr;
            if (max_lyr) {
                max_lyr->bottom = lyr.bottom;
                max_lyr->top = lyr.top;
            }
        }
    }
    return max_lr;
}

float lapse_rate_max(PressureLayer layer, const float depth,
                     const float pressure[], const float height[],
                     const float temperature[], const std::ptrdiff_t N,
                     PressureLayer* max_lyr) {
    if (max_lyr) {
        max_lyr->bottom = MISSING;
        max_lyr->top = MISSING;
    }
    // Not a QC check: a zero or wrong-signed delta never ends the loop.
    if (!(layer.delta < 0.0f)) return MISSING;
    float max_lr = MISSING;
    for (float p = layer.bottom; p >= (layer.top + depth); p += layer.delta) {
        PressureLayer lyr = {p, p - depth};
        if ((lyr.bottom > pressure[0]) || (lyr.top < pressure[N - 1])) continue;
        float lr = lapse_rate(lyr, pressure, height, temperature, N);
        if (lr > max_lr) {
            max_lr = lr;
            if (max_lyr) {
                max_lyr->bottom = lyr.bottom;
                max_lyr->top = lyr.top;
            }
        }
    }
    return max_lr;
}

void buoyancy(const float pcl_temperature[], const float env_temperature[],
              float buoy_arr[], std::ptrdiff_t N) {
    for (std::ptrdiff_t k = 0; k < N; ++k) {
        buoy_arr[k] = buoyancy(pcl_temperature[k], env_temperature[k]);
    }
}

float pbl_top(const float pressure[], const float thetav[],
              const std::ptrdiff_t N, float offset) {
    float thetav_sfc = thetav[0];
    float thetav_top = thetav_sfc + offset;

    return sharp::find_first_pressure(thetav_top, pressure, thetav, N);
}

PressureLayer temperature_layer(const float pressure[],
                                const float temperature[], const float tmpk_1,
                                const float tmpk_2, const std::ptrdiff_t N,
                                const float pres_min) {
    auto interp_logp = [](float p1, float p2, float w) -> float {
        return std::pow(10.0f, sharp::lerp(std::log10(p1), std::log10(p2), w));
    };

    float layer_bot = MISSING;
    float layer_top = MISSING;

    const float t_lo = std::min(tmpk_1, tmpk_2);
    const float t_hi = std::max(tmpk_1, tmpk_2);

    float last_p = MISSING;
    float last_t = MISSING;
    std::ptrdiff_t idx = N - 1;
    for (; idx >= 0; --idx) {
        if (pressure[idx] == MISSING || temperature[idx] == MISSING) continue;
        if (pressure[idx] < pres_min) continue;
        last_p = pressure[idx];
        last_t = temperature[idx];
        break;
    }
    if (idx < 0) return {MISSING, MISSING};
    --idx;

    for (; idx >= 0; --idx) {
        const float p_curr = pressure[idx];
        const float t_curr = temperature[idx];

        if (p_curr == MISSING || t_curr == MISSING) continue;

        const float t1 = last_t, t2 = t_curr;
        const float p1 = last_p, p2 = p_curr;
        float w_in = 0.0f;
        float w_out = 1.0f;
        const float dt = t2 - t1;

        if (std::abs(dt) > 1e-5f) {
            const float inv_dt = 1.0f / dt;
            const float w1 = (t_lo - t1) * inv_dt;
            const float w2 = (t_hi - t1) * inv_dt;
            w_in = std::max(0.0f, std::min(w1, w2));
            w_out = std::min(1.0f, std::max(w1, w2));
        } else {
            if (t1 >= t_lo && t1 <= t_hi) {
                w_in = 0.0f;
                w_out = 1.0f;
            } else {
                w_in = 1.0f;
                w_out = 0.0f;
            }
        }

        if (w_in < w_out) {
            if (layer_top == MISSING) {
                layer_top = interp_logp(p1, p2, w_in);
            }

            layer_bot = interp_logp(p1, p2, w_out);

            if (w_out < 1.0f - 1e-5f) {
                break;
            }
        } else if (layer_top != MISSING) {
            break;
        }

        last_p = p_curr;
        last_t = t_curr;
    }

    return {layer_bot, layer_top};
}

/// @cond DOXYGEN_IGNORE

template float wetbulb<lifter_wobus>(lifter_wobus& lifter, float pressure,
                                     float temperature, float dewpoint);

template float wetbulb<lifter_cm1>(lifter_cm1& lifter, float pressure,
                                   float temperature, float dewpoint);

template float wetbulb<lifter_lut<lifter_wobus>>(
    lifter_lut<lifter_wobus>& lifter, float pressure, float temperature,
    float dewpoint);

template float wetbulb<lifter_lut<lifter_cm1>>(lifter_lut<lifter_cm1>& lifter,
                                               float pressure,
                                               float temperature,
                                               float dewpoint);

template float theta_wetbulb<lifter_wobus>(lifter_wobus& lifter, float pressure,
                                           float temperature, float dewpoint);

template float theta_wetbulb<lifter_cm1>(lifter_cm1& lifter, float pressure,
                                         float temperature, float dewpoint);

template float theta_wetbulb<lifter_lut<lifter_cm1>>(
    lifter_lut<lifter_cm1>& lifter, float pressure, float temperature,
    float dewpoint);

template float theta_wetbulb<lifter_lut<lifter_wobus>>(
    lifter_lut<lifter_wobus>& lifter, float pressure, float temperature,
    float dewpoint);

/// @endcond

}  // end namespace sharp
