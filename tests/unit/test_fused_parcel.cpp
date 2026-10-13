#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <SHARPlib/constants.h>
#include <SHARPlib/gpu/fused_parcel.cuh>
#include <SHARPlib/gpu/strided_ptr.h>
#include <SHARPlib/layer.h>
#include <SHARPlib/parcel.h>
#include <SHARPlib/thermo.h>

#include <cmath>
#include <iostream>
#include <memory>

#include "doctest.h"

namespace {

auto array_from_range = [](const float bottom, const float top,
                           const std::ptrdiff_t size) {
    auto arr = std::make_unique<float[]>(size);
    float delta = (top - bottom) / static_cast<float>(size);
    for (std::ptrdiff_t k = 0; k < size; ++k) {
        arr[k] = bottom + static_cast<float>(k) * delta;
    }
    return arr;
};

auto tmpk_snd = [](const float tmpk_sfc, const float delta_tmpk_cap,
                   const float delta_tmpk_troposphere, const float hght_pbl_top,
                   const float hght_trop_top, const float height[],
                   const std::ptrdiff_t size) {
    sharp::HeightLayer free_troposphere =
        sharp::HeightLayer(hght_pbl_top, hght_trop_top);
    sharp::LayerIndex idx =
        sharp::get_layer_index(free_troposphere, height, size);

    const float dse_pbl = sharp::moist_static_energy(0.0f, tmpk_sfc, 0.0f);
    const float delta_dse_trop =
        sharp::GRAVITY - (sharp::CP_DRYAIR * delta_tmpk_troposphere);

    auto tmpk_arr = std::make_unique<float[]>(size);
    tmpk_arr[0] = tmpk_sfc;

    for (std::ptrdiff_t k = 1; k < idx.kbot; ++k) {
        tmpk_arr[k] = (dse_pbl - height[k] * sharp::GRAVITY) / sharp::CP_DRYAIR;
    }

    const float dse_troposphere_bottom = sharp::moist_static_energy(
        height[idx.kbot], tmpk_arr[idx.kbot - 1] + delta_tmpk_cap, 0.0f);

    for (std::ptrdiff_t k = idx.kbot; k < idx.ktop + 1; ++k) {
        float dse = dse_troposphere_bottom +
                    delta_dse_trop * (height[k] - hght_pbl_top);
        tmpk_arr[k] = (dse - sharp::GRAVITY * height[k]) / sharp::CP_DRYAIR;
    }

    for (std::ptrdiff_t k = idx.ktop + 1; k < size; ++k) {
        tmpk_arr[k] = tmpk_arr[idx.ktop];
    }

    return tmpk_arr;
};

auto pres_dry_snd = [](const float pres_sfc, const float height[],
                       const float tmpk[], const std::ptrdiff_t size) {
    auto pres_arr = std::make_unique<float[]>(size);
    pres_arr[0] = pres_sfc;

    for (std::ptrdiff_t k = 1; k < size; ++k) {
        float tmpk_mean = 0.5f * (tmpk[k] + tmpk[k - 1]);
        float delta_z = height[k] - height[k - 1];
        float delta_p =
            -(sharp::GRAVITY * delta_z) / (sharp::RDGAS * tmpk_mean);
        pres_arr[k] = pres_arr[k - 1] * std::exp(delta_p);
    }

    return pres_arr;
};

auto pres_moist_snd = [](const float pres_sfc, const float height[],
                         const float tmpk[], const float mixr[],
                         const std::ptrdiff_t size) {
    auto pres_arr = std::make_unique<float[]>(size);
    pres_arr[0] = pres_sfc;

    for (std::ptrdiff_t k = 1; k < size; ++k) {
        const float tmpk_mean = 0.5f * (tmpk[k] + tmpk[k - 1]);
        const float mixr_mean = 0.5f * (mixr[k] + mixr[k - 1]);
        const float tv_mean = sharp::virtual_temperature(tmpk_mean, mixr_mean);
        const float delta_z = height[k] - height[k - 1];
        const float delta_p =
            -(sharp::GRAVITY * delta_z) / (sharp::RDGAS * tv_mean);
        pres_arr[k] = pres_arr[k - 1] * std::exp(delta_p);
    }

    return pres_arr;
};

auto mixr_snd = [](const float pres_sfc, const float mixr_sfc,
                   const float relh_troposphere, const float height[],
                   const float tmpk[], const std::ptrdiff_t size) {
    auto pres_arr = pres_dry_snd(pres_sfc, height, tmpk, size);
    auto mixr_arr = std::make_unique<float[]>(size);
    mixr_arr[0] = mixr_sfc;

    for (std::ptrdiff_t k = 1; k < size; ++k) {
        const float sat_mixr = sharp::mixratio(pres_arr[k], tmpk[k]);
        mixr_arr[k] = relh_troposphere * sat_mixr;
    }

    return mixr_arr;
};

auto dwpk_snd = [](const float pres[], const float mixr[],
                   const std::ptrdiff_t size) {
    auto dwpk_arr = std::make_unique<float[]>(size);
    for (std::ptrdiff_t k = 0; k < size; ++k) {
        dwpk_arr[k] = sharp::temperature_at_mixratio(mixr[k], pres[k]);
    }
    return dwpk_arr;
};

auto vtmpk_snd = [](const float tmpk[], const float mixr[],
                    const std::ptrdiff_t size) {
    auto vtmpk_arr = std::make_unique<float[]>(size);
    for (std::ptrdiff_t k = 0; k < size; ++k) {
        vtmpk_arr[k] = sharp::virtual_temperature(tmpk[k], mixr[k]);
    }
    return vtmpk_arr;
};

struct SoundingData {
    std::unique_ptr<float[]> hght;
    std::unique_ptr<float[]> pres;
    std::unique_ptr<float[]> tmpk;
    std::unique_ptr<float[]> dwpk;
    std::unique_ptr<float[]> mixr;
    std::unique_ptr<float[]> vtmpk;
    std::ptrdiff_t n;
};

SoundingData generate_sounding(float tmpk_sfc, float mixr_sfc,
                               float delta_tmpk_cap, float delta_tmpk_trop,
                               float relh_troposphere = 0.85f,
                               std::ptrdiff_t n = 5000) {
    constexpr float pres_sfc = 100000.0f;
    constexpr float hght_sfc = 0.0f;
    constexpr float hght_top = 15000.0f;
    constexpr float hght_tropopause = 12000.0f;
    constexpr float hght_pbl_top = 850.0f;

    auto hght = array_from_range(hght_sfc, hght_top, n);
    auto tmpk = tmpk_snd(tmpk_sfc, delta_tmpk_cap, delta_tmpk_trop,
                         hght_pbl_top, hght_tropopause, hght.get(), n);
    auto mixr = mixr_snd(pres_sfc, mixr_sfc, relh_troposphere, hght.get(),
                         tmpk.get(), n);
    auto pres = pres_moist_snd(pres_sfc, hght.get(), tmpk.get(), mixr.get(), n);
    auto dwpk = dwpk_snd(pres.get(), mixr.get(), n);
    auto vtmpk = vtmpk_snd(tmpk.get(), mixr.get(), n);

    return {std::move(hght), std::move(pres), std::move(tmpk),
            std::move(dwpk), std::move(mixr), std::move(vtmpk), n};
}

} // namespace

TEST_CASE("testing fused_parcel vs two-pass Parcel with wobus lifter on standard sounding") {
    auto snd = generate_sounding(301.5f, 0.0157f, 1.0f, 0.00725f, 0.85f, 5000);

    // Two-pass SHARPlib Parcel
    auto buoy = std::make_unique<float[]>(snd.n);
    auto pcl_vtmpk = std::make_unique<float[]>(snd.n);
    sharp::Parcel sfc_pcl =
        sharp::Parcel::surface_parcel(snd.pres[0], snd.tmpk[0], snd.dwpk[0]);
    sharp::lifter_wobus lifter;

    sfc_pcl.lift_parcel(lifter, snd.pres.get(), pcl_vtmpk.get(), snd.n);
    sharp::buoyancy(pcl_vtmpk.get(), snd.vtmpk.get(), buoy.get(), snd.n);
    sfc_pcl.cape_cinh(snd.pres.get(), snd.hght.get(), buoy.get(), snd.n);

    // Fused parcel
    auto out = sharp::gpu::lift_and_integrate_column(
        lifter, snd.pres[0], snd.tmpk[0], snd.mixr[0],
        snd.pres.get(), snd.hght.get(), snd.tmpk.get(),
        snd.mixr.get(), snd.vtmpk.get(), static_cast<int>(snd.n));

    // Verify numerical equivalence
    CHECK(out.cape == doctest::Approx(sfc_pcl.cape).epsilon(0.001));
    CHECK(std::abs(out.cape - sfc_pcl.cape) <= 0.5f);
    CHECK(out.cin == doctest::Approx(sfc_pcl.cinh).epsilon(0.001));
    CHECK(std::abs(out.cin - sfc_pcl.cinh) <= 0.5f);

    CHECK(out.lcl_pres == doctest::Approx(sfc_pcl.lcl_pressure).epsilon(0.001));
    CHECK(out.lfc_pres == doctest::Approx(sfc_pcl.lfc_pressure).epsilon(0.001));
    CHECK(out.el_pres == doctest::Approx(sfc_pcl.eql_pressure).epsilon(0.001));

    float ref_lcl_hght_agl =
        sharp::interp_pressure(sfc_pcl.lcl_pressure, snd.pres.get(), snd.hght.get(), snd.n) - snd.hght[0];
    float ref_lfc_hght_agl =
        sharp::interp_pressure(sfc_pcl.lfc_pressure, snd.pres.get(), snd.hght.get(), snd.n) - snd.hght[0];
    float ref_el_hght_agl =
        sharp::interp_pressure(sfc_pcl.eql_pressure, snd.pres.get(), snd.hght.get(), snd.n) - snd.hght[0];

    CHECK(std::abs(out.lcl_hght_agl - ref_lcl_hght_agl) <= 1.0f);
    CHECK(std::abs(out.lfc_hght_agl - ref_lfc_hght_agl) <= 1.0f);
    CHECK(std::abs(out.el_hght_agl - ref_el_hght_agl) <= 1.0f);
}

TEST_CASE("testing fused_parcel with varying atmospheric profiles") {
    struct TestCase {
        const char* name;
        float tmpk_sfc;
        float mixr_sfc;
        float delta_cap;
        float delta_trop;
        float relh;
    } cases[] = {
        {"High CAPE, moderate CIN", 305.0f, 0.018f, 2.5f, 0.008f, 0.88f},
        {"Moderate CAPE, weak CIN", 298.0f, 0.013f, 0.5f, 0.007f, 0.80f},
        {"Low CAPE, high CIN", 293.0f, 0.009f, 4.0f, 0.0065f, 0.70f},
    };

    sharp::lifter_wobus lifter;

    for (const auto& tc : cases) {
        CAPTURE(tc.name);
        auto snd = generate_sounding(tc.tmpk_sfc, tc.mixr_sfc, tc.delta_cap,
                                     tc.delta_trop, tc.relh, 4000);

        auto buoy = std::make_unique<float[]>(snd.n);
        auto pcl_vtmpk = std::make_unique<float[]>(snd.n);
        sharp::Parcel sfc_pcl =
            sharp::Parcel::surface_parcel(snd.pres[0], snd.tmpk[0], snd.dwpk[0]);

        sfc_pcl.lift_parcel(lifter, snd.pres.get(), pcl_vtmpk.get(), snd.n);
        sharp::buoyancy(pcl_vtmpk.get(), snd.vtmpk.get(), buoy.get(), snd.n);
        sfc_pcl.cape_cinh(snd.pres.get(), snd.hght.get(), buoy.get(), snd.n);

        auto out = sharp::gpu::lift_and_integrate_column(
            lifter, snd.pres[0], snd.tmpk[0], snd.mixr[0],
            snd.pres.get(), snd.hght.get(), snd.tmpk.get(),
            snd.mixr.get(), snd.vtmpk.get(), static_cast<int>(snd.n));

        if (sfc_pcl.cape > 0.0f) {
            CHECK(out.cape == doctest::Approx(sfc_pcl.cape).epsilon(0.001));
            CHECK(std::abs(out.cape - sfc_pcl.cape) <= 0.5f);
            CHECK(out.cin == doctest::Approx(sfc_pcl.cinh).epsilon(0.001));
            CHECK(std::abs(out.cin - sfc_pcl.cinh) <= 0.5f);

            CHECK(out.lcl_pres == doctest::Approx(sfc_pcl.lcl_pressure).epsilon(0.001));
            CHECK(out.lfc_pres == doctest::Approx(sfc_pcl.lfc_pressure).epsilon(0.001));
            CHECK(out.el_pres == doctest::Approx(sfc_pcl.eql_pressure).epsilon(0.001));
        } else {
            CHECK(out.cape == 0.0f);
        }
    }
}

TEST_CASE("testing fused_parcel with completely capped / zero CAPE sounding") {
    // Strongly inverted atmosphere where no parcel can achieve positive buoyancy
    auto snd = generate_sounding(285.0f, 0.004f, 15.0f, 0.004f, 0.40f, 2000);

    sharp::lifter_wobus lifter;
    auto buoy = std::make_unique<float[]>(snd.n);
    auto pcl_vtmpk = std::make_unique<float[]>(snd.n);
    sharp::Parcel sfc_pcl =
        sharp::Parcel::surface_parcel(snd.pres[0], snd.tmpk[0], snd.dwpk[0]);

    sfc_pcl.lift_parcel(lifter, snd.pres.get(), pcl_vtmpk.get(), snd.n);
    sharp::buoyancy(pcl_vtmpk.get(), snd.vtmpk.get(), buoy.get(), snd.n);
    sfc_pcl.cape_cinh(snd.pres.get(), snd.hght.get(), buoy.get(), snd.n);

    auto out = sharp::gpu::lift_and_integrate_column(
        lifter, snd.pres[0], snd.tmpk[0], snd.mixr[0],
        snd.pres.get(), snd.hght.get(), snd.tmpk.get(),
        snd.mixr.get(), snd.vtmpk.get(), static_cast<int>(snd.n));

    CHECK(sfc_pcl.cape == 0.0f);
    CHECK(out.cape == 0.0f);
    CHECK(out.cin == 0.0f);
    CHECK(out.lfc_pres == sharp::MISSING);
    CHECK(out.el_pres == sharp::MISSING);
}

TEST_CASE("testing fused_parcel with StridedPtr abstraction") {
    auto snd = generate_sounding(301.5f, 0.0157f, 1.0f, 0.00725f, 0.85f, 1000);

    // Simulate multi-profile stride: stride of 2 floats
    constexpr int stride = 2;
    auto pres_strided = std::make_unique<float[]>(snd.n * stride);
    auto hght_strided = std::make_unique<float[]>(snd.n * stride);
    auto tmpk_strided = std::make_unique<float[]>(snd.n * stride);
    auto mixr_strided = std::make_unique<float[]>(snd.n * stride);
    auto vtmpk_strided = std::make_unique<float[]>(snd.n * stride);

    for (std::ptrdiff_t k = 0; k < snd.n; ++k) {
        pres_strided[k * stride] = snd.pres[k];
        hght_strided[k * stride] = snd.hght[k];
        tmpk_strided[k * stride] = snd.tmpk[k];
        mixr_strided[k * stride] = snd.mixr[k];
        vtmpk_strided[k * stride] = snd.vtmpk[k];
    }

    sharp::StridedPtr<const float> p_ptr(pres_strided.get(), stride);
    sharp::StridedPtr<const float> z_ptr(hght_strided.get(), stride);
    sharp::StridedPtr<const float> t_ptr(tmpk_strided.get(), stride);
    sharp::StridedPtr<const float> r_ptr(mixr_strided.get(), stride);
    sharp::StridedPtr<const float> tv_ptr(vtmpk_strided.get(), stride);

    sharp::lifter_wobus lifter;
    auto out = sharp::gpu::lift_and_integrate_column(
        lifter, snd.pres[0], snd.tmpk[0], snd.mixr[0],
        p_ptr, z_ptr, t_ptr, r_ptr, tv_ptr, static_cast<int>(snd.n));

    auto out_ref = sharp::gpu::lift_and_integrate_column(
        lifter, snd.pres[0], snd.tmpk[0], snd.mixr[0],
        snd.pres.get(), snd.hght.get(), snd.tmpk.get(),
        snd.mixr.get(), snd.vtmpk.get(), static_cast<int>(snd.n));

    CHECK(out.cape == out_ref.cape);
    CHECK(out.cin == out_ref.cin);
    CHECK(out.lcl_pres == out_ref.lcl_pres);
    CHECK(out.lfc_pres == out_ref.lfc_pres);
    CHECK(out.el_pres == out_ref.el_pres);
    CHECK(out.lcl_hght_agl == out_ref.lcl_hght_agl);
    CHECK(out.lfc_hght_agl == out_ref.lfc_hght_agl);
    CHECK(out.el_hght_agl == out_ref.el_hght_agl);
}

TEST_CASE("testing fused_parcel alternate argument order overload") {
    auto snd = generate_sounding(301.5f, 0.0157f, 1.0f, 0.00725f, 0.85f, 1000);
    sharp::lifter_wobus lifter;

    auto out1 = sharp::gpu::lift_and_integrate_column(
        lifter, snd.pres[0], snd.tmpk[0], snd.mixr[0],
        snd.pres.get(), snd.hght.get(), snd.tmpk.get(),
        snd.mixr.get(), snd.vtmpk.get(), static_cast<int>(snd.n));

    auto out2 = sharp::gpu::lift_and_integrate_column(
        lifter,
        snd.pres.get(), snd.hght.get(), snd.tmpk.get(),
        snd.mixr.get(), snd.vtmpk.get(), static_cast<int>(snd.n),
        snd.pres[0], snd.tmpk[0], snd.mixr[0]);

    CHECK(out1.cape == out2.cape);
    CHECK(out1.cin == out2.cin);
    CHECK(out1.lcl_pres == out2.lcl_pres);
    CHECK(out1.lfc_pres == out2.lfc_pres);
    CHECK(out1.el_pres == out2.el_pres);
}
