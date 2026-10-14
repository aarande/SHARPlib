#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <cuda_runtime.h>
#include <SHARPlib/constants.h>
#include <SHARPlib/gpu/batch_kernels.h>
#include <SHARPlib/gpu/lut_texture.cuh>
#include <SHARPlib/thermo.h>
#include <SHARPlib/winds.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "doctest.h"

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            FAIL("CUDA error: " << cudaGetErrorString(err) << " at "     \
                                << __FILE__ << ":" << __LINE__);         \
        }                                                                \
    } while (0)

TEST_CASE("Testing GPU batched convective analysis kernel") {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        WARN("No CUDA-capable device found, skipping test_cuda_batch");
        return;
    }

    constexpr int n_levels = 50;
    constexpr int n_profiles = 64;

    // Generate a synthetic sounding profile with surface temp = 305 K, dwpk = 295 K,
    // strong instability, and wind shear.
    std::vector<float> h_pres(n_levels * n_profiles);
    std::vector<float> h_hght(n_levels * n_profiles);
    std::vector<float> h_tmpk(n_levels * n_profiles);
    std::vector<float> h_dwpk(n_levels * n_profiles);
    std::vector<float> h_u(n_levels * n_profiles);
    std::vector<float> h_v(n_levels * n_profiles);

    for (int p = 0; p < n_profiles; ++p) {
        float p_sfc = 100000.0f;
        float z_sfc = 100.0f + static_cast<float>(p) * 2.0f;
        float t_sfc = 303.0f + static_cast<float>(p % 5) * 0.5f;
        float td_sfc = 299.0f + static_cast<float>(p % 3) * 0.5f;

        for (int k = 0; k < n_levels; ++k) {
            float frac = static_cast<float>(k) / static_cast<float>(n_levels - 1);
            float z = z_sfc + frac * 16000.0f;  // up to 16 km MSL
            float pres = p_sfc * std::exp(-(z - z_sfc) / 8000.0f);
            float z_agl = z - z_sfc;

            // Temperature: lapse rate 6.5 C/km up to 11 km, isothermal above
            float temp;
            if (z_agl < 11000.0f) {
                temp = t_sfc - 0.0065f * z_agl;
            } else {
                temp = t_sfc - 0.0065f * 11000.0f;
            }

            // Dewpoint: moist near surface, dry aloft
            float dwpk;
            if (z_agl < 1500.0f) {
                dwpk = td_sfc - 0.001f * z_agl;
            } else if (z_agl < 6000.0f) {
                dwpk = td_sfc - 0.001f * 1500.0f - 0.006f * (z_agl - 1500.0f);
            } else {
                dwpk = temp - 25.0f;
            }

            // Wind profile: south-southeasterly at surface turning southwesterly and strengthening aloft
            float u = -8.0f + 35.0f * (z_agl / 12000.0f);
            float v = 8.0f + 25.0f * (z_agl / 12000.0f);

            std::size_t idx = static_cast<std::size_t>(k) * n_profiles + p;
            h_pres[idx] = pres;
            h_hght[idx] = z;
            h_tmpk[idx] = temp;
            h_dwpk[idx] = dwpk;
            h_u[idx] = u;
            h_v[idx] = v;
        }
    }

    const std::size_t grid_bytes = static_cast<std::size_t>(n_levels) * n_profiles * sizeof(float);
    const std::size_t out_bytes = static_cast<std::size_t>(n_profiles) * sizeof(float);

    float *d_pres = nullptr, *d_hght = nullptr, *d_tmpk = nullptr, *d_dwpk = nullptr, *d_u = nullptr, *d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pres, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_hght, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmpk, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_dwpk, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_u, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_v, grid_bytes));

    CUDA_CHECK(cudaMemcpy(d_pres, h_pres.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hght, h_hght.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tmpk, h_tmpk.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dwpk, h_dwpk.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), grid_bytes, cudaMemcpyHostToDevice));

    float *d_sbcape = nullptr, *d_sbcin = nullptr;
    float *d_mlcape = nullptr, *d_mlcin = nullptr;
    float *d_mucape = nullptr, *d_mucin = nullptr;
    float *d_srh01 = nullptr, *d_srh03 = nullptr;
    float *d_shr06 = nullptr, *d_stp = nullptr;

    CUDA_CHECK(cudaMalloc(&d_sbcape, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_sbcin, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_mlcape, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_mlcin, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_mucape, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_mucin, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_srh01, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_srh03, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_shr06, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_stp, out_bytes));

    SUBCASE("Texture LUT mode with fast MU thetae") {
        sharp::gpu::launch_convective_batch(
            d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
            d_sbcape, d_sbcin,
            d_mlcape, d_mlcin,
            d_mucape, d_mucin,
            d_srh01, d_srh03,
            d_shr06, d_stp,
            n_levels, n_profiles,
            true,   // use_texture_lut
            true    // fast_mu_thetae
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> sbcape(n_profiles), sbcin(n_profiles);
        std::vector<float> mlcape(n_profiles), mlcin(n_profiles);
        std::vector<float> mucape(n_profiles), mucin(n_profiles);
        std::vector<float> srh01(n_profiles), srh03(n_profiles);
        std::vector<float> shr06(n_profiles), stp(n_profiles);

        CUDA_CHECK(cudaMemcpy(sbcape.data(), d_sbcape, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sbcin.data(), d_sbcin, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlcape.data(), d_mlcape, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlcin.data(), d_mlcin, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mucape.data(), d_mucape, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mucin.data(), d_mucin, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(srh01.data(), d_srh01, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(srh03.data(), d_srh03, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(shr06.data(), d_shr06, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(stp.data(), d_stp, out_bytes, cudaMemcpyDeviceToHost));

        for (int p = 0; p < n_profiles; ++p) {
            CHECK(sbcape[p] > 500.0f);
            CHECK(sbcin[p] <= 0.0f);
            CHECK(mlcape[p] > 200.0f);
            CHECK(mucape[p] >= sbcape[p]);
            CHECK(srh01[p] > 0.0f);
            CHECK(srh03[p] > srh01[p]);
            CHECK(shr06[p] > 10.0f);
            CHECK(stp[p] > 0.0f);
        }
    }

    SUBCASE("Wobus lifter mode with exact MU search") {
        sharp::gpu::launch_convective_batch(
            d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
            d_sbcape, d_sbcin,
            d_mlcape, d_mlcin,
            d_mucape, d_mucin,
            d_srh01, d_srh03,
            d_shr06, d_stp,
            n_levels, n_profiles,
            false,  // use_texture_lut
            false   // fast_mu_thetae
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> sbcape(n_profiles);
        std::vector<float> mucape(n_profiles);
        CUDA_CHECK(cudaMemcpy(sbcape.data(), d_sbcape, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mucape.data(), d_mucape, out_bytes, cudaMemcpyDeviceToHost));

        for (int p = 0; p < n_profiles; ++p) {
            CHECK(sbcape[p] > 500.0f);
            CHECK(mucape[p] >= sbcape[p]);
        }
    }

    CUDA_CHECK(cudaFree(d_pres));
    CUDA_CHECK(cudaFree(d_hght));
    CUDA_CHECK(cudaFree(d_tmpk));
    CUDA_CHECK(cudaFree(d_dwpk));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));

    CUDA_CHECK(cudaFree(d_sbcape));
    CUDA_CHECK(cudaFree(d_sbcin));
    CUDA_CHECK(cudaFree(d_mlcape));
    CUDA_CHECK(cudaFree(d_mlcin));
    CUDA_CHECK(cudaFree(d_mucape));
    CUDA_CHECK(cudaFree(d_mucin));
    CUDA_CHECK(cudaFree(d_srh01));
    CUDA_CHECK(cudaFree(d_srh03));
    CUDA_CHECK(cudaFree(d_shr06));
    CUDA_CHECK(cudaFree(d_stp));
}

TEST_CASE("Testing GPU batched convective analysis kernel - Full Catalog and Pruning") {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        WARN("No CUDA-capable device found, skipping test_cuda_batch");
        return;
    }

    constexpr int n_levels = 50;
    constexpr int n_profiles = 64;

    std::vector<float> h_pres(n_levels * n_profiles);
    std::vector<float> h_hght(n_levels * n_profiles);
    std::vector<float> h_tmpk(n_levels * n_profiles);
    std::vector<float> h_dwpk(n_levels * n_profiles);
    std::vector<float> h_u(n_levels * n_profiles);
    std::vector<float> h_v(n_levels * n_profiles);

    for (int p = 0; p < n_profiles; ++p) {
        float p_sfc = 100000.0f;
        float z_sfc = 100.0f + static_cast<float>(p) * 2.0f;
        float t_sfc = 303.0f + static_cast<float>(p % 5) * 0.5f;
        float td_sfc = 299.0f + static_cast<float>(p % 3) * 0.5f;

        for (int k = 0; k < n_levels; ++k) {
            float frac = static_cast<float>(k) / static_cast<float>(n_levels - 1);
            float z = z_sfc + frac * 16000.0f;
            float pres = p_sfc * std::exp(-(z - z_sfc) / 8000.0f);
            float z_agl = z - z_sfc;

            float temp;
            if (z_agl < 11000.0f) {
                temp = t_sfc - 0.0065f * z_agl;
            } else {
                temp = t_sfc - 0.0065f * 11000.0f;
            }

            float dwpk;
            if (z_agl < 1500.0f) {
                dwpk = td_sfc - 0.001f * z_agl;
            } else if (z_agl < 6000.0f) {
                dwpk = td_sfc - 0.001f * 1500.0f - 0.006f * (z_agl - 1500.0f);
            } else {
                dwpk = temp - 25.0f;
            }

            float u = -8.0f + 35.0f * (z_agl / 12000.0f);
            float v = 8.0f + 25.0f * (z_agl / 12000.0f);

            std::size_t idx = static_cast<std::size_t>(k) * n_profiles + p;
            h_pres[idx] = pres;
            h_hght[idx] = z;
            h_tmpk[idx] = temp;
            h_dwpk[idx] = dwpk;
            h_u[idx] = u;
            h_v[idx] = v;
        }
    }

    const std::size_t grid_bytes = static_cast<std::size_t>(n_levels) * n_profiles * sizeof(float);
    const std::size_t out_bytes = static_cast<std::size_t>(n_profiles) * sizeof(float);

    float *d_pres = nullptr, *d_hght = nullptr, *d_tmpk = nullptr, *d_dwpk = nullptr, *d_u = nullptr, *d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pres, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_hght, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmpk, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_dwpk, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_u, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_v, grid_bytes));

    CUDA_CHECK(cudaMemcpy(d_pres, h_pres.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hght, h_hght.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tmpk, h_tmpk.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dwpk, h_dwpk.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), grid_bytes, cudaMemcpyHostToDevice));

    struct DeviceOutputs {
        sharp::gpu::ConvectiveBatchOutputs host_ptrs;
        std::vector<float*> allocated;

        float* alloc(std::size_t bytes) {
            float* ptr = nullptr;
            cudaMalloc(&ptr, bytes);
            allocated.push_back(ptr);
            return ptr;
        }

        void init_all(std::size_t bytes) {
            #define ALLOC_F(f) host_ptrs.f = alloc(bytes)
            ALLOC_F(sbcape); ALLOC_F(sbcin); ALLOC_F(sblcl_hght_agl); ALLOC_F(sblcl_pres);
            ALLOC_F(sblfc_hght_agl); ALLOC_F(sblfc_pres); ALLOC_F(sbel_hght_agl); ALLOC_F(sbel_pres); ALLOC_F(sb3cape);
            ALLOC_F(mlcape); ALLOC_F(mlcin); ALLOC_F(mllcl_hght_agl); ALLOC_F(mllcl_pres);
            ALLOC_F(mllfc_hght_agl); ALLOC_F(mllfc_pres); ALLOC_F(mlel_hght_agl); ALLOC_F(mlel_pres); ALLOC_F(ml3cape);
            ALLOC_F(mucape); ALLOC_F(mucin); ALLOC_F(mulcl_hght_agl); ALLOC_F(mulcl_pres);
            ALLOC_F(mulfc_hght_agl); ALLOC_F(mulfc_pres); ALLOC_F(muel_hght_agl); ALLOC_F(muel_pres); ALLOC_F(mu3cape);
            ALLOC_F(bunkers_rm_u); ALLOC_F(bunkers_rm_v); ALLOC_F(bunkers_lm_u); ALLOC_F(bunkers_lm_v);
            ALLOC_F(mean_wind_u); ALLOC_F(mean_wind_v);
            ALLOC_F(shr01); ALLOC_F(shr03); ALLOC_F(shr06);
            ALLOC_F(srh01); ALLOC_F(srh03); ALLOC_F(srh05); ALLOC_F(crit_angle);
            ALLOC_F(pwat); ALLOC_F(lr03); ALLOC_F(lr36); ALLOC_F(lr75);
            ALLOC_F(fz_level_agl); ALLOC_F(wbz_level_agl);
            ALLOC_F(stp); ALLOC_F(scp); ALLOC_F(ehi01); ALLOC_F(ehi03);
            #undef ALLOC_F
        }

        ~DeviceOutputs() {
            for (float* p : allocated) {
                cudaFree(p);
            }
        }
    };

    SUBCASE("Flexible full parameter catalog execution") {
        DeviceOutputs dev;
        dev.init_all(out_bytes);

        sharp::gpu::launch_convective_batch_flexible(
            d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
            dev.host_ptrs,
            n_levels, n_profiles,
            true,   // use_texture_lut
            true    // fast_mu_thetae
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        auto read_buf = [&](float* dev_ptr) {
            std::vector<float> res(n_profiles);
            CUDA_CHECK(cudaMemcpy(res.data(), dev_ptr, out_bytes, cudaMemcpyDeviceToHost));
            return res;
        };

        auto sbcape = read_buf(dev.host_ptrs.sbcape);
        auto sbcin = read_buf(dev.host_ptrs.sbcin);
        auto sblcl_hght = read_buf(dev.host_ptrs.sblcl_hght_agl);
        auto sblcl_pres = read_buf(dev.host_ptrs.sblcl_pres);
        auto sblfc_hght = read_buf(dev.host_ptrs.sblfc_hght_agl);
        auto sbel_hght = read_buf(dev.host_ptrs.sbel_hght_agl);
        auto sb3cape = read_buf(dev.host_ptrs.sb3cape);

        auto mlcape = read_buf(dev.host_ptrs.mlcape);
        auto ml3cape = read_buf(dev.host_ptrs.ml3cape);
        auto mucape = read_buf(dev.host_ptrs.mucape);
        auto mu3cape = read_buf(dev.host_ptrs.mu3cape);

        auto b_rm_u = read_buf(dev.host_ptrs.bunkers_rm_u);
        auto b_rm_v = read_buf(dev.host_ptrs.bunkers_rm_v);
        auto b_lm_u = read_buf(dev.host_ptrs.bunkers_lm_u);
        auto b_lm_v = read_buf(dev.host_ptrs.bunkers_lm_v);
        auto mw_u = read_buf(dev.host_ptrs.mean_wind_u);
        auto mw_v = read_buf(dev.host_ptrs.mean_wind_v);
        auto shr01 = read_buf(dev.host_ptrs.shr01);
        auto shr03 = read_buf(dev.host_ptrs.shr03);
        auto shr06 = read_buf(dev.host_ptrs.shr06);
        auto srh01 = read_buf(dev.host_ptrs.srh01);
        auto srh03 = read_buf(dev.host_ptrs.srh03);
        auto srh05 = read_buf(dev.host_ptrs.srh05);
        auto crit_angle = read_buf(dev.host_ptrs.crit_angle);

        auto pwat = read_buf(dev.host_ptrs.pwat);
        auto lr03 = read_buf(dev.host_ptrs.lr03);
        auto lr36 = read_buf(dev.host_ptrs.lr36);
        auto lr75 = read_buf(dev.host_ptrs.lr75);
        auto fz_agl = read_buf(dev.host_ptrs.fz_level_agl);
        auto wbz_agl = read_buf(dev.host_ptrs.wbz_level_agl);

        auto stp = read_buf(dev.host_ptrs.stp);
        auto scp = read_buf(dev.host_ptrs.scp);
        auto ehi01 = read_buf(dev.host_ptrs.ehi01);
        auto ehi03 = read_buf(dev.host_ptrs.ehi03);

        for (int p = 0; p < n_profiles; ++p) {
            // SB, ML, MU
            CHECK(sbcape[p] > 500.0f);
            CHECK(sbcin[p] <= 0.0f);
            CHECK(sblcl_hght[p] > 0.0f);
            CHECK(sblcl_pres[p] > 0.0f);
            CHECK(sblcl_pres[p] < 100000.0f);
            CHECK(sblfc_hght[p] >= sblcl_hght[p]);
            CHECK(sbel_hght[p] > sblfc_hght[p]);
            CHECK(sb3cape[p] >= 0.0f);
            CHECK(sb3cape[p] <= sbcape[p]);

            CHECK(mlcape[p] > 200.0f);
            CHECK(ml3cape[p] >= 0.0f);
            CHECK(mucape[p] >= sbcape[p]);
            CHECK(mu3cape[p] >= 0.0f);

            // Kinematics
            CHECK(shr01[p] > 0.0f);
            CHECK(shr03[p] > shr01[p]);
            CHECK(shr06[p] > shr03[p]);
            CHECK(srh05[p] > 0.0f);
            CHECK(srh01[p] > 0.0f);
            CHECK(srh03[p] > srh01[p]);
            CHECK(crit_angle[p] >= 0.0f);
            CHECK(crit_angle[p] <= 180.0f);
            CHECK(std::abs(b_rm_u[p]) > 0.0f);
            CHECK(std::abs(b_lm_u[p]) > 0.0f);

            // Thermo
            CHECK(pwat[p] > 10.0f); // mm of PWAT
            CHECK(lr03[p] > 4.0f);  // ~6.5 C/km
            CHECK(lr36[p] > 4.0f);
            CHECK(lr75[p] > 4.0f);
            CHECK(fz_agl[p] > 2000.0f);
            CHECK(wbz_agl[p] > 1000.0f);

            // Composite
            CHECK(stp[p] > 0.0f);
            CHECK(scp[p] > 0.0f);
            CHECK(ehi01[p] > 0.0f);
            CHECK(ehi03[p] > 0.0f);
        }
    }

    SUBCASE("Selective execution pruning - Thermo only") {
        sharp::gpu::ConvectiveBatchOutputs thermo_only;
        float* d_pwat = nullptr;
        float* d_lr03 = nullptr;
        float* d_fz = nullptr;
        CUDA_CHECK(cudaMalloc(&d_pwat, out_bytes));
        CUDA_CHECK(cudaMalloc(&d_lr03, out_bytes));
        CUDA_CHECK(cudaMalloc(&d_fz, out_bytes));

        thermo_only.pwat = d_pwat;
        thermo_only.lr03 = d_lr03;
        thermo_only.fz_level_agl = d_fz;

        sharp::gpu::launch_convective_batch_flexible(
            d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
            thermo_only,
            n_levels, n_profiles,
            true, true
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> pwat(n_profiles), lr03(n_profiles), fz(n_profiles);
        CUDA_CHECK(cudaMemcpy(pwat.data(), d_pwat, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lr03.data(), d_lr03, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(fz.data(), d_fz, out_bytes, cudaMemcpyDeviceToHost));

        for (int p = 0; p < n_profiles; ++p) {
            CHECK(pwat[p] > 10.0f);
            CHECK(lr03[p] > 4.0f);
            CHECK(fz[p] > 2000.0f);
        }

        cudaFree(d_pwat);
        cudaFree(d_lr03);
        cudaFree(d_fz);
    }

    SUBCASE("Selective execution pruning - Kinematics only") {
        sharp::gpu::ConvectiveBatchOutputs kin_only;
        float* d_srh01 = nullptr;
        float* d_shr06 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_srh01, out_bytes));
        CUDA_CHECK(cudaMalloc(&d_shr06, out_bytes));

        kin_only.srh01 = d_srh01;
        kin_only.shr06 = d_shr06;

        sharp::gpu::launch_convective_batch_flexible(
            d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
            kin_only,
            n_levels, n_profiles,
            true, true
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> srh01(n_profiles), shr06(n_profiles);
        CUDA_CHECK(cudaMemcpy(srh01.data(), d_srh01, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(shr06.data(), d_shr06, out_bytes, cudaMemcpyDeviceToHost));

        for (int p = 0; p < n_profiles; ++p) {
            CHECK(srh01[p] > 0.0f);
            CHECK(shr06[p] > 10.0f);
        }

        cudaFree(d_srh01);
        cudaFree(d_shr06);
    }

    CUDA_CHECK(cudaFree(d_pres));
    CUDA_CHECK(cudaFree(d_hght));
    CUDA_CHECK(cudaFree(d_tmpk));
    CUDA_CHECK(cudaFree(d_dwpk));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));
}

