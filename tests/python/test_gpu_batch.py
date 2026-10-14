"""
Tests for GPU-accelerated batch sounding routines using nanobind DLPack integration.
Validates numerical accuracy against 1D CPU SHARPlib and benchmarks full-grid throughput.
"""

import math
import pytest
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from nwsspc.sharp.calc import layer, parcel, thermo, winds, params, constants

try:
    from nwsspc.sharp.calc import gpu, compute_sounding_grid
    HAS_GPU = hasattr(gpu, "convective_batch")
except (ImportError, AttributeError):
    HAS_GPU = False


pytestmark = pytest.mark.skipif(
    not (HAS_CUPY and HAS_GPU),
    reason="CUDA/CuPy or SHARPlib GPU module not available"
)


def make_sounding(stype: str, n_levels: int = 50):
    """
    Generate synthetic sounding profiles for diverse atmospheric regimes.
    """
    z = np.linspace(100.0, 16000.0, n_levels, dtype=np.float32)
    z_agl = z - z[0]
    pres = (100000.0 * np.exp(-z_agl / 8000.0)).astype(np.float32)

    if stype == "dry_capped":
        t_sfc = 300.0
        td_sfc = 275.0
        tmpk = np.zeros(n_levels, dtype=np.float32)
        dwpk = np.zeros(n_levels, dtype=np.float32)
        for k in range(n_levels):
            h = z_agl[k]
            if h < 1000.0:
                tmpk[k] = t_sfc - 0.0098 * h
            elif h < 2000.0:
                tmpk[k] = (t_sfc - 0.0098 * 1000.0) + 0.004 * (h - 1000.0)
            elif h < 11000.0:
                tmpk[k] = (t_sfc - 0.0098 * 1000.0 + 0.004 * 1000.0) - 0.0065 * (h - 2000.0)
            else:
                tmpk[k] = tmpk[k - 1]
            dwpk[k] = max(180.0, td_sfc - 0.008 * h)
        u = np.full(n_levels, 2.0, dtype=np.float32)
        v = np.full(n_levels, 3.0, dtype=np.float32)

    elif stype == "moderate":
        t_sfc = 298.0
        td_sfc = 290.0
        tmpk = np.where(
            z_agl < 11000.0,
            t_sfc - 0.0065 * z_agl,
            t_sfc - 0.0065 * 11000.0
        ).astype(np.float32)
        dwpk = np.where(
            z_agl < 2000.0,
            td_sfc - 0.002 * z_agl,
            td_sfc - 0.002 * 2000.0 - 0.007 * (z_agl - 2000.0)
        ).astype(np.float32)
        dwpk = np.maximum(dwpk, 180.0)
        u = (-3.0 + 18.0 * (z_agl / 12000.0)).astype(np.float32)
        v = (4.0 + 12.0 * (z_agl / 12000.0)).astype(np.float32)

    elif stype == "supercell":
        t_sfc = 305.0
        td_sfc = 297.0
        tmpk = np.where(
            z_agl < 11000.0,
            t_sfc - 0.0075 * z_agl,
            t_sfc - 0.0075 * 11000.0
        ).astype(np.float32)
        dwpk = np.where(
            z_agl < 1500.0,
            td_sfc - 0.001 * z_agl,
            td_sfc - 0.001 * 1500.0 - 0.006 * (z_agl - 1500.0)
        ).astype(np.float32)
        dwpk = np.maximum(dwpk, 180.0)
        u = (-8.0 + 38.0 * (z_agl / 12000.0)).astype(np.float32)
        v = (8.0 + 25.0 * (z_agl / 12000.0)).astype(np.float32)

    elif stype == "tropical":
        t_sfc = 301.0
        td_sfc = 299.0
        tmpk = np.where(
            z_agl < 12000.0,
            t_sfc - 0.0055 * z_agl,
            t_sfc - 0.0055 * 12000.0
        ).astype(np.float32)
        dwpk = np.where(
            z_agl < 4000.0,
            td_sfc - 0.001 * z_agl,
            td_sfc - 0.001 * 4000.0 - 0.005 * (z_agl - 4000.0)
        ).astype(np.float32)
        dwpk = np.maximum(dwpk, 180.0)
        u = (-5.0 + 8.0 * (z_agl / 12000.0)).astype(np.float32)
        v = (2.0 + 6.0 * (z_agl / 12000.0)).astype(np.float32)

    else:
        raise ValueError(f"Unknown sounding type: {stype}")

    return pres, z, tmpk, dwpk, u, v


def compute_cpu_reference_1d(pres, hght, tmpk, dwpk, u, v):
    """
    Compute reference sounding parameters using 1D CPU SHARPlib algorithms.
    """
    mixr = thermo.mixratio(pres, dwpk)
    vtmp = thermo.virtual_temperature(tmpk, mixr)
    lifter = parcel.lifter_wobus()

    # 1. Surface-Based Parcel
    sb_pcl = parcel.Parcel.surface_parcel(pres[0], tmpk[0], dwpk[0])
    sb_vtmpk = sb_pcl.lift_parcel(lifter, pres)
    sb_buoy = thermo.buoyancy(sb_vtmpk, vtmp)
    sb_cape, sb_cin = sb_pcl.cape_cinh(pres, hght, sb_buoy)

    # 2. 100-hPa Mixed-Layer Parcel
    mix_lyr = layer.PressureLayer(pres[0], pres[0] - 10000.0)
    theta = thermo.theta(pres, tmpk)
    ml_pcl = parcel.Parcel.mixed_layer_parcel(mix_lyr, pres, theta, mixr)
    ml_vtmpk = ml_pcl.lift_parcel(lifter, pres)
    ml_buoy = thermo.buoyancy(ml_vtmpk, vtmp)
    ml_cape, ml_cin = ml_pcl.cape_cinh(pres, hght, ml_buoy)

    # 3. Most-Unstable Parcel (lowest 300 hPa)
    search_layer = layer.PressureLayer(pres[0], pres[0] - 30000.0)
    mu_pcl = parcel.Parcel.most_unstable_parcel(
        search_layer, lifter, pres, hght, tmpk, vtmp, dwpk
    )
    mu_cape = mu_pcl.cape
    mu_cin = mu_pcl.cinh

    # 4. Kinematics
    shr_lyr = layer.HeightLayer(0.0, 6000.0)
    mw_lyr = layer.HeightLayer(0.0, 6000.0)
    sm = params.storm_motion_bunkers(pres, hght, u, v, mw_lyr, shr_lyr)

    lyr01 = layer.HeightLayer(0.0, 1000.0)
    srh01 = winds.helicity(lyr01, sm, hght, u, v)

    lyr03 = layer.HeightLayer(0.0, 3000.0)
    srh03 = winds.helicity(lyr03, sm, hght, u, v)

    lyr06 = layer.HeightLayer(0.0, 6000.0)
    shr06_cmp = winds.wind_shear(lyr06, hght, u, v)
    shr06 = winds.vector_magnitude(shr06_cmp.u, shr06_cmp.v)

    return {
        "sbcape": sb_cape,
        "sbcin": sb_cin,
        "mlcape": ml_cape,
        "mlcin": ml_cin,
        "mucape": mu_cape,
        "mucin": mu_cin,
        "srh01": srh01,
        "srh03": srh03,
        "shr06": shr06,
    }


def test_cupy_dlpack_integration():
    """
    Test zero-copy DLPack integration with CuPy device arrays and custom stream.
    """
    n_levels = 50
    n_profiles = 8

    p_1d, z_1d, t_1d, d_1d, u_1d, v_1d = make_sounding("moderate", n_levels)

    pres = cp.tile(cp.asarray(p_1d)[:, None], (1, n_profiles))
    hght = cp.tile(cp.asarray(z_1d)[:, None], (1, n_profiles))
    tmpk = cp.tile(cp.asarray(t_1d)[:, None], (1, n_profiles))
    dwpk = cp.tile(cp.asarray(d_1d)[:, None], (1, n_profiles))
    u = cp.tile(cp.asarray(u_1d)[:, None], (1, n_profiles))
    v = cp.tile(cp.asarray(v_1d)[:, None], (1, n_profiles))

    out_sbcape = cp.zeros(n_profiles, dtype=cp.float32)
    out_sbcin  = cp.zeros(n_profiles, dtype=cp.float32)
    out_mlcape = cp.zeros(n_profiles, dtype=cp.float32)
    out_mlcin  = cp.zeros(n_profiles, dtype=cp.float32)
    out_mucape = cp.zeros(n_profiles, dtype=cp.float32)
    out_mucin  = cp.zeros(n_profiles, dtype=cp.float32)
    out_srh01  = cp.zeros(n_profiles, dtype=cp.float32)
    out_srh03  = cp.zeros(n_profiles, dtype=cp.float32)
    out_shr06  = cp.zeros(n_profiles, dtype=cp.float32)
    out_stp    = cp.zeros(n_profiles, dtype=cp.float32)

    stream = cp.cuda.Stream()
    gpu.convective_batch(
        pres, hght, tmpk, dwpk, u, v,
        out_sbcape, out_sbcin, out_mlcape, out_mlcin,
        out_mucape, out_mucin, out_srh01, out_srh03,
        out_shr06, out_stp,
        use_texture_lut=True,
        fast_mu_thetae=True,
        cuda_stream_ptr=stream.ptr
    )
    stream.synchronize()

    assert cp.all(out_sbcape > 0.0)
    assert cp.all(out_mlcape > 0.0)
    assert cp.all(out_mucape >= out_sbcape)
    assert cp.all(out_shr06 > 5.0)


def test_pytorch_dlpack_integration():
    """
    Test zero-copy DLPack integration with PyTorch CUDA tensors if available.
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        pytest.skip("PyTorch with CUDA support is not available")

    n_levels = 50
    n_profiles = 4

    p_1d, z_1d, t_1d, d_1d, u_1d, v_1d = make_sounding("moderate", n_levels)

    pres = torch.from_numpy(p_1d).unsqueeze(1).expand(-1, n_profiles).cuda()
    hght = torch.from_numpy(z_1d).unsqueeze(1).expand(-1, n_profiles).cuda()
    tmpk = torch.from_numpy(t_1d).unsqueeze(1).expand(-1, n_profiles).cuda()
    dwpk = torch.from_numpy(d_1d).unsqueeze(1).expand(-1, n_profiles).cuda()
    u = torch.from_numpy(u_1d).unsqueeze(1).expand(-1, n_profiles).cuda()
    v = torch.from_numpy(v_1d).unsqueeze(1).expand(-1, n_profiles).cuda()

    out_sbcape = torch.zeros(n_profiles, dtype=torch.float32, device="cuda")
    out_sbcin  = torch.zeros(n_profiles, dtype=torch.float32, device="cuda")
    out_mlcape = torch.zeros(n_profiles, dtype=torch.float32, device="cuda")
    out_mlcin  = torch.zeros(n_profiles, dtype=torch.float32, device="cuda")
    out_mucape = torch.zeros(n_profiles, dtype=torch.float32, device="cuda")
    out_mucin  = torch.zeros(n_profiles, dtype=torch.float32, device="cuda")
    out_srh01  = torch.zeros(n_profiles, dtype=torch.float32, device="cuda")
    out_srh03  = torch.zeros(n_profiles, dtype=torch.float32, device="cuda")
    out_shr06  = torch.zeros(n_profiles, dtype=torch.float32, device="cuda")
    out_stp    = torch.zeros(n_profiles, dtype=torch.float32, device="cuda")

    stream = torch.cuda.current_stream()
    gpu.convective_batch(
        pres, hght, tmpk, dwpk, u, v,
        out_sbcape, out_sbcin, out_mlcape, out_mlcin,
        out_mucape, out_mucin, out_srh01, out_srh03,
        out_shr06, out_stp,
        use_texture_lut=True,
        fast_mu_thetae=True,
        cuda_stream_ptr=stream.cuda_stream
    )
    torch.cuda.synchronize()

    assert torch.all(out_sbcape > 0.0)
    assert torch.all(out_mucape >= out_sbcape)


def test_exact_wobus_mode():
    """
    Test 1: Exact Wobus Mode (use_texture_lut=False, fast_mu_thetae=False).
    Validates numerical equivalence against 1D CPU SHARPlib:
      - SBCAPE, MLCAPE, MUCAPE match within <= 1.0 J/kg or 0.1% relative error.
      - SRH and Bulk Shear match within <= 1e-3.
    """
    sounding_types = ["dry_capped", "moderate", "supercell", "tropical"]
    n_levels = 50
    n_profiles = len(sounding_types)

    h_pres = np.empty((n_levels, n_profiles), dtype=np.float32)
    h_hght = np.empty((n_levels, n_profiles), dtype=np.float32)
    h_tmpk = np.empty((n_levels, n_profiles), dtype=np.float32)
    h_dwpk = np.empty((n_levels, n_profiles), dtype=np.float32)
    h_u = np.empty((n_levels, n_profiles), dtype=np.float32)
    h_v = np.empty((n_levels, n_profiles), dtype=np.float32)

    cpu_results = []
    for i, stype in enumerate(sounding_types):
        p, z, t, d, u, v = make_sounding(stype, n_levels)
        h_pres[:, i] = p
        h_hght[:, i] = z
        h_tmpk[:, i] = t
        h_dwpk[:, i] = d
        h_u[:, i] = u
        h_v[:, i] = v
        cpu_results.append(compute_cpu_reference_1d(p, z, t, d, u, v))

    d_pres = cp.asarray(h_pres)
    d_hght = cp.asarray(h_hght)
    d_tmpk = cp.asarray(h_tmpk)
    d_dwpk = cp.asarray(h_dwpk)
    d_u = cp.asarray(h_u)
    d_v = cp.asarray(h_v)

    out_sbcape = cp.zeros(n_profiles, dtype=cp.float32)
    out_sbcin  = cp.zeros(n_profiles, dtype=cp.float32)
    out_mlcape = cp.zeros(n_profiles, dtype=cp.float32)
    out_mlcin  = cp.zeros(n_profiles, dtype=cp.float32)
    out_mucape = cp.zeros(n_profiles, dtype=cp.float32)
    out_mucin  = cp.zeros(n_profiles, dtype=cp.float32)
    out_srh01  = cp.zeros(n_profiles, dtype=cp.float32)
    out_srh03  = cp.zeros(n_profiles, dtype=cp.float32)
    out_shr06  = cp.zeros(n_profiles, dtype=cp.float32)
    out_stp    = cp.zeros(n_profiles, dtype=cp.float32)

    gpu.convective_batch(
        d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
        out_sbcape, out_sbcin, out_mlcape, out_mlcin,
        out_mucape, out_mucin, out_srh01, out_srh03,
        out_shr06, out_stp,
        use_texture_lut=False,
        fast_mu_thetae=False
    )
    cp.cuda.Stream.null.synchronize()

    gpu_sbcape = out_sbcape.get()
    gpu_mlcape = out_mlcape.get()
    gpu_mucape = out_mucape.get()
    gpu_srh01  = out_srh01.get()
    gpu_srh03  = out_srh03.get()
    gpu_shr06  = out_shr06.get()

    for i, stype in enumerate(sounding_types):
        ref = cpu_results[i]

        # CAPE validation: <= 1.0 J/kg absolute or 0.1% relative error
        for name, val_gpu, val_cpu in [
            ("SBCAPE", gpu_sbcape[i], ref["sbcape"]),
            ("MLCAPE", gpu_mlcape[i], ref["mlcape"]),
            ("MUCAPE", gpu_mucape[i], ref["mucape"]),
        ]:
            if math.isnan(val_cpu) or val_cpu <= 0.0:
                assert val_gpu <= 1.0, f"{stype} {name}: expected ~0, got {val_gpu}"
            else:
                abs_err = abs(val_gpu - val_cpu)
                rel_err = abs_err / val_cpu
                assert (abs_err <= 1.0 or rel_err <= 0.001), (
                    f"{stype} {name} mismatch: GPU={val_gpu:.3f}, CPU={val_cpu:.3f}, "
                    f"abs_err={abs_err:.3f}, rel_err={rel_err*100:.3f}%"
                )

        # Kinematics validation: <= 1e-3
        for name, val_gpu, val_cpu in [
            ("SRH01", gpu_srh01[i], ref["srh01"]),
            ("SRH03", gpu_srh03[i], ref["srh03"]),
            ("SHR06", gpu_shr06[i], ref["shr06"]),
        ]:
            if not math.isnan(val_cpu):
                assert abs(val_gpu - val_cpu) <= 1e-3, (
                    f"{stype} {name} mismatch: GPU={val_gpu:.4f}, CPU={val_cpu:.4f}"
                )


def test_fast_texture_lut_mode():
    """
    Test 2: Fast Texture LUT Mode (use_texture_lut=True).
    Validates that hardware texture bilinear LUT CAPE matches within <= 1.5% of exact lifter.
    """
    sounding_types = ["moderate", "supercell", "tropical"]
    n_levels = 50
    n_profiles = len(sounding_types)

    h_pres = np.empty((n_levels, n_profiles), dtype=np.float32)
    h_hght = np.empty((n_levels, n_profiles), dtype=np.float32)
    h_tmpk = np.empty((n_levels, n_profiles), dtype=np.float32)
    h_dwpk = np.empty((n_levels, n_profiles), dtype=np.float32)
    h_u = np.zeros((n_levels, n_profiles), dtype=np.float32)
    h_v = np.zeros((n_levels, n_profiles), dtype=np.float32)

    for i, stype in enumerate(sounding_types):
        p, z, t, d, u, v = make_sounding(stype, n_levels)
        h_pres[:, i] = p
        h_hght[:, i] = z
        h_tmpk[:, i] = t
        h_dwpk[:, i] = d
        h_u[:, i] = u
        h_v[:, i] = v

    d_pres = cp.asarray(h_pres)
    d_hght = cp.asarray(h_hght)
    d_tmpk = cp.asarray(h_tmpk)
    d_dwpk = cp.asarray(h_dwpk)
    d_u = cp.asarray(h_u)
    d_v = cp.asarray(h_v)

    out_exact = [cp.zeros(n_profiles, dtype=cp.float32) for _ in range(10)]
    out_lut   = [cp.zeros(n_profiles, dtype=cp.float32) for _ in range(10)]

    gpu.convective_batch(
        d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
        *out_exact,
        use_texture_lut=False,
        fast_mu_thetae=False
    )

    gpu.convective_batch(
        d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
        *out_lut,
        use_texture_lut=True,
        fast_mu_thetae=False
    )
    cp.cuda.Stream.null.synchronize()

    exact_sbcape = out_exact[0].get()
    lut_sbcape   = out_lut[0].get()

    exact_mlcape = out_exact[2].get()
    lut_mlcape   = out_lut[2].get()

    for i, stype in enumerate(sounding_types):
        # SBCAPE
        rel_err_sb = abs(lut_sbcape[i] - exact_sbcape[i]) / exact_sbcape[i]
        assert rel_err_sb <= 0.015, (
            f"{stype} SBCAPE LUT error {rel_err_sb*100:.3f}% exceeds 1.5% limit! "
            f"Exact={exact_sbcape[i]:.1f}, LUT={lut_sbcape[i]:.1f}"
        )

        # MLCAPE
        rel_err_ml = abs(lut_mlcape[i] - exact_mlcape[i]) / exact_mlcape[i]
        assert rel_err_ml <= 0.015, (
            f"{stype} MLCAPE LUT error {rel_err_ml*100:.3f}% exceeds 1.5% limit! "
            f"Exact={exact_mlcape[i]:.1f}, LUT={lut_mlcape[i]:.1f}"
        )


def test_hrrr_full_grid_benchmark():
    """
    Test 3: Full-Grid Throughput Benchmark.
    Simulates a full CONUS HRRR grid:
      N_levels = 50, N_profiles = 1,905,141 (~1799 x 1059 grid).
    Verifies execution time < 50 ms on RTX 4070 Ti SUPER and logs throughput.
    """
    n_levels = 50
    n_profiles = 1905141

    z = np.linspace(100.0, 16000.0, n_levels, dtype=np.float32)
    z_agl = z - z[0]
    p_1d = (100000.0 * np.exp(-z_agl / 8000.0)).astype(np.float32)
    t_1d = np.where(z_agl < 11000.0, 303.0 - 0.007 * z_agl, 303.0 - 0.007 * 11000.0).astype(np.float32)
    td_1d = np.where(z_agl < 1500.0, 296.0 - 0.001 * z_agl, 296.0 - 0.001 * 1500.0 - 0.006 * (z_agl - 1500.0)).astype(np.float32)
    u_1d = (-8.0 + 35.0 * (z_agl / 12000.0)).astype(np.float32)
    v_1d = (8.0 + 25.0 * (z_agl / 12000.0)).astype(np.float32)

    d_pres = cp.tile(cp.asarray(p_1d)[:, None], (1, n_profiles))
    d_hght = cp.tile(cp.asarray(z)[:, None], (1, n_profiles))
    d_tmpk = cp.tile(cp.asarray(t_1d)[:, None], (1, n_profiles))
    d_dwpk = cp.tile(cp.asarray(td_1d)[:, None], (1, n_profiles))
    d_u = cp.tile(cp.asarray(u_1d)[:, None], (1, n_profiles))
    d_v = cp.tile(cp.asarray(v_1d)[:, None], (1, n_profiles))

    out_sbcape = cp.empty(n_profiles, dtype=cp.float32)
    out_sbcin  = cp.empty(n_profiles, dtype=cp.float32)
    out_mlcape = cp.empty(n_profiles, dtype=cp.float32)
    out_mlcin  = cp.empty(n_profiles, dtype=cp.float32)
    out_mucape = cp.empty(n_profiles, dtype=cp.float32)
    out_mucin  = cp.empty(n_profiles, dtype=cp.float32)
    out_srh01  = cp.empty(n_profiles, dtype=cp.float32)
    out_srh03  = cp.empty(n_profiles, dtype=cp.float32)
    out_shr06  = cp.empty(n_profiles, dtype=cp.float32)
    out_stp    = cp.empty(n_profiles, dtype=cp.float32)

    # Warmup run
    gpu.convective_batch(
        d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
        out_sbcape, out_sbcin, out_mlcape, out_mlcin,
        out_mucape, out_mucin, out_srh01, out_srh03,
        out_shr06, out_stp,
        use_texture_lut=True,
        fast_mu_thetae=True
    )
    cp.cuda.Stream.null.synchronize()

    # Timed benchmark run using CUDA events
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    start_event.record()
    gpu.convective_batch(
        d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v,
        out_sbcape, out_sbcin, out_mlcape, out_mlcin,
        out_mucape, out_mucin, out_srh01, out_srh03,
        out_shr06, out_stp,
        use_texture_lut=True,
        fast_mu_thetae=True
    )
    end_event.record()
    end_event.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    throughput = n_profiles / (elapsed_ms / 1000.0)

    print(f"\n[HRRR Full-Grid Benchmark] Profiles: {n_profiles:,} | Levels: {n_levels}")
    print(f"Elapsed Time: {elapsed_ms:.2f} ms | Throughput: {throughput:,.0f} profiles/sec")

    assert elapsed_ms < 50.0, f"Execution time {elapsed_ms:.2f} ms exceeded 50 ms limit!"


def test_dimension_validation():
    """
    Test that mismatched tensor shapes raise BufferError.
    """
    pres = cp.zeros((50, 4), dtype=cp.float32)
    hght = cp.zeros((40, 4), dtype=cp.float32)  # mismatched n_levels
    tmpk = cp.zeros((50, 4), dtype=cp.float32)
    dwpk = cp.zeros((50, 4), dtype=cp.float32)
    u = cp.zeros((50, 4), dtype=cp.float32)
    v = cp.zeros((50, 4), dtype=cp.float32)

    outputs = [cp.zeros(4, dtype=cp.float32) for _ in range(10)]

    with pytest.raises(BufferError, match="All input 2D grids must have shape"):
        gpu.convective_batch(pres, hght, tmpk, dwpk, u, v, *outputs)

    # Mismatched output size
    hght_ok = cp.zeros((50, 4), dtype=cp.float32)
    outputs_bad = [cp.zeros(8, dtype=cp.float32) for _ in range(10)]  # size 8 vs 4
    with pytest.raises(BufferError, match="All output 1D arrays must have shape"):
        gpu.convective_batch(pres, hght_ok, tmpk, dwpk, u, v, *outputs_bad)


def test_compute_sounding_grid_all():
    """
    Test compute_sounding_grid with params='all' across diverse soundings.
    Verifies all supported parameters return finite, expected values.
    """
    sounding_types = ["dry_capped", "moderate", "supercell", "tropical"]
    n_levels = 50
    n_profiles = len(sounding_types)

    h_pres = np.empty((n_levels, n_profiles), dtype=np.float32)
    h_hght = np.empty((n_levels, n_profiles), dtype=np.float32)
    h_tmpk = np.empty((n_levels, n_profiles), dtype=np.float32)
    h_dwpk = np.empty((n_levels, n_profiles), dtype=np.float32)
    h_u = np.empty((n_levels, n_profiles), dtype=np.float32)
    h_v = np.empty((n_levels, n_profiles), dtype=np.float32)

    for i, stype in enumerate(sounding_types):
        p, z, t, d, u, v = make_sounding(stype, n_levels)
        h_pres[:, i] = p
        h_hght[:, i] = z
        h_tmpk[:, i] = t
        h_dwpk[:, i] = d
        h_u[:, i] = u
        h_v[:, i] = v

    d_pres = cp.asarray(h_pres)
    d_hght = cp.asarray(h_hght)
    d_tmpk = cp.asarray(h_tmpk)
    d_dwpk = cp.asarray(h_dwpk)
    d_u = cp.asarray(h_u)
    d_v = cp.asarray(h_v)

    res = compute_sounding_grid(d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v, params="all")

    # Verify all supported parameters are present
    assert len(res) == len(gpu.SUPPORTED_PARAMS)
    for p in gpu.SUPPORTED_PARAMS:
        assert p in res, f"Missing parameter {p} in compute_sounding_grid output"

    for name, arr in res.items():
        assert isinstance(arr, cp.ndarray)
        assert arr.shape == (n_profiles,)
        assert arr.dtype == cp.float32
        vals = arr.get()
        assert np.all(np.isfinite(vals)), f"Parameter {name} has non-finite values: {vals}"

    # Specific physical checks
    # Supercell sounding (index 2) should have substantial CAPE and shear
    assert res["sbcape"][2] > 2000.0
    assert res["mlcape"][2] > 2000.0
    assert res["mucape"][2] >= res["sbcape"][2]
    assert res["shr06"][2] > 15.0
    assert res["stp"][2] > 0.0
    assert res["scp"][2] > 0.0

    # Dry capped sounding (index 0) should have zero/negligible CAPE
    assert res["sbcape"][0] <= 1.0
    assert res["mlcape"][0] <= 1.0

    # PWAT should be positive for all soundings
    assert cp.all(res["pwat"] > 0.0)


def test_compute_sounding_grid_selective():
    """
    Test selective parameter computation with compute_sounding_grid.
    Only requested keys should be allocated and calculated.
    """
    n_levels = 50
    n_profiles = 4

    p_1d, z_1d, t_1d, d_1d, u_1d, v_1d = make_sounding("moderate", n_levels)

    pres = cp.tile(cp.asarray(p_1d)[:, None], (1, n_profiles))
    hght = cp.tile(cp.asarray(z_1d)[:, None], (1, n_profiles))
    tmpk = cp.tile(cp.asarray(t_1d)[:, None], (1, n_profiles))
    dwpk = cp.tile(cp.asarray(d_1d)[:, None], (1, n_profiles))
    u = cp.tile(cp.asarray(u_1d)[:, None], (1, n_profiles))
    v = cp.tile(cp.asarray(v_1d)[:, None], (1, n_profiles))

    # Test 1: Selective thermo/moisture parameters
    res_thermo = compute_sounding_grid(pres, hght, tmpk, dwpk, u, v, params=["sbcape", "pwat"])
    assert set(res_thermo.keys()) == {"sbcape", "pwat"}
    assert cp.all(res_thermo["sbcape"] > 0.0)
    assert cp.all(res_thermo["pwat"] > 0.0)

    # Test 2: Selective kinematics parameters
    res_kin = compute_sounding_grid(pres, hght, tmpk, dwpk, u, v, params=["srh01", "shr06", "bunkers_rm_u"])
    assert set(res_kin.keys()) == {"srh01", "shr06", "bunkers_rm_u"}
    assert cp.all(cp.isfinite(res_kin["srh01"]))
    assert cp.all(res_kin["shr06"] > 0.0)
    assert cp.all(cp.isfinite(res_kin["bunkers_rm_u"]))

    # Test 3: Single parameter as string
    res_single = compute_sounding_grid(pres, hght, tmpk, dwpk, u, v, params="sbcape")
    assert set(res_single.keys()) == {"sbcape"}
    assert cp.all(res_single["sbcape"] > 0.0)


def test_compute_sounding_grid_error_handling():
    """
    Test error handling for invalid parameter names, shapes, and types.
    """
    pres = cp.zeros((50, 4), dtype=cp.float32)
    hght = cp.zeros((50, 4), dtype=cp.float32)
    tmpk = cp.zeros((50, 4), dtype=cp.float32)
    dwpk = cp.zeros((50, 4), dtype=cp.float32)
    u = cp.zeros((50, 4), dtype=cp.float32)
    v = cp.zeros((50, 4), dtype=cp.float32)

    # Invalid parameter in list raises ValueError
    with pytest.raises(ValueError, match="Unknown parameter name"):
        compute_sounding_grid(pres, hght, tmpk, dwpk, u, v, params=["sbcape", "invalid_param"])

    # Invalid parameter string raises ValueError
    with pytest.raises(ValueError, match="Unknown parameter name"):
        compute_sounding_grid(pres, hght, tmpk, dwpk, u, v, params="not_a_param")

    # In C++ convective_batch_flexible directly: invalid name raises ValueError
    with pytest.raises(ValueError, match="Unknown parameter name"):
        gpu.convective_batch_flexible(pres, hght, tmpk, dwpk, u, v, {"fake_key": cp.zeros(4, dtype=cp.float32)})

    # In C++ convective_batch_flexible: output shape mismatch raises BufferError
    with pytest.raises(BufferError, match="must have shape"):
        gpu.convective_batch_flexible(pres, hght, tmpk, dwpk, u, v, {"sbcape": cp.zeros(8, dtype=cp.float32)})

    # 1D array input raises ValueError
    with pytest.raises(ValueError, match="must be a 2D grid"):
        compute_sounding_grid(cp.zeros(50, dtype=cp.float32), hght, tmpk, dwpk, u, v)

    # CPU numpy array raises TypeError
    with pytest.raises(TypeError, match="Inputs must be either CuPy ndarrays or PyTorch CUDA tensors"):
        compute_sounding_grid(
            np.zeros((50, 4), dtype=np.float32),
            hght, tmpk, dwpk, u, v
        )


def test_compute_sounding_grid_pytorch():
    """
    Test PyTorch DLPack integration with compute_sounding_grid if torch with CUDA is present.
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        pytest.skip("PyTorch with CUDA support is not available")

    n_levels = 50
    n_profiles = 4

    p_1d, z_1d, t_1d, d_1d, u_1d, v_1d = make_sounding("moderate", n_levels)

    pres = torch.from_numpy(p_1d).unsqueeze(1).expand(-1, n_profiles).cuda()
    hght = torch.from_numpy(z_1d).unsqueeze(1).expand(-1, n_profiles).cuda()
    tmpk = torch.from_numpy(t_1d).unsqueeze(1).expand(-1, n_profiles).cuda()
    dwpk = torch.from_numpy(d_1d).unsqueeze(1).expand(-1, n_profiles).cuda()
    u = torch.from_numpy(u_1d).unsqueeze(1).expand(-1, n_profiles).cuda()
    v = torch.from_numpy(v_1d).unsqueeze(1).expand(-1, n_profiles).cuda()

    stream = torch.cuda.current_stream()
    res = compute_sounding_grid(
        pres, hght, tmpk, dwpk, u, v,
        params=["sbcape", "pwat", "srh01"],
        stream=stream
    )
    torch.cuda.synchronize()

    assert set(res.keys()) == {"sbcape", "pwat", "srh01"}
    for name, tensor in res.items():
        assert torch.is_tensor(tensor)
        assert tensor.is_cuda
        assert tensor.shape == (n_profiles,)
        assert tensor.dtype == torch.float32
        assert torch.all(torch.isfinite(tensor))

    assert torch.all(res["sbcape"] > 0.0)
    assert torch.all(res["pwat"] > 0.0)


def test_compute_sounding_grid_hrrr_benchmark():
    """
    Benchmark compute_sounding_grid with params='all' on the 1.9M HRRR grid.
    Reports execution time and throughput for full-parameter computation.
    """
    n_levels = 50
    n_profiles = 1905141

    z = np.linspace(100.0, 16000.0, n_levels, dtype=np.float32)
    z_agl = z - z[0]
    p_1d = (100000.0 * np.exp(-z_agl / 8000.0)).astype(np.float32)
    t_1d = np.where(z_agl < 11000.0, 303.0 - 0.007 * z_agl, 303.0 - 0.007 * 11000.0).astype(np.float32)
    td_1d = np.where(z_agl < 1500.0, 296.0 - 0.001 * z_agl, 296.0 - 0.001 * 1500.0 - 0.006 * (z_agl - 1500.0)).astype(np.float32)
    u_1d = (-8.0 + 35.0 * (z_agl / 12000.0)).astype(np.float32)
    v_1d = (8.0 + 25.0 * (z_agl / 12000.0)).astype(np.float32)

    d_pres = cp.tile(cp.asarray(p_1d)[:, None], (1, n_profiles))
    d_hght = cp.tile(cp.asarray(z)[:, None], (1, n_profiles))
    d_tmpk = cp.tile(cp.asarray(t_1d)[:, None], (1, n_profiles))
    d_dwpk = cp.tile(cp.asarray(td_1d)[:, None], (1, n_profiles))
    d_u = cp.tile(cp.asarray(u_1d)[:, None], (1, n_profiles))
    d_v = cp.tile(cp.asarray(v_1d)[:, None], (1, n_profiles))

    # Warmup run
    res = compute_sounding_grid(d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v, params="all")
    cp.cuda.Stream.null.synchronize()

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    start_event.record()
    res = compute_sounding_grid(d_pres, d_hght, d_tmpk, d_dwpk, d_u, d_v, params="all")
    end_event.record()
    end_event.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    throughput = n_profiles / (elapsed_ms / 1000.0)

    print(f"\n[HRRR Full-Grid compute_sounding_grid Benchmark (50 parameters)]")
    print(f"Profiles: {n_profiles:,} | Levels: {n_levels}")
    print(f"Elapsed Time: {elapsed_ms:.2f} ms | Throughput: {throughput:,.0f} profiles/sec")

    assert len(res) == len(gpu.SUPPORTED_PARAMS)
    assert elapsed_ms < 60.0, f"Execution time {elapsed_ms:.2f} ms exceeded 60 ms limit!"

