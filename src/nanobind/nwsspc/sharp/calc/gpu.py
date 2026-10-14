"""
GPU-accelerated batch sounding analysis routines.
"""

from typing import Any, Iterable, Optional, Sequence, Union, Dict

try:
    from nwsspc.sharp.calc.gpu import (
        convective_batch,
        convective_batch_flexible,
        SUPPORTED_PARAMS,
    )
except ImportError:
    SUPPORTED_PARAMS = (
        "sbcape", "sbcin", "sblcl_hght_agl", "sblcl_pres", "sblfc_hght_agl", "sblfc_pres", "sbel_hght_agl", "sbel_pres", "sb3cape",
        "mlcape", "mlcin", "mllcl_hght_agl", "mllcl_pres", "mllfc_hght_agl", "mllfc_pres", "mlel_hght_agl", "mlel_pres", "ml3cape",
        "mucape", "mucin", "mulcl_hght_agl", "mulcl_pres", "mulfc_hght_agl", "mulfc_pres", "muel_hght_agl", "muel_pres", "mu3cape",
        "bunkers_rm_u", "bunkers_rm_v", "bunkers_lm_u", "bunkers_lm_v", "mean_wind_u", "mean_wind_v",
        "shr01", "shr03", "shr06", "srh01", "srh03", "srh05", "crit_angle",
        "pwat", "lr03", "lr36", "lr75", "fz_level_agl", "wbz_level_agl",
        "stp", "scp", "ehi01", "ehi03",
    )
    convective_batch = None
    convective_batch_flexible = None


def compute_sounding_grid(
    pres,
    hght,
    tmpk,
    dwpk,
    u_wind,
    v_wind,
    params: Union[str, Iterable[str], None] = "all",
    use_texture_lut: bool = True,
    fast_mu_thetae: bool = True,
    stream: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compute convective, kinematic, thermodynamic, and composite parameters across
    a batched 2D grid of vertical atmospheric soundings on the GPU.

    Parameters
    ----------
    pres : CuPy ndarray or PyTorch CUDA Tensor of shape [N_levels, N_profiles]
        Pressure grid (Pa). Must be C-contiguous.
    hght : CuPy ndarray or PyTorch CUDA Tensor of shape [N_levels, N_profiles]
        Height grid (m MSL). Must be C-contiguous.
    tmpk : CuPy ndarray or PyTorch CUDA Tensor of shape [N_levels, N_profiles]
        Temperature grid (K). Must be C-contiguous.
    dwpk : CuPy ndarray or PyTorch CUDA Tensor of shape [N_levels, N_profiles]
        Dewpoint temperature grid (K). Must be C-contiguous.
    u_wind : CuPy ndarray or PyTorch CUDA Tensor of shape [N_levels, N_profiles]
        U-wind component (m/s). Must be C-contiguous.
    v_wind : CuPy ndarray or PyTorch CUDA Tensor of shape [N_levels, N_profiles]
        V-wind component (m/s). Must be C-contiguous.
    params : str or Iterable[str] or None, default="all"
        Parameters to compute. If "all" or None, all supported parameters are computed.
        Otherwise, an iterable of parameter names selected from SUPPORTED_PARAMS.
    use_texture_lut : bool, default=True
        If True, use hardware 2D texture cache for moist pseudoadiabatic ascent.
        If False, use exact Wobus polynomial lifter.
    fast_mu_thetae : bool, default=True
        If True, choose most unstable parcel by maximum theta-e in lowest 300 hPa.
        If False, lift parcels from all levels in lowest 300 hPa.
    stream : CuPy Stream or PyTorch CUDA Stream or int, optional
        CUDA stream to use for asynchronous kernel execution.

    Returns
    -------
    dict[str, Array/Tensor]
        Dictionary mapping requested parameter name strings to 1D device arrays
        of shape [N_profiles]. The returned arrays match the input tensor framework
        (CuPy or PyTorch).
    """
    supported_set = set(SUPPORTED_PARAMS)
    if params == "all" or params is None:
        target_params = list(SUPPORTED_PARAMS)
    elif isinstance(params, str):
        if params not in supported_set:
            raise ValueError(f"Unknown parameter name: {params!r}. Must be one of {SUPPORTED_PARAMS}")
        target_params = [params]
    else:
        target_params = list(params)
        for p in target_params:
            if p not in supported_set:
                raise ValueError(f"Unknown parameter name: {p!r}. Must be one of {SUPPORTED_PARAMS}")

    if not hasattr(pres, "shape") or len(pres.shape) != 2:
        raise ValueError("Input 'pres' must be a 2D grid of shape [N_levels, N_profiles]")

    n_levels, n_profiles = pres.shape

    # Framework auto-detection
    is_torch = False
    if hasattr(pres, "is_cuda") and pres.is_cuda:
        is_torch = True
    else:
        try:
            import torch
            if torch.is_tensor(pres) and pres.is_cuda:
                is_torch = True
        except ImportError:
            pass

    is_cupy = False
    if not is_torch:
        if hasattr(pres, "__cuda_array_interface__"):
            is_cupy = True
        else:
            try:
                import cupy as cp
                if isinstance(pres, cp.ndarray):
                    is_cupy = True
            except ImportError:
                pass

    if not is_torch and not is_cupy:
        raise TypeError(
            "Inputs must be either CuPy ndarrays or PyTorch CUDA tensors."
        )

    output_dict = {}
    if is_torch:
        import torch
        for name in target_params:
            output_dict[name] = torch.empty(n_profiles, dtype=torch.float32, device=pres.device)
    else:
        import cupy as cp
        for name in target_params:
            output_dict[name] = cp.empty(n_profiles, dtype=cp.float32)

    if stream is None:
        stream_ptr = 0
    elif hasattr(stream, "ptr"):
        stream_ptr = int(stream.ptr)
    elif hasattr(stream, "cuda_stream"):
        stream_ptr = int(stream.cuda_stream)
    elif isinstance(stream, int):
        stream_ptr = stream
    else:
        stream_ptr = 0

    if convective_batch_flexible is None:
        from nwsspc.sharp.calc.gpu import convective_batch_flexible as _cbf
        _cbf(
            pres, hght, tmpk, dwpk, u_wind, v_wind,
            output_dict,
            use_texture_lut=use_texture_lut,
            fast_mu_thetae=fast_mu_thetae,
            cuda_stream_ptr=stream_ptr
        )
    else:
        convective_batch_flexible(
            pres, hght, tmpk, dwpk, u_wind, v_wind,
            output_dict,
            use_texture_lut=use_texture_lut,
            fast_mu_thetae=fast_mu_thetae,
            cuda_stream_ptr=stream_ptr
        )
    return output_dict


__all__ = [
    "convective_batch",
    "convective_batch_flexible",
    "compute_sounding_grid",
    "SUPPORTED_PARAMS",
]
