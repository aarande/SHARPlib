from typing import Annotated

import numpy
from numpy.typing import NDArray


SUPPORTED_PARAMS: tuple = ...

def convective_batch(pres: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], hght: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], tmpk: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], dwpk: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], u_wind: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], v_wind: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], out_sbcape: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_sbcin: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_mlcape: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_mlcin: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_mucape: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_mucin: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_srh01: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_srh03: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_shr06: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_stp: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], use_texture_lut: bool = True, fast_mu_thetae: bool = True, cuda_stream_ptr: int = 0) -> None:
    """
    Compute convective and kinematic sounding parameters across batched 2D GPU grids.
    """

def convective_batch_flexible(pres: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], hght: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], tmpk: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], dwpk: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], u_wind: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], v_wind: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], outputs: dict, use_texture_lut: bool = True, fast_mu_thetae: bool = True, cuda_stream_ptr: int = 0) -> None:
    """
    Compute selected convective and kinematic sounding parameters across batched 2D GPU grids.
    """

def compute_sounding_grid(pres, hght, tmpk, dwpk, u_wind, v_wind, params='all', use_texture_lut=True, fast_mu_thetae=True, stream=None):
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
