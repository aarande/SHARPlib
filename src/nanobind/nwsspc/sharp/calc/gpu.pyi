from typing import Annotated

import numpy
from numpy.typing import NDArray


def convective_batch(pres: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], hght: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], tmpk: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], dwpk: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], u_wind: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], v_wind: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cuda', writable=False)], out_sbcape: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_sbcin: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_mlcape: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_mlcin: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_mucape: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_mucin: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_srh01: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_srh03: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_shr06: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], out_stp: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cuda')], use_texture_lut: bool = True, fast_mu_thetae: bool = True, cuda_stream_ptr: int = 0) -> None:
    """
    Compute convective and kinematic sounding parameters across batched 2D GPU grids.
    """
