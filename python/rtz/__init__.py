import os
import atexit
from pathlib import Path

import numpy as np

from ._rtz import _finalize, solve as _solve

_DATA_PATH = Path(__file__).with_name("data")
atexit.register(_finalize)


def solve(
    state,
    times,
    element_list,
    ion_list,
    photon_bands,
    *,
    data_path=None,
    max_iterations=100000,
    redshift=0.0,
    constant_temperature=False,
    include_h2=False,
    include_co=False,
    include_self_shielding=False,
    fixed_radiation=False,
):
    """Evolve a 2D float64 state array at each 1D float64 time in seconds.

    Physics options select compile-time solver variants. ``fixed_radiation``
    keeps photon densities constant without disabling their effect on chemistry.
    """
    if not isinstance(state, np.ndarray) or state.ndim != 2:
        raise TypeError("state must be a 2D NumPy array")
    if state.dtype != np.float64:
        raise TypeError("state must have dtype float64")
    if not isinstance(times, np.ndarray) or times.ndim != 1:
        raise TypeError("times must be a 1D NumPy array")
    if times.dtype != np.float64:
        raise TypeError("times must have dtype float64")

    selected_data_path = (
        Path(os.environ.get("RTZ_DATA_DIR", _DATA_PATH))
        if data_path is None
        else Path(data_path)
    )
    return _solve(
        state,
        times,
        element_list,
        ion_list,
        photon_bands,
        str(selected_data_path),
        max_iterations,
        redshift,
        constant_temperature,
        include_h2,
        include_co,
        include_self_shielding,
        fixed_radiation,
    )


__all__ = ["solve"]
