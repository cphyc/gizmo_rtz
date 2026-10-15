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
    uv_g0=0.0070977,
    constant_temperature=False,
    include_H2=False,
    include_CO=False,
    include_self_shielding=False,
    fixed_radiation=False,
    rt_advect=True,
    equilibrium=False,
    conv_its=20000,
    T_blackbody=1.0e4,
    temperature_is_Tmu=False,
    use_default_depletion=True,
    include_collisional_ionization=True,
    include_photoionization=True,
    include_cosmic_ray_ionization=True,
    include_HM12_UVB=True,
    include_dust_recombination=True,
    include_charge_exchange=True,
):
    """Evolve a 2D float64 state array at each 1D float64 time in seconds.

    Physics options select compile-time solver variants. ``fixed_radiation``
    keeps photon densities constant without disabling their effect on chemistry.
    ``equilibrium`` evolves each requested interval until the solver's
    convergence counter exceeds ``conv_its``. ``rt_advect`` and
    the chemistry options below reproduce the corresponding C++ template and
    runtime configurations.
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
        uv_g0,
        constant_temperature,
        include_H2,
        include_CO,
        include_self_shielding,
        fixed_radiation,
        rt_advect,
        equilibrium,
        conv_its,
        T_blackbody,
        temperature_is_Tmu,
        use_default_depletion,
        include_collisional_ionization,
        include_photoionization,
        include_cosmic_ray_ionization,
        include_HM12_UVB,
        include_dust_recombination,
        include_charge_exchange,
    )


__all__ = ["solve"]
