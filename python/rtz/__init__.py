import os
import atexit
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import unyt as u

from ._rtz import _finalize, solve as _solve

_DATA_PATH = Path(__file__).with_name("data")
atexit.register(_finalize)

_F_NU_UNIT = "erg/(s*cm**2*Hz)"
_B_NU_UNIT = "erg/(s*cm**2*Hz*sr)"


def _quantity_values(value, unit, name, *, allow_plain=True):
    if hasattr(value, "to"):
        try:
            value = value.to(unit).value
        except Exception as exc:
            raise ValueError(f"{name} must be convertible to {unit}") from exc
    elif not allow_plain:
        raise TypeError(f"{name} must be a unitful array convertible to {unit}")
    return np.asarray(value, dtype=np.float64)


def _parse_state(state, element_list, ion_list, photon_bands):
    if isinstance(state, Mapping):
        required = ("temperature", "densities", "ion_fractions", "photon_densities")
        missing = [key for key in required if key not in state]
        extra = set(state) - set(required)
        if missing or extra:
            raise ValueError(
                f"grouped state requires {required}; missing={missing}, extra={sorted(extra)}"
            )

        temperature = _quantity_values(state["temperature"], "K", "temperature")
        if temperature.ndim == 0:
            temperature = temperature.reshape(1)
        elif temperature.ndim == 2 and temperature.shape[1] == 1:
            temperature = temperature[:, 0]
        if temperature.ndim != 1:
            raise ValueError("temperature must have one value per cell")

        groups = [
            temperature[:, None],
            _quantity_values(state["densities"], "cm**-3", "densities"),
            _quantity_values(state["ion_fractions"], "dimensionless", "ion_fractions"),
            _quantity_values(
                state["photon_densities"], "cm**-3", "photon_densities"
            ),
        ]
        n_cells = temperature.size
        expected_widths = (1, len(element_list), len(ion_list), len(photon_bands))
        normalized = [groups[0]]
        for name, group, width in zip(required[1:], groups[1:], expected_widths[1:]):
            if group.ndim == 1 and n_cells == 1:
                group = group.reshape(1, -1)
            if group.ndim != 2 or group.shape != (n_cells, width):
                raise ValueError(
                    f"{name} must have shape ({n_cells}, {width})"
                )
            normalized.append(group)
        if n_cells == 0:
            raise ValueError("state must contain at least one cell")
        return np.ascontiguousarray(np.column_stack(normalized), dtype=np.float64)

    if hasattr(state, "to") or not isinstance(state, np.ndarray):
        raise TypeError(
            "state must be a float64 NumPy matrix or a grouped-state mapping; "
            "use grouped arrays for unitful state data"
        )
    if state.ndim != 2:
        raise TypeError("state must be a 2D NumPy array")
    if state.dtype != np.float64:
        raise TypeError("state must have dtype float64")
    return state


def _parse_times(times):
    if hasattr(times, "to"):
        result = _quantity_values(times, "s", "times")
        if result.ndim != 1:
            raise ValueError("times must be a 1D array")
        return np.ascontiguousarray(result)
    if not isinstance(times, np.ndarray):
        raise TypeError("times must be a 1D NumPy or unyt array")
    if times.dtype != np.float64:
        raise TypeError("times must have dtype float64")
    return np.ascontiguousarray(times)


def _parse_photon_bands(photon_bands):
    if hasattr(photon_bands, "to"):
        bands = _quantity_values(photon_bands, "eV", "photon_bands")
        if bands.ndim != 2 or bands.shape[1] != 2:
            raise ValueError("photon_bands must have shape (number_of_bands, 2)")
        return [tuple(map(float, band)) for band in bands]

    result = []
    for band in photon_bands:
        if not isinstance(band, tuple):
            raise ValueError("each photon band must be a (lower_eV, upper_eV) tuple")
        if len(band) != 2:
            raise ValueError("each photon band must have exactly two energy bounds")
        bounds = [
            float(_quantity_values(value, "eV", "photon band bound"))
            for value in band
        ]
        result.append(tuple(bounds))
    return result


def _parse_sed(sed):
    if sed is None:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if not isinstance(sed, (tuple, list)) or len(sed) != 2:
        raise TypeError("sed must be a (frequency, F_nu) pair")
    frequency = _quantity_values(sed[0], "Hz", "SED frequency")
    try:
        f_nu = _quantity_values(sed[1], _F_NU_UNIT, "SED F_nu")
    except ValueError:
        f_nu = _quantity_values(sed[1], _B_NU_UNIT, "SED F_nu")
    if frequency.ndim != 1 or f_nu.ndim != 1 or frequency.shape != f_nu.shape:
        raise ValueError("SED frequency and F_nu must be equal-length 1D arrays")
    if frequency.size < 2:
        raise ValueError("SED must contain at least two samples")
    if (
        not np.isfinite(frequency).all()
        or np.any(frequency <= 0.0)
        or np.any(np.diff(frequency) <= 0.0)
        or not np.isfinite(f_nu).all()
        or np.any(f_nu < 0.0)
    ):
        raise ValueError(
            "SED frequencies must increase and be positive; "
            "F_nu must be finite and nonnegative"
        )
    if np.max(f_nu) > 0.0:
        f_nu = f_nu / np.max(f_nu)
    return (
        np.ascontiguousarray(frequency, dtype=np.float64),
        np.ascontiguousarray(f_nu, dtype=np.float64),
    )


def _group_history(history, element_list, ion_list, photon_bands):
    density_end = 1 + len(element_list)
    ion_end = density_end + len(ion_list)
    return {
        "temperature": history[:, :, 0] * u.K,
        "densities": history[:, :, 1:density_end] / u.cm**3,
        "ion_fractions": history[:, :, density_end:ion_end],
        "photon_densities": history[:, :, ion_end:ion_end + len(photon_bands)] / u.cm**3,
    }


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
    sed=None,
    temperature_is_Tmu=False,
    use_default_depletion=True,
    include_collisional_ionization=True,
    include_photoionization=True,
    include_cosmic_ray_ionization=True,
    include_HM12_UVB=True,
    include_dust_recombination=True,
    include_charge_exchange=True,
    implicit_ions=False,
    x_pct_rule=0.2,
):
    """Evolve cell states at each requested time.

    Physics options select compile-time solver variants. ``fixed_radiation``
    keeps photon densities constant without disabling their effect on chemistry.
    ``implicit_ions`` solves the coupled atomic backward-Euler problem with
    frozen electron density (opt-in; legacy is the accuracy reference).
    ``x_pct_rule`` limits fractional changes across the coupled chemistry,
    temperature, and radiation updates (default 0.2).
    Set ``implicit_ions=False`` to reproduce the legacy ion integrator.
    ``equilibrium`` evolves each requested interval until the solver's
    convergence counter exceeds ``conv_its``. ``rt_advect`` and
    the chemistry options below reproduce the corresponding C++ template and
    runtime configurations.

    ``state`` may be the legacy float64 matrix or a mapping of unit-aware
    arrays: ``temperature`` (K), ``densities`` (cm**-3),
    ``ion_fractions`` (dimensionless), and ``photon_densities`` (cm**-3).
    ``times`` may be a unyt array convertible to seconds, and photon band
    bounds may carry energy units convertible to eV.

    ``sed`` optionally provides ``(frequency, F_nu)`` samples. Frequency must
    be convertible to Hz and F_nu to erg/(s cm**2 Hz), or to specific
    intensity units including sr^-1; its normalization cancels because the
    spectrum is normalized separately within each photon band. Samples must
    cover all bands. If omitted, ``T_blackbody`` retains the legacy black-body
    spectrum.

    Returns a dictionary with ``temperature`` (K), ``densities`` (cm**-3),
    ``ion_fractions`` (dimensionless), and ``photon_densities`` (cm**-3).
    Each value has leading axes ``(Ntimes, Ncell)``; the remaining axis
    indexes elements, ions, or photon bands as appropriate.
    """
    photon_bands = _parse_photon_bands(photon_bands)
    state = _parse_state(state, element_list, ion_list, photon_bands)
    times = _parse_times(times)
    sed_frequency, sed_fnu = _parse_sed(sed)
    T_blackbody = float(_quantity_values(T_blackbody, "K", "T_blackbody"))

    selected_data_path = (
        Path(os.environ.get("RTZ_DATA_DIR", _DATA_PATH))
        if data_path is None
        else Path(data_path)
    )
    history = _solve(
        state,
        times,
        element_list,
        ion_list,
        photon_bands,
        sed_frequency,
        sed_fnu,
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
        implicit_ions,
        x_pct_rule,
    )
    return _group_history(history, element_list, ion_list, photon_bands)


__all__ = ["solve"]
