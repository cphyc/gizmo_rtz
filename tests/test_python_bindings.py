from pathlib import Path

import numpy as np
import pytest

import rtz


@pytest.fixture
def initial_state():
    elements = ["hydrogen", "helium"]
    ions = [
        ("hydrogen", 1),
        ("hydrogen", 2),
        ("helium", 1),
        ("helium", 2),
        ("helium", 3),
    ]
    photon_bands = [(13.6, 15.2), (15.2, 24.59)]
    state = np.array(
        [[1.0e5, 1e1, 1e-1, 0.1, 0.9, 0.3, 0.3, 0.4, 1, 1e3]],
        dtype=np.float64,
    )
    return state, elements, ions, photon_bands


def test_zero_time_returns_unchanged_state(initial_state):
    state, elements, ions, bands = initial_state
    history = rtz.solve(state, np.array([0.0]), elements, ions, bands)
    assert history.shape == (1, 1, state.shape[1])
    assert np.array_equal(history[0], state)


def test_rejects_non_numpy_state_and_times(initial_state):
    state, elements, ions, bands = initial_state
    with pytest.raises(TypeError):
        rtz.solve(state.tolist(), np.array([0.0]), elements, ions, bands)
    with pytest.raises(TypeError):
        rtz.solve(state.astype(np.float32), np.array([0.0]), elements, ions, bands)
    with pytest.raises(TypeError):
        rtz.solve(state, [0.0], elements, ions, bands)
    with pytest.raises(TypeError):
        rtz.solve(state, 0.0, elements, ions, bands)
    with pytest.raises(TypeError):
        rtz.solve(state, np.array([0.0], dtype=np.float32), elements, ions, bands)


def test_rejects_inconsistent_state_columns(initial_state):
    state, elements, ions, bands = initial_state
    with pytest.raises(ValueError, match="state must have columns"):
        rtz.solve(state[:, :-1], np.array([0.0]), elements, ions, bands)


@pytest.mark.parametrize("element", ["H", "Hydrogen"])
def test_rejects_noncanonical_elements(initial_state, element):
    state, elements, ions, bands = initial_state
    elements[0] = element
    with pytest.raises(ValueError, match="unsupported element"):
        rtz.solve(state, np.array([0.0]), elements, ions, bands)


def test_requires_hydrogen_and_helium(initial_state):
    state, elements, ions, bands = initial_state
    with pytest.raises(ValueError, match="hydrogen and helium"):
        rtz.solve(state[:, :], np.array([0.0]), ["hydrogen"], ions[:2], bands)


def test_requires_photon_band_tuples(initial_state):
    state, elements, ions, bands = initial_state
    bands[0] = [13.6, 15.2]
    with pytest.raises(ValueError, match="each photon band"):
        rtz.solve(state, np.array([0.0]), elements, ions, bands)


def test_include_co_requires_carbon_and_oxygen(initial_state):
    state, elements, ions, bands = initial_state
    with pytest.raises(ValueError, match="carbon and oxygen"):
        rtz.solve(
            state, np.array([0.0]), elements, ions, bands, include_CO=True
        )


def test_rejects_nonincreasing_times(initial_state):
    state, elements, ions, bands = initial_state
    with pytest.raises(ValueError, match="strictly increasing"):
        rtz.solve(state, np.array([1.0, 1.0]), elements, ions, bands)


def test_rejects_unnormalized_ion_fractions(initial_state):
    state, elements, ions, bands = initial_state
    state = state.copy()
    state[0, 3] = 0.6
    with pytest.raises(ValueError, match="must sum to one"):
        rtz.solve(state, np.array([0.0]), elements, ions, bands)


def test_evolves_one_myr_to_golden_state(initial_state):
    state, elements, ions, bands = initial_state
    result = rtz.solve(
        state,
        np.array([0.0, 3.15576e13]),
        elements,
        ions,
        bands,
        max_iterations=100000,
    )
    # Generated with the current OpenMP solver; CPU and CUDA subcycling differ slightly.
    expected = np.array(
        [
            3.8931047515583123e4,
            1.0e1,
            1.0e-1,
            2.3734290088839316e-8,
            9.9999997626570991e-1,
            6.0891624428981994e-2,
            9.3892356903185092e-1,
            1.8480653916703294e-4,
            4.0885923496759230e-1,
            5.2947722324624340e2,
        ]
    )
    assert result.shape == (2, *state.shape)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[1, 0], expected, rtol=1e-8, atol=1e-13)


def test_constant_temperature_and_fixed_radiation(initial_state):
    initial, _, initial_ions, bands = initial_state
    elements = ["hydrogen", "helium", "carbon", "oxygen"]
    ions = [
        *initial_ions,
        ("carbon", 1),
        ("oxygen", 1),
    ]
    state = np.array(
        [
            [
                *initial[0, :3],
                1.0e-4,
                3.0e-4,
                *initial[0, 3:8],
                1.0,
                1.0,
                *initial[0, 8:],
            ]
        ],
        dtype=np.float64,
    )
    fixed = rtz.solve(
        state,
        np.array([0.0, 1.0e12]),
        elements,
        ions,
        bands,
        redshift=2.0,
        constant_temperature=True,
        include_H2=True,
        include_CO=True,
        include_self_shielding=True,
        fixed_radiation=True,
    )
    assert np.isfinite(fixed).all()
    assert fixed[1, 0, 0] == state[0, 0]
    np.testing.assert_array_equal(fixed[1, 0, -len(bands) :], state[0, -len(bands) :])


def test_absorbing_radiation_depletes_photons(initial_state):
    state, elements, ions, bands = initial_state
    times = np.array([0.0, 1.0e12])
    fixed = rtz.solve(state, times, elements, ions, bands, fixed_radiation=True)
    absorbed = rtz.solve(state, times, elements, ions, bands)
    np.testing.assert_array_equal(fixed[1, 0, -len(bands) :], state[0, -len(bands) :])
    assert np.all(absorbed[1, 0, -len(bands) :] <= state[0, -len(bands) :])
    assert np.any(absorbed[1, 0, -len(bands) :] < state[0, -len(bands) :])


def test_hydrogen_helium_equilibrium_after_one_gyr():
    temperatures = np.geomspace(1.0e2, 1.0e8, 100)
    elements = ["hydrogen", "helium"]
    ions = [
        ("hydrogen", 1),
        ("hydrogen", 2),
        ("helium", 1),
        ("helium", 2),
        ("helium", 3),
    ]
    photon_bands = [(13.6, 24.59)]
    state = np.zeros((temperatures.size, 1 + len(elements) + len(ions) + 1))
    state[:, 0] = temperatures
    state[:, 1] = 1.0
    state[:, 2] = 8.51e-2
    state[:, 3] = 1.0
    state[:, 5] = 1.0

    history = rtz.solve(
        state,
        np.array([0.0, 3.15576e16]),
        elements,
        ions,
        photon_bands,
        redshift=0.0,
        uv_g0=1.0,
        constant_temperature=True,
        fixed_radiation=True,
    )

    equilibrium = history[-1]
    expected = np.loadtxt(
        Path(__file__).parent / "golden" / "hhe_equilibrium.txt"
    )
    assert history.shape == (2, temperatures.size, state.shape[1])
    assert np.isfinite(history).all()
    np.testing.assert_array_equal(expected[:, 0], temperatures)
    np.testing.assert_array_equal(equilibrium[:, 0], temperatures)
    np.testing.assert_array_equal(equilibrium[:, -1], 0.0)
    np.testing.assert_allclose(equilibrium[:, 3:8], expected[:, 1:], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(equilibrium[:, 3:5].sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(equilibrium[:, 5:8].sum(axis=1), 1.0, atol=1e-10)


_DRIVER_ELEMENTS = [
    "hydrogen",
    "helium",
    "carbon",
    "nitrogen",
    "oxygen",
    "neon",
    "magnesium",
    "silicon",
    "sulfur",
    "iron",
]
_DRIVER_ATOMIC_NUMBERS = [1, 2, 6, 7, 8, 10, 12, 14, 16, 26]
_DRIVER_SOLAR_ABUNDANCES = np.array(
    [1.0, 8.51e-2, 2.69e-4, 6.76e-5, 4.90e-4, 8.51e-5, 3.98e-5, 3.24e-5, 1.32e-5, 3.16e-5]
)
_DRIVER_DEPLETION = np.array([1.0, 1.0, 0.5, 0.6, 0.73, 1.0, 0.16, 0.1, 1.0, 0.01])
_DRIVER_IONS = [
    (element, stage)
    for element, atomic_number in zip(_DRIVER_ELEMENTS, _DRIVER_ATOMIC_NUMBERS)
    for stage in range(1, atomic_number + 2)
]
_DRIVER_BANDS = [
    (5.6, 11.2),
    (11.2, 13.6),
    (13.6, 15.2),
    (15.2, 24.5),
    (24.5, 54.4),
    (54.4, 500.0),
]


def _read_cpp_golden(name):
    golden_path = Path(__file__).parent / "golden" / f"gizmo_{name}.txt"
    numeric_lines = [
        line
        for line in golden_path.read_text().splitlines()
        if line and line[0].isdigit()
    ]
    return np.loadtxt(numeric_lines)


def _driver_state(temperatures, hydrogen_densities, metallicity, depletion, bands, photons):
    temperatures = np.asarray(temperatures, dtype=np.float64)
    hydrogen_densities = np.broadcast_to(
        np.asarray(hydrogen_densities, dtype=np.float64), temperatures.shape
    )
    densities = (
        hydrogen_densities[:, None]
        * _DRIVER_SOLAR_ABUNDANCES[None, :]
        * metallicity
        * depletion
    )
    densities[:, 0] = hydrogen_densities
    fractions = np.full((temperatures.size, len(_DRIVER_IONS)), 1.0e-20)
    ion_offset = 0
    for atomic_number in _DRIVER_ATOMIC_NUMBERS:
        fractions[:, ion_offset] = 1.0
        ion_offset += atomic_number + 1
    photon_values = np.broadcast_to(
        np.asarray(photons, dtype=np.float64), (temperatures.size, len(bands))
    )
    return np.column_stack((temperatures, densities, fractions, photon_values))


def test_python_cie_matches_gizmo_golden():
    expected = _read_cpp_golden("cie")
    state = _driver_state(
        expected[:1, 0],
        np.full(1, 1.0e-2),
        0.1,
        _DRIVER_DEPLETION,
        [_DRIVER_BANDS[2]],
        [0.0],
    )
    ion_offset = 1 + len(_DRIVER_ELEMENTS)
    actual_states = []
    for expected_row in expected:
        state[0, 0] = expected_row[0]
        history = rtz.solve(
            state,
            np.array([1.0]),
            _DRIVER_ELEMENTS,
            _DRIVER_IONS,
            [_DRIVER_BANDS[2]],
            constant_temperature=True,
            include_H2=True,
            rt_advect=False,
            equilibrium=True,
            conv_its=20000,
            max_iterations=40000,
            include_collisional_ionization=True,
            include_photoionization=False,
            include_cosmic_ray_ionization=False,
            include_HM12_UVB=False,
            include_dust_recombination=False,
            include_charge_exchange=False,
        )
        state = history[-1].copy()
        actual_states.append(state[0].copy())

    actual_states = np.asarray(actual_states)
    actual_ions = actual_states[:, ion_offset : ion_offset + len(_DRIVER_IONS)]
    np.testing.assert_allclose(actual_states[:, 0], expected[:, 0], rtol=1e-5)
    np.testing.assert_allclose(actual_ions, expected[:, 1:], rtol=1e-5, atol=1e-25)


def test_python_ism_matches_gizmo_golden():
    expected = _read_cpp_golden("ism")
    hydrogen_densities = np.power(
        10.0, -3.0 + np.arange(expected.shape[0], dtype=np.float64) * 7.0 / 100.0
    )
    photon_values = [
        0.0026284373311105036,
        0.00087614577703683456,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    state = _driver_state(
        [1.0e5],
        [hydrogen_densities[0]],
        0.1,
        np.ones_like(_DRIVER_DEPLETION),
        _DRIVER_BANDS,
        photon_values,
    )
    actual_states = []
    for hydrogen_density in hydrogen_densities:
        state[0, 0] = 1.0e5
        state[0, 1 : 1 + len(_DRIVER_ELEMENTS)] = (
            hydrogen_density * _DRIVER_SOLAR_ABUNDANCES * 0.1
        )
        state[0, 1] = hydrogen_density
        history = rtz.solve(
            state,
            np.array([1.0]),
            _DRIVER_ELEMENTS,
            _DRIVER_IONS,
            _DRIVER_BANDS,
            uv_g0=0.0,
            include_H2=True,
            include_CO=True,
            fixed_radiation=True,
            equilibrium=True,
            conv_its=20000,
            max_iterations=40000,
            T_blackbody=1.0e5,
            temperature_is_Tmu=True,
            use_default_depletion=False,
            include_HM12_UVB=False,
        )
        state = history[-1].copy()
        actual_states.append(state[0].copy())

    result = np.asarray(actual_states)
    carbon_column = 1 + _DRIVER_ELEMENTS.index("carbon")
    ion_offset = 1 + len(_DRIVER_ELEMENTS)
    hydrogen_neutral_column = ion_offset
    hydrogen_ionized_column = ion_offset + 1
    assert history.shape == (2, *state.shape)
    np.testing.assert_allclose(result[:, 0], expected[:, 1], rtol=1e-5)
    np.testing.assert_allclose(result[:, carbon_column], expected[:, 4], rtol=1e-5)
    np.testing.assert_allclose(
        result[:, hydrogen_neutral_column], expected[:, 6], rtol=1e-5, atol=1e-25
    )
    np.testing.assert_allclose(
        result[:, hydrogen_ionized_column], expected[:, 7], rtol=1e-5, atol=1e-25
    )


def test_python_photoheat_matches_gizmo_golden():
    expected = _read_cpp_golden("photoheat")
    elements = ["hydrogen", "helium"]
    ions = [
        ("hydrogen", 1),
        ("hydrogen", 2),
        ("helium", 1),
        ("helium", 2),
        ("helium", 3),
    ]
    state = np.zeros((1, 1 + len(elements) + len(ions) + len(_DRIVER_BANDS)))
    state[0, 0] = 5000.0 / 1.24
    state[0, 1:3] = [1.0e-2, 1.0e-2 * 8.51e-2]
    state[0, 3:8] = [1.0 - 1e-8, 1e-8, 1.0 - 1e-8, 1e-8, 0.0]
    state[0, 3 + len(ions) + 2] = 1.0
    times = np.arange(expected.shape[0] + 1, dtype=np.float64) * 3.0e15
    history = rtz.solve(
        state,
        times,
        elements,
        ions,
        _DRIVER_BANDS,
        max_iterations=4000000,
        redshift=0.0,
        constant_temperature=False,
        fixed_radiation=True,
        rt_advect=True,
        T_blackbody=1.0e5,
        temperature_is_Tmu=True,
        include_collisional_ionization=True,
        include_photoionization=True,
        include_cosmic_ray_ionization=False,
        include_HM12_UVB=False,
        include_dust_recombination=False,
        include_charge_exchange=False,
    )

    result = history[1:, 0]
    assert history.shape == (expected.shape[0] + 1, 1, state.shape[1])
    np.testing.assert_allclose(result[:, 0], expected[:, 3], rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(result[:, 3:6], expected[:, 4:7], rtol=1e-5, atol=1e-25)
