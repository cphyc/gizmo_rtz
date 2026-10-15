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
            state, np.array([0.0]), elements, ions, bands, include_co=True
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
        include_h2=True,
        include_co=True,
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
