# gizmo_rtz

`gizmo_rtz` is the standalone C++ RTZ chemistry and radiative-cooling solver.
It can also be built as a Kokkos-backed Python extension using nanobind.

## Python bindings

The project is pip-installable from a checkout. Its build backend pins nanobind
to 3.1.0 and uses CMake/scikit-build-core to compile Kokkos and the Python
extension. NumPy is installed as a runtime dependency; pytest is available in
the `test` extra. Install the Kokkos submodule first:

```bash
git submodule update --init --recursive
```

Using [uv](https://docs.astral.sh/uv/), create an environment, install the
checkout, and run the tests. The default build enables Kokkos OpenMP:

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python '.[test]'
.venv/bin/pytest
```

Kokkos is fetched at its 5.0.2 release if the submodule is not present in the
source tree. The installed wheel includes the rate tables under `rtz/data`.

To build the same package with CUDA and OpenMP, use a separate environment and
a separate CMake build directory. Point CMake at Kokkos's `nvcc_wrapper`, set
the CUDA toolkit root, and select the target GPU architecture. For this
machine's Turing 7.5 GPU with CUDA 13.4:

```bash
uv venv --python 3.14 .venv-cuda
uv pip install --python .venv-cuda/bin/python '.[test]' \
  --config-setting=cmake.define.Kokkos_ENABLE_CUDA=ON
.venv-cuda/bin/pytest
```

To run the test suite from an existing Python environment instead, install
`.[test]` with pip and invoke `pytest`.

The `state` columns are ordered as follows:

1. Temperature in K.
2. One number density in cm^-3 for every item in `element_list`.
3. One ion fraction for every item in `ion_list`.
4. One photon number density in cm^-3 for every item in `photon_bands`.

`state` must be a 2D NumPy `float64` array, and `times` must be a 1D NumPy
`float64` array. Times are in seconds, finite, nonnegative, and strictly
increasing. The element order and ion order are the order of their respective
lists. Element names must be lowercase full names; both `"hydrogen"` and
`"helium"` are required. The supported elements are hydrogen, helium, carbon,
nitrogen, oxygen, neon, magnesium, silicon, sulfur, and iron. Ions must be
`(element_name, stage)` tuples with a 1-based integer stage (`1` is neutral,
`2` is singly ionized). Every element needs at least one ion fraction, and each
element's supplied fractions must sum to one. When `include_H2=True`, the
hydrogen fractions may instead sum to less than one; the residual is
interpreted as the initial H2 fraction, allowing molecular hydrogen to be
carried between solves. All unlisted atomic ion stages start at zero.

The solver supports up to six photon bands, each supplied as a tuple
`(lower_eV, upper_eV)`. If fewer are supplied, the final band's energy bounds
are repeated internally and the extra photon densities are zero. Photon bounds
must be finite and satisfy `0 < lower_eV < upper_eV`.

`rtz.solve(state, times, element_list, ion_list, photon_bands)` integrates from
t=0 to each target time and returns a history with shape
`(Ntimes, Ncell, Ncolumns)`. An entry at t=0 is an unchanged copy of the input.
`max_iterations` is an optional per-interval iteration limit (default 100000).
The rate tables are loaded from this repository's `data/` directory by
default; set `RTZ_DATA_DIR` or pass `data_path=...` to select another data
directory. For 50 snapshots over 1 Gyr, use
`np.linspace(0.0, 3.15576e16, 50)`.

Optional keyword arguments configure the physics:

| Option | Default | Effect |
|---|---:|---|
| `redshift` | `0.0` | Redshift used to update the HM12 UV background. |
| `uv_g0` | `0.0070977` | Dimensionless local UV field strength used by the chemistry network. |
| `constant_temperature` | `False` | Hold the input temperature fixed while evolving chemistry. |
| `include_H2` | `False` | Enable molecular hydrogen chemistry. Its abundance is internal and is not a separate state column. |
| `include_CO` | `False` | Enable CO chemistry; requires carbon and oxygen in `element_list`. CO abundance is internal. |
| `include_self_shielding` | `False` | Enable H2/CO self-shielding factors. |
| `fixed_radiation` | `False` | Hold the supplied photon number densities fixed. Radiation still affects chemistry; when false, absorption depletes the photon field. |
| `rt_advect` | `True` | Select the RT-enabled or RT-disabled solver template. |
| `equilibrium` | `False` | Evolve to convergence instead of to the requested elapsed time. |
| `conv_its` | `20000` | In equilibrium mode, stop after the convergence counter exceeds this threshold. |
| `max_iterations` | `100000` | Maximum iterations per interval before raising an error. |
| `T_blackbody` | `1e4` | Blackbody temperature used to calculate radiation cross-sections, in K. |
| `temperature_is_Tmu` | `False` | Interpret the initial temperature column as the solver's `T/mu` value rather than physical temperature. |
| `use_default_depletion` | `True` | Apply the built-in per-element depletion factors. |
| `include_collisional_ionization` | `True` | Enable collisional ionization. |
| `include_photoionization` | `True` | Enable photoionization. |
| `include_cosmic_ray_ionization` | `True` | Enable cosmic-ray ionization. |
| `include_HM12_UVB` | `True` | Enable the HM12 UV background. |
| `include_dust_recombination` | `True` | Enable dust recombination. |
| `include_charge_exchange` | `True` | Enable charge exchange. |

Example:

```python
import numpy as np
import rtz

elements = ["hydrogen", "helium"]
ions = [("hydrogen", 1), ("hydrogen", 2), ("helium", 1), ("helium", 2)]
bands = [(13.6, 15.2), (15.2, 24.59)]

# T, n_H, n_He, x_HI, x_HII, x_HeI, x_HeII, N_gamma_1, N_gamma_2
state = np.array([[1.0e4, 1.0, 1.0e-2, 0.7, 0.3, 0.9, 0.1, 1.0, 0.3]])
times = np.linspace(0.0, 3.15576e16, 50)
history = rtz.solve(
    state,
    times,
    elements,
    ions,
    bands,
    redshift=0.0,
    uv_g0=1.0,
    include_H2=True,
    fixed_radiation=True,
)
final_state = history[-1]
```

The binding uses `aexp=1` and a cell size of 4.4 pc. It
evolves each row independently with Kokkos, launching the solver's
per-iteration stages as separate kernels in the same order as the PRISM update
path. It does not use a straggler-tail path, and preserves the input column
layout in the result.

The Python tests include a 1 Gyr constant-temperature H/He equilibrium sweep
from 100 K to 1e8 K, using solar He/H, `n_H=1 cm^-3`, `redshift=0`, `uv_g0=1`,
and zero fixed RT photon densities. Golden fractions and a plot resembling
`abundances.pdf` are in `tests/golden/hhe_equilibrium.{txt,pdf}`. To regenerate
the plot, install the plot extra and run:

```bash
python -m pip install -e '.[plot]'
python tests/plot_hhe_equilibrium.py
```

`tests/test_python_bindings.py` also reproduces the default `gizmo CIE`,
`gizmo ISM`, and `gizmo PHOTOHEAT` driver setups and compares the Python-visible
results with the C++ stdout goldens in `tests/golden/gizmo_*.txt`. These driver
cases use `equilibrium=True` for CIE and ISM; PHOTOHEAT is advanced through the
same 1000 fixed-duration steps as the C++ driver. The C++ driver's internal
H2 and CO abundances are not separate Python state columns, so parity checks
cover temperature, density and exposed atomic ion fractions.
