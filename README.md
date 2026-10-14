# gizmo_rtz

`gizmo_rtz` is the standalone C++ RTZ chemistry and radiative-cooling solver.
It can also be built as a Kokkos-backed Python extension using nanobind.

## Python bindings

The project is pip-installable from a checkout. Its build backend pins nanobind
to 3.1.0 and uses CMake/scikit-build-core to compile both Kokkos and the Python
extension. Initialize the Kokkos submodule when building from a clone:

```bash
git submodule update --init --recursive
python -m pip install .
```

The pip build enables Kokkos OpenMP by default for CPU parallelism. CUDA can be
enabled with the Kokkos CMake option; compile the extension with Kokkos's
`nvcc_wrapper` and provide a CUDA toolkit:

```bash
NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/c++ \
CXX="$PWD/extern/kokkos/bin/nvcc_wrapper" \
CXXFLAGS="-allow-unsupported-compiler" \
  python -m pip install . \
  -Ccmake.define.Kokkos_ENABLE_CUDA=ON \
  -Ccmake.define.Kokkos_ENABLE_OPENMP=ON \
  -Ccmake.define.Kokkos_ARCH_TURING75=ON
```

Kokkos is fetched at its 5.0.2 release if the submodule is not present in the
source tree. The installed wheel includes the rate tables under `rtz/data`.
The `CXXFLAGS` override above is only needed with a CUDA toolkit whose supported
host-compiler range excludes the installed GCC version. This machine has a
Turing 7.5 GPU; replace `Kokkos_ARCH_TURING75` with the matching Kokkos
architecture option for a different GPU.

The `rtz` module expects `state` to be a two-dimensional NumPy `float64` array
with columns in this order:

1. Temperature in K.
2. One number density in cm^-3 for every item in `element_list`.
3. One ion fraction for every item in `ion_list`.
4. One photon number density in cm^-3 for every item in `photon_bands`.

The element order and ion order are the order of their respective lists. Ion
stages are 1-based (`1` is neutral, `2` is singly ionized); ions can be named as
`"hydrogen_01"` or `("hydrogen", 1)`. Element names can be full names or
symbols. The supported elements are H, He, C, N, O, Ne, Mg, Si, S, and Fe.
Every element needs at least one ion fraction, and each element's supplied
fractions must sum to one. All unlisted ion stages start at zero.

The solver supports up to six photon bands. If fewer are supplied, the final
band's energy bounds are repeated internally and the extra photon densities are
zero. Photon bounds are in eV and must have positive lower bounds below their
upper bounds.

`rtz.solve(state, times, element_list, ion_list, photon_bands)` integrates from
t=0 to each target time in seconds. A scalar `times` returns the final state
with shape `(Ncell, Ncolumns)`. A sequence such as a NumPy array returns a
history with shape `(Ntimes, Ncell, Ncolumns)`; an entry at t=0 is an unchanged
copy of the initial state. `max_iterations` is an optional per-interval
iteration limit (default 100000). The rate tables are loaded from this
repository's `data/` directory by default; set `RTZ_DATA_DIR` or pass
`data_path=...` to select another data directory. A scalar time integrates from
t=0; pass the full time sequence in one call to preserve the requested
intervals. For 50 snapshots over 1 Gyr, use `np.linspace(..., 50)`: an
`np.arange(..., 50)` step requests a snapshot every 50 seconds.

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
history = rtz.solve(state, times, elements, ions, bands)
final_state = history[-1]
```

The binding uses the solver's standalone defaults: redshift 0, `aexp=1`,
`T_blackbody=1e4 K`, and a cell size of 4.4 pc. It evolves each row
independently with Kokkos and preserves the input column layout in the result.
