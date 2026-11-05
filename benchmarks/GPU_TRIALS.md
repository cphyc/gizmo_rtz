# Galaxy solver GPU trials

Objective: measure the local GPU baseline, then test one change at a time.
Retain and commit only improvements with measured speedups and correctness
checks. Failed trials stay recorded here.

## Workload and measurement

The input is `recombination_lag/galaxy_cells.npz`: 10,000 independent atomic
ISM+CGM cells, 56 ion fractions, five radiation bands, fixed local radiation,
cooling plus chemistry for 100 years. Initial conditions are identical for every
repetition. `benchmarks/galaxy_gpu.py` loads the input directly and explicitly
selects a compiled extension; it never rebuilds the equilibrium table.

Report wall time for the complete synchronous Python solve (including table
initialization, transfers, allocations, and output). Discard one warmup and
report all five timed samples and their median. Profiling is separate from
timing. Compare every output against the baseline with `rtol=1e-10,
atol=1e-12`; report maximum absolute differences too. Input and extension
SHA256 hashes are recorded in result JSON files. Binary output snapshots are
local artifacts; preserve them for subsequent comparisons.

Hardware observed 2026-11-05: NVIDIA Quadro T1000 with Max-Q Design, 4 GiB,
driver 615.78.08; CUDA 13.4.92, GCC 16.2.1, Kokkos 5.0.2, Python 3.14.9.
Release build, double precision, no fast math, CUDA sm_75, Serial host backend.
GPU clocks are unmanaged; rerun the baseline when assessing small gains.

## Build and run

```sh
env PATH=/opt/cuda/bin:/usr/bin:/bin NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++ \
  cmake -S . -B build/galaxy-gpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$PWD/extern/kokkos/bin/nvcc_wrapper" \
  -DCMAKE_CXX_FLAGS=-allow-unsupported-compiler \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ARCH_TURING75=ON -DGIZMO_RTZ_BUILD_PYTHON=ON \
  -DGIZMO_RTZ_FAST_MATH=OFF -DPython_EXECUTABLE=/usr/bin/python3.14
env PATH=/opt/cuda/bin:/usr/bin:/bin NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++ \
  cmake --build build/galaxy-gpu --target _rtz -j 2
/usr/bin/python3.14 benchmarks/galaxy_gpu.py \
  --extension build/galaxy-gpu/_rtz.cpython-314-x86_64-linux-gnu.so \
  --output benchmarks/results/baseline
```

The PATH override bypasses an sccache shim whose daemon is unavailable in the
sandbox. The install RPATH setting avoids a CMake/Ninja nvcc-wrapper detection
failure. Existing CUDA binaries predate the current SED API, so they cannot
serve as the baseline. The existing Python 3.13 environment remains unchanged.

## Baseline

Source revision: `e35c9e0`. Median **40.818548 s**, 244.99 cells/s.
Samples: 40.819084, 40.817201, 40.813961, 40.825974, 40.818548 s;
warmup 42.473317 s. GPU at completion: 66 C, 1530 MHz. See
`results/baseline.json` for hashes and raw timings. Preserved baseline binary:
`build/galaxy-gpu/baseline.cpython-314-x86_64-linux-gnu.so` (local artifact).
CUDA device access requires running outside the filesystem sandbox; a CUDA
driver query returned one device and the benchmark reached 100% GPU utilization.

Inspection: each subcycle launches prep/radiation, two cooling lanes per cell,
a team per cell for molecules/ions/commit, then two host-result reductions.
All launches span the original cell count until the last cell finishes. This
suggests testing active-cell compaction and reducing launch/synchronization
overhead. Register/occupancy limits need measurement before changing launch
geometry.

`cuobjdump --dump-resource-usage` reports 130 registers per thread for cooling,
95 for ions/commit, and 64 for prep/radiation; each reports zero local memory.
These are compiled resource counts, not measured occupancy. The selected
template is `run_solver<false,false,false,true,false>`; resource records are
in `results/baseline-resources.json` (lambda #1 init, #2 prep, #3 cooling,
team lambda ions/commit, #4 finish).

## Trial 01: combine pending and iteration-limit reductions

Change: compute pending count and maximum unfinished-cell iteration in one
Kokkos reduction with two reducers. Remove the separate limit reduction and
its host synchronization. Preserve the condition that a cell completing on
the last allowed iteration does not trigger an error. No physics changes.

Measurement and correctness validation pending.
