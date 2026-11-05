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

Nsight Systems (`nsys profile --trace=cuda --sample=none --cpuctxsw=none`;
one cold solve, separate from benchmark samples) observed 1,505 subcycles:

| Phase | GPU time | Share | Launch geometry |
| --- | ---: | ---: | --- |
| Ions/commit | 34.135 s | 81.4% | 10,000 blocks × 128 threads |
| Cooling | 5.937 s | 14.2% | 157 blocks × 128 threads |
| Prep/radiation | 0.560 s | 1.3% | 79 blocks × 128 threads |
| Pending reduction | 0.0269 s | 0.1% | 40 blocks × 256 threads |
| Iteration-limit reduction | 0.0214 s | 0.1% | 40 blocks × 256 threads |

The ion kernel launches all 10,000 cell teams on every subcycle. CUDA API
`cudaStreamSynchronize` time (40.51 s) mostly waits for these kernels and
must not be added to GPU time. This profile suggests active-cell compaction
will matter more than eliminating the very short reduction kernels. Raw
summary CSVs are in `results/baseline-profile_cuda_*_sum.csv`; the full trace
and exported SQLite database are local artifacts.

## Trial 01: combine pending and iteration-limit reductions

Change: compute pending count and maximum unfinished-cell iteration in one
Kokkos reduction with two reducers. Remove the separate limit reduction and
its host synchronization. Preserve the condition that a cell completing on
the last allowed iteration does not trigger an error. No physics changes.

Median **40.784318 s**, speedup **1.00084×** (0.084% less time). Samples:
40.807639, 40.784318, 40.789906, 40.784294, 40.779445 s. Every output
is bitwise identical to baseline. See `results/trial01.json`.

**Not retained**: the 34 ms gain is negligible for this workload; no paired
rerun establishes it independently from drift. The kernel profile already
showed the reductions are a small part of runtime. Source patch preserved in
`trial01-combined-reduction.patch`; trial 02 starts from the original solver.

## Trial 02: compact active-cell indices

Change: keep cell states in their original locations, but launch subcycle
kernels and reductions through an active index list. When cells finish, a
stable device scan builds the next list; swap lists and shrink launch ranges.
Reset the list for every output interval. Skip compaction if no cells finish.
Keep the original two reductions, timestep control, team size, and physics.
Extra index storage: two arrays of 10,000 `size_t` values (160 KB total).

Median **36.682302 s**, **1.11276×** faster than baseline (10.13% less time).
Samples: 36.682302, 36.676885, 36.680548, 36.684004, 36.686162 s.
All four output arrays are bitwise identical to baseline over all 10,000 cells.
18 CUDA binding tests pass (4.11 s), including golden 1 Myr evolution,
1 Gyr H/He equilibrium, mixed-cell/multiple-interval ordering, fixed and
absorbing radiation, and the iteration limit. **Retained**.

The three sequential C++ golden sweep comparisons were excluded from the
GPU regression run. A baseline attempt was explicitly interrupted during CIE
after 18 tests passed in 249.89 s; the CIE sweep alone asks for roughly six
million single-cell subcycles. No passing result is claimed for those sweeps.
See `results/trial02.json` and `results/trial02-tests.json`.
