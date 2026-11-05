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

Portability check after compaction: a fresh Release OpenMP build with fast
math disabled also passes those 18 binding tests (0.99 s, two host threads).
See `results/trial02-cpu-tests.json`. The CUDA-only team-size trial does not
change the host policy. This is a correctness check, not a CPU speedup claim.

The separate Nsight Systems trace confirms 1,505 subcycles and a reduction
from 15,050,000 to 2,389,309 cell teams (84.1% fewer). Active counts at
subcycles 0/99/499/999/1499 are 10000/5210/1599/233/1. This confirms a
long straggler tail, but the modest time improvement indicates substantial
cost remains in active-cell work. See `results/trial02-active-counts.json`
and the trial02 profile summaries.

Phase times clarify the gain: cooling falls from 5.937 s to 2.018 s, while
ions/commit stays at 33.986 s (89.9% of GPU kernel time) versus 34.135 s in
baseline. Removing completed cell teams alone barely changes ion time;
packing active cells into the cooling lanes accounts for most of the win.
The next trial therefore targets utilization within the active ion teams.

Nsight Compute LaunchStats/Occupancy collection was attempted on a 256-cell
solve, but failed with `ERR_NVGPUCTRPERM`: the driver denies hardware counter
access. No occupancy measurement is claimed. Static resource counts and
Nsight Systems launch geometry remain available; no driver settings changed.

The user subsequently ran the same `ncu` command with sudo successfully.
For the first ion launch (256 cells), 95 registers/thread limit residency to
5 blocks/SM: theoretical occupancy 62.50%, achieved 55.64% (17.81 active
warps/SM). Shared memory is not the binding resource (14-block limit).
The launch has 3.66 waves/SM on 14 SMs; its partial-wave estimate is specific
to this small launch, not a measured whole-workload speedup. Recorded in
`results/trial02-occupancy.json`.

## Trial 03: one CUDA warp per cell team

Starting from committed trial 02, set CUDA ion teams to 32 threads (baseline
AUTO chose 128). Host backends keep AUTO. All ion work still uses the same
TeamThreadRange loops and scratch snapshot; stages beyond 32 are handled
by additional loop iterations. This tests team utilization without changing
rate equations, precision, timestep control, or convergence settings.

Median **36.467472 s**, **1.00589×** faster than committed compaction
(0.586% less time), and **1.11931×** faster than original baseline.
Samples: 36.467472, 36.470889, 36.475718, 36.459842, 36.460235 s.
All outputs are bitwise identical to the original baseline. 18 CUDA tests
pass in 4.04 s. **Retained**.

A subsequent 128-thread control run takes 38.219789 s cold, then 36.694135
and 36.695595 s warm, confirming the small improvement against adjacent
control measurements with the GPU still at 1530 MHz (69 C). The cold control
sample includes setup and is not compared with warm candidate samples.

Compiled ion registers remain 95 per thread; this is a team geometry gain,
not a reduction in registers/thread. See `results/trial03.json`,
`results/trial03-control.json`, and `results/trial03-tests.json`.

## Trial 04: float32 chemistry state storage

Starting from trial 03, enable `GIZMO_RTZ_REDUCED_PRECISION`: float chemistry
densities and ion-fraction arrays, with the existing double arithmetic and
temperature state. This is a storage trial, not all-float arithmetic. The
existing CMake option was not propagated to `_rtz`, so this trial adds that
definition to the Python extension target. Its old uint16 description was
stale; actual storage uses float. The project option default remains OFF;
the local GPU trial build explicitly enables it.

The user's initial ion tolerance was relative 1e-6 for reference fractions
above 1e-4. The first full-dataset check failed it: 19,224 of 203,932 ion
values exceeded that limit. The user then explicitly requested retaining the
trial and relaxed the ion limit to **less than 5%**. The temperature limit
remains **less than 1%**. The initially attempted double rebuild was stopped,
and the local GPU configuration was set back to reduced precision ON.

Under the revised criteria, the full-dataset result **passes**: maximum ion
relative error **0.0294172** (2.94%), maximum temperature relative error
**0.000737818** (0.0738%). The worst ion is Fe XIV in cell 1961: reference
0.000108944314, float32 0.000112149151, initially T=991450 K and
nH=349.94 cm^-3. Maximum absolute error among reference fractions at/below
1e-4 is 2.93e-6. The initial strict-failure JSON is preserved for provenance.

The harness `--accuracy physical --ion-rtol .05` checks every cell/output,
records violations, and saves results before raising on failure. Density and
photon outputs also get relative 1e-6 checks. Strict comparison remains the
default; it was used for all double-precision trials. The physical accuracy
limits are workload validation limits, not relaxed internal solver settings.

**Retained at the user's explicit request.** Median **36.369796 s**:
36.373434, 36.372413, 36.369796, 36.355144, 36.357038 s.
This is **1.00269×** faster than trial 03 and **1.12232×** faster than the
original baseline. The storage-only gain is small; it does not establish the
benefit of float32 arithmetic. See `results/trial04.json` for timings and
passing revised accuracy metrics. The canonical `_rtz` binary matches the
preserved trial04 binary (SHA256 recorded there), and the local CMake cache
has reduced precision ON.

13 precision-compatible CUDA API/physics regressions pass in 2.66 s,
including mixed-batch interval ordering, iteration limits, fixed-temperature
molecular/radiation behavior, and absorbing photons. Eight tests requiring
exact float64 round trips or tight float64 golden results were deselected;
this does not claim that the float32 build passes those strict tests. Default
double builds retain their previously validated strict behavior.

## Next: arithmetic propagation

The user asked whether float32 reaches all headers/source calculations. It
does not in trial04: only `CompactIonData` densities/fractions are float;
rate functions, intermediates, and temperature remain double. The next trial
must separately propagate a computation type through the device chemistry
and cooling path, audit literals/overloads to prevent promotion back to
double, and check the same full-dataset <5% ion / <1% temperature limits.
Host file parsing and Python's float64 API buffers require explicit boundary
conversions rather than a blanket keyword replacement.

## Trial 05: propagate float32 computation through the cell evolution

Experimental `GIZMO_RTZ_FLOAT32_ARITHMETIC=ON` selects `Real=float`;
the default is `Real=double`. This includes the tabulated coefficient views,
element properties, chemistry/radiation/thermal cell state, rate functions,
cooling and molecular functions, subcycle updates, and their signatures.
Explicit Real literals and math wrappers prevent integer exponents/global C
math overloads from promoting the calculations back to double. Python's
float64 buffers, host text parsing, SED quadrature, and the external time list
retain explicit double boundaries. No fast-math flags are introduced.

The reference implementation in `dyablo_api.cpp` retains its default double
build; the experimental macro is set for the GPU solver executable and the
Python extension. This trial is a separate build option from float32 storage.
CPU syntax checks pass; CUDA build and numerical validation are in progress.
The accepted trial04 binary remains preserved for comparisons and fallback.

The initial arithmetic build compiles on both CPU and CUDA but fails to
finish: CPU exceeds 100,000 iterations on 256 cells; a bounded CUDA check
also exceeds 3,000 iterations. CPU diagnostics identify cell 5 stuck at
elapsed=0, dt=0, residual=9.727425. The float32 line Planck prefactor cubes
~1e13 Hz frequencies, overflowing before multiplication by Planck's constant
and contaminating fine-structure cooling. Trial05's next revision replaces
`2*h*nu^3/c^2` with `(2*h*nu)*(nu/c)*(nu/c)` in float mode, retaining the
original double expression in double mode. This is an algebraic range fix,
not a changed rate or solver tolerance. Diagnostics now include the first
unfinished cell's timestep, temperature, electron density, and residual.

Trial05 repaired arithmetic benchmark **completed on the local GPU** over
all 10,000 cells, 100 years, default 100,000 iteration limit. Warmup 5.809498 s;
timed samples 4.308273, 4.298244, 4.301146, 4.301546, 4.301220 s.
Median **4.301220 s**, raw speedup **9.48999x** over the original baseline
and **8.45569x** over storage-only trial04. No convergence settings changed.

**Experimental, not accuracy-accepted:** 17 of 203,932 checked ion fractions
exceed relative 5% error (maximum 11.7698%). One temperature exceeds 1%
(maximum 1.22227%). All solves finish; the harness exits with status 1 only
after saving the completed timings/output because the accuracy gate fails.
See `results/trial05.json`. The NPZ output and compiled trial05 binary remain
local artifacts. The float32 arithmetic option remains OFF by project default;
the local trial build enables it. The previously accepted storage-only binary
is preserved as `build/galaxy-gpu/trial04.cpython-314-x86_64-linux-gnu.so`.
No claim is made that default-backend regression tests have been rerun after
this experimental computation-type refactor.


## Compact tables / mapping policies (float32)

Requested structural follow-up: rename dense setup data to RawTabulatedData,
share immutable cell offsets, and use CompactTabulatedData with generic
C++20 mapping policies. See `COMPACT_TABLES.md` for the interface and tests.
The raw cache remains on the host; original GPU spectrum setup uses temporary
dense staging so numerical behavior stays unchanged. The first host-only
setup revision was rejected for tiny output differences. The final full GPU
comparison is bit-for-bit identical to trial05 in every output array.

Median 6.611024 s (five warm runs), versus raw float32 4.301220 s: **slower**,
not a successful speed optimization. Sparse fixture table payload drops from
786246 to 124611 bytes; this is a memory-layout feature. The fixed per-cell
integer array becomes an 8-byte pointer to shared storage. Full timing and
exact comparison are in `results/compact-policy-final-gpu.json`.


## Trial 06: one device-resident table descriptor per solve (float32)

Profile of the compact baseline: ions/commit consumes 64.5% of GPU kernel
time (4.868 s in the cold profile); prep/radiation 0.913 s and cooling 0.490 s.
The ion kernel uses 72 registers/thread and 13,696 bytes of parameter constant
space, versus 64 registers and 3,520 bytes in the raw float32 build.

Change: construct one descriptor in device storage, keep its host owners
alive for the solve, and pass a pointer in subcycle kernels. Construction and
destruction use Kokkos kernels with explicit object lifetime; no view ownership
is transferred by a raw byte copy. NVCC disallows extended lambdas directly
inside constructors/destructors, so those launches live in named static helpers.
Mappings and arithmetic are unchanged.

Median **6.523015 s**, versus compact baseline **6.611024 s**: **1.01349x**
speedup (1.33% less time). Samples: 6.523132, 6.517449, 6.523015, 6.550084,
6.520428 s. Full 10,000-cell GPU results remain bit-for-bit identical to the
raw float32 reference. Ion registers drop to 64/thread and parameter space
to 744 bytes. No local-memory allocation is reported. The large parameter
block was reduced, but most of the performance gap remains.

**Retained.** Default-double CPU tests: 18 passed (three long sweeps excluded).
Float32 CUDA tests: 13 passed (eight float64-specific tests excluded). Data
and timing hashes are in `results/descriptor-gpu.json`; kernel summaries in
`results/compact-profile_cuda_*_sum.csv`. The raw float32 accuracy limitations
against the double reference remain unchanged; this trial adds no differences.


## Trial 07: compile-time extents (float32; not retained)

Precision: both `GIZMO_RTZ_FLOAT32_ARITHMETIC=ON` and
`GIZMO_RTZ_REDUCED_PRECISION=ON`, as in trial06. Default-double CPU tests
are separate and must not be mistaken for float64 GPU performance results.

Change: use compile-time raw extents for the built-in policies, retaining
runtime extents for dynamic shapes and leaving custom mapping contracts
unchanged. Added a dynamic-shape fallback check. Median **6.566065 s** versus
trial06 **6.523015 s**; no measured improvement. Samples: 6.582285, 6.570811,
6.563905, 6.555041, 6.566065 s. Full GPU output is bit-for-bit identical.
Register count remains 64/thread; parameter space remains 744 bytes.
**Reverted**. Source patch: `trial07-static-extents.patch`; measurements:
`results/static-shape-gpu.json`.

## Precision correction and float64 GPU continuation

The user explicitly requested float64 GPU testing. The recent compact-table,
descriptor, and static-extent performance trials used float32 GPU arithmetic
and state storage. Their default-double CPU regressions did NOT validate
float64 GPU timing. Those float32 results are retained as historical evidence
and will not be presented as float64 measurements.

The GPU build was reconfigured with BOTH precision options OFF:
`GIZMO_RTZ_FLOAT32_ARITHMETIC=OFF` and
`GIZMO_RTZ_REDUCED_PRECISION=OFF`. This selects `Real=double`,
`CompactIonData::ndens_t=double`, and `xion_t=double`. Tables that were
already float in the original solver (e.g. HM12 and high-T cooling fits)
retain their original types. The build-info/type-size check and explicit
float64 GPU benchmark are recorded below.


Float64 preflight completed: the loaded extension reports execution space
`Cuda`, `real_bytes=8`, `density_bytes=8`, `ion_fraction_bytes=8`.
All four arrays over 10,000 cells are bit-for-bit identical to the original
float64 GPU baseline. One cold solve took 39.249979 s; this is a correctness
preflight, not a warmed performance claim. See
`results/float64-compact-preflight.json`. The warmed measurements are recorded below.

The harness now records the compiled extension's `build_info()` and supports
`--require-float64`, which rejects missing or non-8-byte computation/state
metadata before starting a solve. Subsequent GPU performance runs use this
flag explicitly, alongside `--accuracy exact`.


### Float64 compact-table baseline (verified CUDA build)

Both precision switches are OFF. The binary reports `Cuda`, 8-byte Real,
density, and ion fractions, 2824 bytes per cell, and a 6528-byte table
descriptor. `--require-float64` passed. Five warmed full-dataset samples:
**37.690896, 37.708016, 37.697585, 37.692757, 37.674616 s**; median
**37.692757 s**. Warmup 39.260707 s. All output bit patterns match the
original float64 GPU reference exactly (zero differing values).

This is the baseline for subsequent float64 tuning. It is separate from the
6.523 s float32 descriptor result. Files:
`results/float64-compact-baseline.json` and
`results/float64-compact-preflight.json` include binary/input hashes and
compiled type metadata. The original float64 baseline was 40.818548 s;
current float64 performance is about 1.083x faster. That historical comparison
includes earlier changes; no float64 speedup is inferred from float32 trials.

Reproduce this precision check and timing with:

```sh
cmake -S . -B build/galaxy-gpu \
  -DGIZMO_RTZ_FLOAT32_ARITHMETIC=OFF -DGIZMO_RTZ_REDUCED_PRECISION=OFF
# Build with the existing CUDA toolchain settings documented above.
/usr/bin/python3.14 benchmarks/galaxy_gpu.py \
  --extension build/galaxy-gpu/_rtz.cpython-314-x86_64-linux-gnu.so \
  --output benchmarks/results/float64-compact-baseline \
  --reference benchmarks/results/baseline.npz --accuracy exact --require-float64
```


Float64 CUDA validation: **18 passed**, three long sequential golden sweeps
excluded, in 4.27 s. This run used the actual CUDA binary above, not the CPU
build. The preserved float64 binary is
`build/galaxy-gpu/float64-compact-baseline.cpython-314-x86_64-linux-gnu.so`.
Build and validation metadata: `results/float64-compact-tests.json`.
