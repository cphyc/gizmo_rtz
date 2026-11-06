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


## Float64 best-version kernel deep-dive

Profiled the saved `float64-compact-baseline` binary (SHA recorded in
`results/float64-best-profile.json`), not the uncommitted rate-cache build.
Compiled metadata confirms CUDA and 8-byte computation/density/ion types.
The cold profile completes with output bits identical to the original double
reference. Profiling timing (39.741878 s) is not used as a warmed benchmark.

| Kernel/phase | Launches | GPU seconds | Share of kernel time |
| --- | ---: | ---: | ---: |
| Molecules / H-He / metal ions / commit | 1505 | 34.122123 | 88.0% |
| Cooling rates (two temperature lanes) | 1505 | 2.274776 | 5.9% |
| Prep / radiation | 1505 | 1.129590 | 2.9% |
| Spectrum/cross-section setup | 3 | 1.228774 | 3.2% |
| Pending count | 1505 | 0.013879 | <0.1% |
| Iteration-limit check | 1504 | 0.011245 | <0.1% |
| Active-index scan (multiple passes) | 2150 | 0.010033 | <0.1% |
| Interval initialization / finish | 1 each | 0.002174 | <0.1% |

The setup kernels are first-call work. In repeated solves, the ion/commit
phase accounts for about 90.8% of the listed recurring GPU kernel time.
The current network disables molecules, so the fused dominant phase's
molecular code is compiled out. H/He and commit sections run on one team
thread; metal stages run across the 32-thread team. The ion kernel uses
95 registers/thread in this float64 build. The trace does not separately
time these device subfunctions, so no measured H/He-versus-metal split is
claimed. CUDA synchronization time overlaps kernel execution and is not an
additional cost to sum with it.

Source inspection: adjacent metal stages independently evaluate the same
recombination/collisional coefficients at frozen temperature. Trial08 tests
reusing those coefficients in shared scratch. Further requested experiments
will inspect duplicated exp/pow/log10 within hot functions and integer power
exponents. Candidates will be measured separately; syntactic integer
exponents alone do not help while the current `rtz_math::pow(Real,Real)`
wrapper converts them back to Real. Integer overloads may change rounding
and therefore must pass the exact gate before being retained.

Raw profiling summaries: `results/float64-best-profile_cuda_*_sum.csv`.
The full trace and SQLite export are local ignored artifacts.

### Updated profile after retaining trial 08

The saved `rate-cache-float64` binary is now the best accepted float64
version. Its cold Nsight Systems run takes 34.980819 s and passes the exact
10,000-cell reference comparison. It reports CUDA and 8-byte Real/density/ions.

| Kernel/phase | Launches | GPU seconds | Share of kernel time |
| --- | ---: | ---: | ---: |
| Molecules / H-He / metal ions / commit | 1505 | 29.621790 | 86.4% |
| Cooling rates | 1505 | 2.273153 | 6.6% |
| Prep / radiation | 1505 | 1.128724 | 3.3% |
| Spectrum/cross-section setup | 3 | 1.237610 | 3.6% |
| Pending count | 1505 | 0.013993 | <0.1% |
| Iteration-limit check | 1504 | 0.011364 | <0.1% |
| Active-index scan | 2150 | 0.010048 | <0.1% |
| Interval initialization / finish | 1 each | 0.002242 | <0.1% |

Ionization remains the dominant kernel after caching: 89.6% of recurring
kernel time when the three first-call setup kernels are excluded. Its
profiled total decreased by 4.500333 s (13.2%); cooling and preparation are
essentially unchanged. These are separate cold profiles; use the warmed
trial 08 measurements below for the overall speedup. Further ionization
work has much more potential than launch/reduction/scan work, which together
take under 0.04 s. Cooling optimizations have an overall ceiling of about 7%.

Within the dominant kernel, H/He runs serially per cell, while metal stages
run across team lanes. H/He temperature changes after individual stages, so
carrying recombination/collisional rates across those stages would alter
the calculation. The hydrogen charge-exchange partner sum already reuses
its results between the two H stages. Source inspection also finds
zero-amplitude charge-transfer fits that still evaluate pow/exp; these are
a candidate for a separate exact-output trial. The profile does not measure
these device subfunctions separately.

Artifacts: `results/rate-cache-best-profile.json` and
`results/rate-cache-best-profile_cuda_*_sum.csv`. The full trace and SQLite
export are local ignored files.

## Trial 08: cache neighboring metal rate coefficients (float64)

Two Real scratch arrays cache recombination and collisional coefficients for
metal stages, once per iteration after H/He. Creation/destruction terms read
the same cached coefficient, preserving their existing multiplication and
addition order. Only valid stages are evaluated; the cache respects the
collisional-ionization flag. Temperature/ne used by the metal phase remain
frozen. Existing callers retain an uncached template default; the Python team
path explicitly enables caching. Scratch allocation includes Real alignment
padding and grows by 2*n_ion_work*sizeof(Real). Both float32 options are OFF. Full-dataset preflight passes bit-for-bit.
Median of five warmed runs: **33.199750 s** versus the prior float64 compact
baseline **37.692757 s**, a **1.13533x** speedup (11.92% less time), and
about **1.22948x** versus the original 40.818548 s baseline. Samples:
33.207808, 33.199750, 33.195480, 33.186653, 33.209149 s.
All returned bits are identical to the original float64 GPU reference.
Default-double CPU regressions: 18 passed (three long sweeps excluded).
Float64 CUDA regressions: 18 passed (three long sweeps excluded). **Retained**.
Files: `results/rate-cache-gpu.json`, `results/rate-cache-preflight.json`,
`results/rate-cache-tests.json`. No solver tolerances or rates changed.


Utilization note: the user observed approximately 56% effective load in
nvtop. This is not equated with occupancy or a count of idle launch gaps.
In the best-version trace, the union of kernel intervals is 38.793721 s over
39.179030 s from first kernel start to last kernel end (99.02% activity).
`nvidia-smi dmon` samples during the cached float64 solve report 100% SM
activity and 1–2% memory activity (`results/rate-cache-utilization.txt`).
These coarse activity metrics do not prove full-lane or DP-pipeline
utilization. Single-thread team sections and FP64 throughput remain plausible
limits; hardware/source counters are needed to quantify them.

The PROMISE exploration runs separately in `~/tmp/gizmo-rtz-promise`, using
CPU-only float32/float64 selection with float32 tables/constants. It does not
change or supply timings for the main float64 optimization branch. `/tmp` is
RAM-backed on this machine; persistent experiments use `~/tmp`.

## Trial 09: integer-exponent overload (float64)

Added `rtz_math::pow(Real,int)` forwarding to `::pow(x,y)` so existing calls
such as `pow(two_e4_overT,5)` do not first convert the exponent in our wrapper.
Full dataset remains bit-for-bit identical to the original float64 GPU
reference. Five warmed solves: 33.200495, 33.209230, 33.189877, 33.187075,
33.191554 s; median **33.191554 s**, versus **33.199750 s** for trial 08.
The 0.025% difference is noise, so the overload is **not retained**.

A direct CUDA probe also compares `::pow(x,3.0)` with `::pow(x,3)` on the
10,000 input temperatures. Both generate calls to the same accurate power
routine, produce identical bits, and take about 0.053 ms per launch. With
this NVCC/GNU C++20 setup, an integer literal alone does not select an integer
power implementation. Explicit products require a separate rounding check;
algebraic equality alone does not satisfy the exact-output requirement.

Artifacts: `trial09-integer-power.patch`, `pow_integer_probe.cu`,
`results/integer-power-preflight.json`, `results/integer-power-gpu.json`,
and `results/pow3-integer-probe.json`. Both solver precision switches remain
OFF, and loaded computation/density/ion widths are all 8 bytes.

Explicit cube products were also tested on those inputs: about 0.00406 ms
per launch versus 0.05261 ms for pow, but **3,469/10,000** results differ in
their bits (maximum relative difference 3.04e-16). This faster algebraic
rewrite does not meet the exact no-op requirement and is not applied to
solver expressions. See `results/pow3-integer-products-probe.json`.

## Trial 10: factor repeated math within functions (float64)

Tested explicit reuse of `pow(T3,1.50)` in H2 cooling,
`exp(-T_pivot/T)` in CO cooling, the lower-node `log10` values in both
CO-shielding interpolations, and `sqrt(T/T0)` in alpha_RR. Multiplication
and interpolation order remain unchanged. Default-double CPU regressions:
18 passed, three long sweeps excluded (1.25 s). Full CUDA dataset passes
the exact original float64 reference comparison (one cold solve 34.815228 s).

These are outside the dominant atomic ionization calculation. The dataset
disables molecular evolution, although all_cooling still calls the molecular
cooling functions with zero molecular densities. Alpha_RR is currently
unused (radiative rates use cloudy_rad_rec). The selected template's ionization and prep/radiation
assembly are unchanged; the cooling assembly changes. No overall speedup
is claimed from a cold correctness run. The source experiment is preserved
as `trial10-repeated-math.patch`, not retained in the atomic optimization.
The patch has zero context; apply with `git apply --unidiff-zero`.
Artifacts: `results/factor-math-preflight.json`,
`results/factor-math-sass.json`. This records actual tested factoring rather
than treating fewer source-level function calls as proof of faster code.

## Trial 11: run sequential H/He across cell lanes (float64)

Move cooling update, molecules, and the sequential H/He sweep from the
single-thread section of each team into a preceding RangePolicy kernel.
Each lane then works on a different cell. Metal stages and commit retain
their existing team kernel and scratch cache. Launch order on the same
execution stream preserves the H/He-before-metals dependency. Within-cell
operation order and Gauss-Seidel updates are unchanged. The experiment
adds one launch per subcycle to avoid idle lanes in the serial phase.

Full 10,000-cell preflight and all timed runs match the original float64 GPU
reference **bit-for-bit** in temperature, densities, ion fractions, and
photons. Computation/state widths are 8 bytes; both precision switches OFF.
Five warmed samples: **14.149835, 14.133068, 14.130250, 14.127166,
14.109511 s**; median **14.130250 s**, warmup 15.779326 s.
Versus trial 08's 33.199750 s this is **2.34955x faster**, or **57.44% less
time**. Versus the original 40.818548 s baseline it is **2.88873x faster**.
The cold preflight is 15.716552 s and is not included in that median.
Default-double CPU regressions: 18 passed, three long sweeps excluded (1.55 s).
Float64 CUDA regressions: 18 passed, three long sweeps excluded (4.43 s).
**Retained**. No rates, solver tolerances, or physics flags changed.

Compiled resources for this template: H/He 92 registers/thread; metal/commit
88 (previous fused ion kernel 95); cooling 130; prep/radiation 64. All these
kernels report zero local memory and stack. These are compilation resource
counts, not measured occupancy. The main improvement comes from assigning
serial per-cell work across lanes; no occupancy claim is inferred from the
register decrease.

Artifacts: `results/split-hhe-gpu.json`, `results/split-hhe-preflight.json`,
`results/split-hhe-resources.json`, and `results/split-hhe-tests.json`. Preserved binary:
`build/galaxy-gpu/split-hhe-float64.cpython-314-x86_64-linux-gnu.so`.

### Current best: per-kernel profile after trial 11

One cold Nsight Systems solve completes in 16.098811 s and passes the exact
original float64 reference. This profile is separate from warmed timing.

| Kernel/phase | Launches | GPU seconds | Share of kernel time |
| --- | ---: | ---: | ---: |
| Metal ions / commit | 1505 | 9.035267 | 59.2% |
| Cooling rates | 1505 | 2.268845 | 14.9% |
| Cooling update / molecules / H-He | 1505 | 1.496725 | 9.8% |
| Prep / radiation | 1505 | 1.127635 | 7.4% |
| Spectrum/cross-section setup | 3 | 1.288043 | 8.4% |
| Pending count | 1505 | 0.013524 | 0.1% |
| Iteration-limit check | 1504 | 0.011236 | 0.1% |
| Active-index scan | 2150 | 0.010059 | 0.1% |
| Interval initialization / finish | 1 each | 0.002233 | <0.1% |

The formerly fused ion phase took 29.621790 s in trial 08. The two replacement
kernels total **10.531992 s**, a 2.81x reduction in profiled ion-phase time.
Cooling and prep remain essentially unchanged. Separating the sequential
cell work addresses the dominant phase without changing its arithmetic.
The split also changes code generation/register allocation, so the old
H/He subphase's isolated time cannot be inferred by subtracting kernel totals.

**The remaining dominant kernel is metal ions / commit**, about 64.7% of
recurring kernel time excluding first-call spectrum setup. It still contains
single-thread commit work and repeated charge-transfer coefficients across
neighboring stages. Potential next experiments are caching charge-transfer
coefficients at the frozen metal temperature, skipping zero-amplitude fits,
and measuring commit separately before deciding whether another split helps.
Hardware counters would distinguish DP instruction throughput, latency, and
lane underuse; this timeline alone does not establish those limits.
Cooling is now a meaningful second target (16.2% of recurring time). It
continues to evaluate H2/CO cooling with zero molecular densities, which
could be tested with exact floor-preserving early exits. None of these
additional hypotheses is presented as a measured improvement.

Artifacts: `results/split-hhe-best-profile.json` and
`results/split-hhe-best-profile_cuda_*_sum.csv`. Full trace and SQLite export
remain local ignored artifacts.

PROMISE restart checkpoint: the separate CPU-only experiment now uses the
user-requested GPT-6.1 Sol medium agent. It annotates all Real occurrences
across the solver rather than one selected rate function. Table storage and
constants are forced float32 in that experiment. The untouched-double CPU
control passes the full physical
oracle against the original GPU reference (maximum ion error 2.03e-10,
temperature 9.00e-13). Precision selection remains separate and in progress;
no candidate from it is applied to the main float64 benchmark.

### Remaining cell-count imbalance after trial 11

The metal grid has exactly one block per active cell. Grouping the current
best trace by grid size gives:

| Active cells | Launches | Metal GPU seconds |
| --- | ---: | ---: |
| 1–14 | 72 | 0.013376 |
| 15–128 | 290 | 0.078167 |
| 129–1000 | 556 | 0.849717 |
| 1001–10000 | 587 | 8.094007 |

Compaction executes 2,389,309 cell-iterations versus 15,050,000 without
compaction (84.1% fewer). The <=128-cell tail is only 0.091544 s, about 1%
of metal kernel time. Most remaining time is spent while over 1000 cells
are active. This does not rule out unequal per-cell work within each launch,
but shows that optimizing only the final sparse tail has limited potential.
Artifact: `results/split-hhe-load-balance.json`.

## Trial 12: skip zero-amplitude charge-transfer fits (float64)

After the existing stage guards, return the existing 1e-100 rate floor when
the fit amplitude is zero. The original zero product is either zero or NaN
if an unused pow/exp overflows; fmax returns the same positive floor in both
cases. Nonzero fits and high-stage statistical rates retain their original
calculation. Full float64 accuracy and timing gates determine retention.


Five warmed candidate samples: 13.870679, 13.837078, 13.828415, 13.830928,
13.828767 s; median **13.830928 s**. A fresh saved-trial-11 baseline recheck
under the same conditions gives 14.118373, 14.127240, 14.128673, 14.113867,
14.088809 s; median **14.118373 s**. This is **1.02078x faster (2.04% less
time)**, about 2.951x faster than the original 40.818548 s baseline.
Both full-dataset runs pass bit-for-bit against the original float64 output;
compiled Real/density/ion widths are all 8 bytes. CPU regressions: 18 passed,
three long sweeps excluded (1.46 s). CUDA regressions: 18 passed, three long
sweeps excluded (3.99 s). **Retained**. Both precision switches remain OFF.
Artifacts: `results/zero-ct-gpu.json`, `results/zero-ct-baseline-recheck.json`,
`results/zero-ct-tests.json`. Saved binary: `build/galaxy-gpu/zero-ct-float64.cpython-314-x86_64-linux-gnu.so`.

Current-best cold profile (15.695882 s, exact output) identifies metal/commit
as still dominant: **9.062119 s / 60.6%** of kernel time. Cooling is
2.114452 s / 14.2%, H/He 1.347129 s / 9.0%, prep 1.127761 s / 7.5%.
Compared with trial 11, savings occur in cooling and H/He; metal time is
essentially unchanged (9.035267 s previously). Thus this trial improves the
full solve but does not demonstrate faster metal execution. Profile artifacts:
`results/zero-ct-profile.json`, `results/zero-ct-profile_cuda_gpu_kern_sum.csv`.
The binary still uses the same active-cell compaction and kernel split.

## Trial 13: cache metal charge-transfer coefficients (float64)

Extend the existing per-stage recombination/collisional cache with two
charge-transfer arrays at the frozen post-H/He metal temperature. Neighboring
stages reuse coefficients while retaining the multiplication and accumulation
order of their creation/destruction terms. Only valid stages are evaluated,
and the charge-exchange flag gates all new calculations. The uncached template
default remains available; Python enables this experiment explicitly.
Scratch grows by 2*n_ion_work*sizeof(Real), so occupancy costs could offset
less repeated math. Full-dataset exact comparison and timing decide retention.


Five warmed candidate samples: 13.217129, 13.219614, 13.177843, 13.161342, 13.174008 s; median **13.177843 s**. Fresh saved-trial-12 baseline samples: 13.843497, 13.856950, 13.850062, 13.825967, 13.850594 s; median **13.850062 s**. This is **1.05101x faster (4.85% less time)**, or **3.09751x** versus the original baseline. Both full-dataset runs pass the original float64 reference bit-for-bit. Loaded computation/density/ion widths remain 8 bytes. CPU regressions: 18 passed (1.28 s); CUDA: 18 passed (3.99 s); three long sweeps excluded in each.

**Retained**. Current-best cold profile completes in 15.136843 s and also
passes exact float64 comparison. Metal/commit takes **8.401891 s / 58.9%**,
down from trial 12's 9.062119 s (7.29% less profiled metal time). Cooling
2.111844 s, H/He 1.345022 s, and prep 1.128690 s remain essentially unchanged.
Thus this trial directly improves the dominant kernel. Compiled metal
registers decrease from 88 to 82/thread, with zero local memory and stack;
this does not itself prove improved occupancy. Shared scratch increases as
noted above. Saved binary: `build/galaxy-gpu/ct-cache-float64.cpython-314-x86_64-linux-gnu.so`.
Artifacts: `results/ct-cache-gpu.json`, `results/ct-cache-baseline-recheck.json`,
`results/ct-cache-tests.json`, `results/ct-cache-profile.json`, and
`results/ct-cache-profile_cuda_gpu_kern_sum.csv`. Both precision switches OFF.

## Trial 14: current code with float32 computation/state

User-requested precision check of revision 156ba01, both precision options
ON. Loaded CUDA metadata confirms 4-byte Real/density/ion fractions, 1424
bytes/cell. Existing table field types follow their precision configuration;
this is the full-float32 arithmetic option, not the separate PROMISE table32
experiment. Five warmed solves: 3.464247, 3.456701, 3.452585, 3.453469,
3.456014 s; median **3.456014 s**, warmup 4.999109 s. This is **3.813x faster**
than the current float64 median 13.177843 s.

Full 10,000-cell solve completes, but fails physical accuracy limits against
the original float64 reference: among 203,932 ion values with reference
abundance >1e-4, 17 exceed 5% relative error; maximum **11.7698%**. One
cell exceeds the 1% temperature tolerance; maximum **1.22227%**. Maximum
absolute error among smaller ions is 6.42621e-6. This remains experimental;
no precision change is retained in main. Both build options restored OFF,
and default extension restored from the saved current float64 best binary.
Float32 binary preserved as `build/galaxy-gpu/current-float32.cpython-314-x86_64-linux-gnu.so`.
Results: `results/current-float32.json` (accuracy failure is intentional
benchmark exit status 1; timings/output were saved).

## Per-file precision, step 1: collisional ionization

Add GIZMO_RTZ_FLOAT32_COLLISIONAL and an independent CollisionalReal scalar
for device arithmetic in collisional_ionization.hpp, with explicit float
math overloads. Keep global Real, state, table storage and cooling unchanged.
The Python extension uses the same definition and reports
collisional_ionization_bytes=4 alongside Real/density/ion bytes=8.
Five warmed GPU samples: 12.920085, 12.926781, 13.064292, 12.967459,
12.942590 s; median **12.942590 s** (prior float64 best 13.177843 s;
about 1.01818x, 1.79% less time; no fresh-baseline recheck yet).
Full physical gate passes: max ion relative error **0.0714333%**, max
T relative error **0.000985753%**, zero violations. Artifact:
results/per-file-collisional.json. The 18 selected CUDA regressions yield
17 passed and one strict float64 golden failure (max relative difference
1.204e-6 against its 1e-8 tolerance), consistent with intentional precision
rounding. That test's golden tolerance is not weakened. Cooling remains last
in the conversion sequence. This step changes arithmetic only; existing
host loading buffers and table storage retain their original types.

Precision-gate update from user: require **1e-5 relative** for the full dataset
(ions above reference abundance 1e-4 and temperature), as well as the golden
regression. Step 1's collisional arithmetic is therefore **rejected for the
selected configuration**: max ion error 7.14333e-4 exceeds this new limit;
max temperature error 9.85753e-6 passes narrowly. The optional implementation
remains available, but its switch will be OFF for subsequent independent
trials. Benchmark now accepts --temperature-rtol so both limits are explicit.
Charge transfer is tested independently with collisional arithmetic float64.

## Per-file precision, step 2: charge transfer

Charge-transfer arithmetic uses ChargeTransferReal=float and float math,
with explicit casts at table reads (the underlying CT tables were already
float32, so casts do not change the generated binary). Collisional arithmetic
is restored to double. GIZMO_RTZ_FLOAT32_CHARGE_TRANSFER is now ON by default;
set it OFF along with the other precision switches to reproduce all-double
arithmetic. rtz_bindings reports charge_transfer_bytes=4 and
collisional_ionization_bytes=8; Real/density/ion storage remain 8 bytes.
CMake applies the per-file definitions to standalone and Python targets.

Five warmed samples: 11.794138, 11.795324, 11.789451, 11.783373,
11.794311 s; median **11.794138 s**, about **1.11732x faster (10.50% less
time)** than the 13.177843 s float64 best. Full strict physical gate passes:
max ion error **3.70191e-6**, max temperature error **4.42555e-9**, zero
violations at rtol=1e-5. Selected CUDA regressions: 18 passed (3.76 s),
three long sweeps excluded. CPU CI-only regression with the revised golden
tolerance: 18 passed (0.95 s). Charge-transfer-only CPU checks follow below.
Artifacts: results/per-file-charge-float32.json; the earlier cold preflight
is results/per-file-charge-only.json. Both have the same binary hash.
Saved binary: build/galaxy-gpu/per-file-charge-float32.cpython-314-x86_64-linux-gnu.so.

The revised collisional gate, recomputed from its saved outputs, has 159 ion
violations and zero temperature violations: results/per-file-collisional-strict.json.
The collisional switch stays OFF. No other physics file has yet been changed.
Charge-transfer-only CPU regressions: 18 passed, three long sweeps excluded (0.84 s).

## Per-file precision, step 3: recombination

Add RecombinationReal with explicit float math and coefficient conversion,
plus GIZMO_RTZ_FLOAT32_RECOMBINATION and recombination_bytes in rtz_bindings.
Trial keeps accepted charge transfer float32 and all other files float64.
Full-dataset cold preflight: 9.724720 s, **fails** at rtol=1e-5: maximum ion
relative error 1.86017e-3, 130 violations; temperature maximum 9.46282e-6,
zero violations. No warmed speedup claim for this rejected precision.
Recombination remains float64 in the selected configuration (option OFF).
Artifact: results/per-file-recombination.json. Cooling remains unchanged.

## Per-file precision, step 4: secondary cosmic-ray arithmetic

CosmicRayReal is independently selectable; cosmic_ray_bytes in rtz_bindings
confirms 4 bytes. Accepted charge transfer stays float32; collisional and
recombination calculations stay float64. Full-dataset strict preflight passes:
max ion relative error 3.70191e-6; temperature 4.42555e-9; zero violations at
1e-5. Cold solve 13.361736 s; no warmed speedup claim for this step.
Cosmic-ray arithmetic switch ON by default. Host rate-loading buffers retain
their original types. Artifact: results/per-file-cosmic.json.

## Per-file precision, step 5: dust recombination

DustRecombinationReal=float, with float math and explicit coefficient casts;
dust_recombination_bytes=4 is reported by rtz_bindings. Charge-transfer and
cosmic-ray arithmetic remain float32; collisional/recombination and state
stay float64. Full 10,000-cell gate passes at 1e-5: max ion 3.70171e-6,
max temperature 4.42484e-9, zero violations. Cold preflight 12.232884 s;
no warmed speedup claimed for this step alone. Dust option ON by default.
Artifact: results/per-file-dust.json.

Benchmark precision guard correction: --require-float64 now checks every
reported arithmetic width, not only global Real/state. Mixed-file runs use
--require-float64-state. The negative guard was verified on a saved mixed
binary before any solve; full-float64 mode correctly rejects it. For earlier
per-file result commands, replace --require-float64 with --require-float64-state.

## Per-file precision, step 6: molecular helpers

MolecularReal=float with float math and coefficient casts; molecular_bytes=4
is reflected in rtz_bindings. Full atomic dataset passes 1e-5: max ion
3.70171e-6, T 4.42484e-9 (cold 12.258382 s). Since that dataset disables
molecules, an additional 42-cell T/nH-stratified probe enables H2/CO and
self-shielding. Its matching all-float64 reference is generated with the saved
ct-cache-float64 binary. Before this conversion, accepted mixed arithmetic
passes that probe (max ion 2.30522e-7, T 4.28625e-10). After conversion it
still passes: max ion 2.30522e-7, T 4.30419e-10, zero violations. Molecular
option ON by default; no warmed speedup claim from these cold checks.

Reproduce probe inputs with benchmarks/molecular_precision_dataset.py
--output benchmarks/results/molecular-precision-input.npz. Artifacts:
results/molecular-precision-reference.json, results/molecular-precision-before.json,
results/per-file-molecules.json, results/per-file-molecules-probe.json.
The probe covers these sampled cells and 100 years, not all molecular regimes.

## Per-file precision, step 7: UV background

UVBReal=float converts photoionization_UVB.hpp's loading buffers and redshift
interpolation together. fscanf uses a matching %f/%lf format selected with
the type. uvb_bytes=4 is reported in rtz_bindings; this original file used
explicit double, so OFF retains double even under the global Real option.
Full-dataset 1e-5 gate passes: max ion 3.70568e-6, temperature 4.41734e-9,
zero violations (cold 12.243331 s). UVB option ON by default. No warmed
speedup claimed for this setup-only change. Artifact: results/per-file-uvb.json.

## Per-file precision, step 8: spectrum/cross-section integration

CrossSectionReal=float applies to device spectrum/quadrature/cross-section
arithmetic and its temporary views; public SED input vectors remain double.
Physical constants, fit coefficients, and SED values are converted before
arithmetic. Float32 blackbody uses a reordered prefactor avoiding nu^3
overflow; double keeps the original expression. cross_section_bytes=4 is
reported in rtz_bindings. Original table-loading buffers remain double.

Full 10,000-cell preflight completes in 11.255806 s but **fails** 1e-5:
max ion error 2.01018e-3 (31,381 violations), T 7.81806e-5 (2,007 violations).
All outputs are finite. Cross-section option stays OFF; no warmed speedup
claim. Artifact: results/per-file-cross.json. The benchmark now rejects
non-finite values explicitly; all nine earlier per-file snapshots were
verified finite as well.
