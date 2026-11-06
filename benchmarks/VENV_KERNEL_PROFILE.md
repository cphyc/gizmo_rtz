# Installed venv float64 solver profile

Exact extension: `.venv/lib/python3.13/site-packages/rtz/_rtz.cpython-313-x86_64-linux-gnu.so` (SHA256 `0a1d3908fca85c6b9f69bd6f8cb70df0385fc6fe8ac5323d6468c1c6faf0434d`). All reported arithmetic/state widths are 8 bytes. Installed wrapper selected with `--installed-package`; no rebuild.

Workload: 10,000 cached galaxy cells, 100 yr. One warmup then one measured solve under Nsight Systems. Warmup 14.799212 s, measured wall 13.104266 s. Warm-only kernel sum 12.993859 s. Profiling timings are not a five-run performance benchmark.

| Part | Launches | GPU s | GPU % | Registers/thread | CUDA block shape | Dynamic shared B |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Metals / commit | 1505 | 8.399926 | 64.65 | [82] | [[1, 32]] | [2815] |
| Cooling rates | 1505 | 2.077167 | 15.99 | [128] | [[1, 128]] | [0] |
| Cooling update / molecules / H-He | 1505 | 1.347653 | 10.37 | [92] | [[1, 128]] | [0] |
| Preparation / radiation | 1505 | 1.131948 | 8.71 | [64] | [[1, 128]] | [0] |
| Pending count | 1505 | 0.013544 | 0.10 | [25] | [[1, 256]] | [2064] |
| Iteration-limit check | 1504 | 0.011278 | 0.09 | [25] | [[1, 256]] | [1032] |
| Active-cell compaction scan | 2150 | 0.010076 | 0.08 | [56] | [[1, 128]] | [1040] |
| Interval initialization | 1 | 0.001341 | 0.01 | [64] | [[1, 128]] | [0] |
| Final output | 1 | 0.000925 | 0.01 | [63] | [[1, 128]] | [0] |
| Descriptor destruction | 1 | 0.000001 | 0.00 | [16] | [[1, 128]] | [0] |

Block shapes show x/y (Kokkos teams use y). All listed kernels report zero local-memory bytes/thread in the trace; this alone is not a measured spill-traffic result. Grid sizes shrink with active-cell compaction; ranges and full demangled names are in `results/venv-float64-kernel-breakdown.json`.

Metals/commit is the primary target. Cooling has the largest register count (128/thread). H/He has 92/thread. Prep has 64/thread. These resource counts do not establish measured occupancy or the instruction/memory bottleneck. One-warp teams can hit the resident-block cap before the warp cap; shared-memory partitioning also matters. Nsight Compute Occupancy/SchedulerStats/WarpStateStats/MemoryWorkloadAnalysis/SpeedOfLight sections are required to quantify these limits.

Pending count, iteration-limit checks and scans together cost about 0.035 s: removing these launches alone has little total-time potential. The fused metal/commit kernel does not separately expose commit cost. CUDA API synchronization time overlaps GPU execution and must not be added to kernel time.

The trace includes first-call spectrum setup and warmup; those are excluded by selecting kernels from the second interval initialization onward. Raw whole-trace CSV summaries include both solves and must not be mistaken for the warm-only table.

Nsight Compute hardware counters were subsequently captured with sudo. The completed analysis follows below.

Artifacts: `results/venv-float64-profile.nsys-rep` and SQLite export (local), `results/venv-float64-profile_cuda_gpu_kern_sum.csv`, `results/venv-float64-profile_cuda_api_sum.csv`, and the warm-only breakdown JSON.


## T11 kernel identity

T11 moved cooling update + molecular work + the sequential H/He sweep out
of the metal team kernel into its own RangePolicy launch. In this workload
H2/CO are disabled. The logical Kokkos label is `rtz_python_molecules_hhe`
(rtz_bindings.cpp:803); CUDA displays the enclosing generated lambda:
`run_solver<0,0,0,1,0> ... lambda(unsigned long) (instance 4)`.
It is separate from `rtz_python_metals_commit` (line860), displayed as the
CudaTeamMember lambda. H/He is sequential within each cell, while different
cells run across threads. The 1.347653s NSYS total includes cooling update;
it is not an isolated device-function timing of H/He alone.

## Nsight Compute: first invocation of every distinct kernel

The compressed report is `results/venv-float64-ncu.ncu-repz`; detailed output
and parsed section metrics are `results/venv-float64-ncu-details.txt` and
`results/venv-float64-ncu-metrics.json`. Extension SHA matches the NSYS run.
NCU wall45.642865s includes counter replay, instrumentation and cold setup;
it is not a production runtime or a regression versus the warm13.104266s.
NCU sample durations below are individual first launches, not full-solve totals.

| Kernel | First launch | Theoretical / achieved occupancy % | SM throughput % | Memory throughput % | Eligible warps/scheduler |
| --- | ---: | ---: | ---: | ---: | ---: |
| Architecture query | 1.57 us | 50 / 3.04 | 0.01 | 1.87 | 0.01 |
| Atomic-lock initialization | 42.78 us | 100 / 73.27 | 3.06 | 8.64 | 0.03 |
| Spectrum groups (setup) | 105.09 ms | 100 / 3.13 | 1.92 | 0.05 | 0.06 |
| Cross-section quadrature (setup) | 1.17 s | 50 / 6.99 | 27.86 | 0.19 | 0.04 |
| H2 cross sections (setup) | 73.84 ms | 100 / 3.13 | 2.22 | 0.02 | 0.06 |
| Cell construction | 1.19 ms | 100 / 58.94 | 1.10 | 57.91 | 0.01 |
| Descriptor construction | 29.57 us | 25 / 3.41 | 0.33 | 1.52 | 0.09 |
| Interval init | 1.26 ms | 100 / 68.70 | 16.65 | 45.88 | 0.01 |
| Prep / radiation | 5.61 ms | 100 / 69.05 | 6.54 | 41.89 | 0.02 |
| Cooling rates | 7.81 ms | 50 / 47.07 | 78.87 | 3.54 | 0.06 |
| **H/He + cooling update** | 3.86 ms | 62.50 / 54.98 | 62.90 | 18.94 | 0.05 |
| Metals / commit | 20.00 ms | 50 / 43.38 | 84.40 | 22.73 | 0.06 |
| Pending count | 18.72 us | 100 / 76.71 | 9.11 | 35.49 | 0.18 |
| Iteration-limit check | 21.95 us | 100 / 82.01 | 7.61 | 38.28 | 0.17 |
| Compaction scan | 16.64 us | 100 / 76.79 | 6.08 | 41.21 | 0.11 |
| Final output | 909.25 us | 100 / 68.36 | 13.96 | 45.07 | 0.01 |
| Descriptor destruction | 1.22 us | 100 / 12.37 | 0.02 | 1.97 | 0.01 |

### Interpretation and priorities

- **Metals/commit (64.65% of warm GPU time):** 82 registers, 32-thread blocks.
  Occupancy is block-count limited: SM block limit16, register block limit20,
  shared-memory limit21, warp limit32. Thus reducing registers alone cannot
  raise the 50% theoretical occupancy ceiling at this launch shape. Achieved
  occupancy43.38%. Only11.53 active /11.02 non-predicated threads per warp;
  0.06 eligible warps/scheduler. NCU attributes70.8% of inter-issue warp cycles
  to long-scoreboard L1TEX dependencies. L2 hit98.87%, DRAM throughput0.66%:
  this is not evidence of saturated external-memory bandwidth. SM throughput
 84.40% is an aggregate throughput indicator, not proof that all compute units
  or specifically FP64 are saturated. Test64-thread metal teams as a separate
  hypothesis to increase resident warps; also examine dependent table/offset
  loads and stage divergence. A measured test is required before claiming gain.
- **Cooling rates (15.99%):** 128 registers/thread; registers limit residency
  to4 blocks,50% theoretical occupancy,47.07% achieved. SM throughput78.87%
  versus memory3.54%. Prioritize redundant compute and register lifetime;
  reducing registers helps only if it crosses an allocation/residency boundary.
  ComputeWorkloadAnalysis/InstructionStats would identify the particular busy
  arithmetic pipelines; these sections were not collected here.
- **H/He + cooling update (10.37%):** separate RangePolicy kernel,92 registers,
 5 resident-block register limit,62.5% theoretical /54.98% achieved occupancy.
  SM62.90%, memory18.94%, only0.05 eligible warps/scheduler. NCU reports a
  long-scoreboard dependency contribution77.6%. Per-cell data locality,
  independent work to hide latency, and temperature/branch grouping are
  hypotheses; this does not undo T11's large measured improvement.
- **Prep/radiation (8.71%):** 64 registers,100% theoretical /69.05% achieved
  occupancy, SM6.54%, memory41.89%. NCU long-scoreboard contribution87.4%.
  This points toward data-access latency/poor coalescing rather than register
  residency as the first target; inspect AoS cell accesses and cache reuse.
- **Pending reduction, limit reduction and scan:** theoretical100% occupancy;
  achieved76.71%,82.01%,76.79%. Small first-launch durations and only about
 0.035s combined warm total make them low priorities. Scan sampling covers
  the first invocation, not every internal pass or late sparse grid.
- **Interval init / final output:** theoretical100%, achieved68.70% /68.36%;
  memory throughput45.88% /45.07%, long-scoreboard87.4% /90.0%. One launch
  each and about0.0023s total in NSYS: optimizing them barely affects the solve.
- **Setup spectrum groups and H2 integration:** each runs only one128-thread
  block; achieved occupancy3.13%, so most SMs are idle. Parallelizing the
 10001-point quadrature within/across groups could improve cold startup.
- **Setup cross-section quadrature:** 176 blocks of32 threads,80 registers;
  theoretical50% but achieved6.99%, SM27.86%, memory0.19%. The large
  theoretical/achieved gap merits examining unequal element/ion integration
  work and inactive/zero-threshold pairs. Cold-only work: do not count it as
  a warmed-solver bottleneck.
- **Cell construction:** memory throughput57.91%, achieved occupancy58.94%,
  low SM1.10%; initialization traffic, not recurring chemistry.
- **Descriptor construction/destruction, architecture query, lock init:** tiny
  or one-time kernels with low-grid utilization. Descriptor construction uses
 255 registers but one block; that fact does not make it a material solver
  bottleneck. Atomic-lock initialization achieved73.27% occupancy but occurs
  only at initialization.

All sampled kernels report zero local/shared spilling requests. That is a
counter observation for these sampled launches, not a universal guarantee.
Stall statistics are aggregate; no source-PC localization was captured. The
sampled first iteration is not representative of all later temperatures,
active-cell counts or branch distributions. Optimization priorities above
combine these snapshots with the full warmed NSYS timing distribution.
