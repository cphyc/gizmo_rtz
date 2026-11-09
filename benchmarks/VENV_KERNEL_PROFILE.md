# Installed venv float64 solver profile (updated after commit split)

Installed CUDA extension SHA256 `0bb1c6e126be102bb3f4dd5e034b382e051dc456a36dcb87d00f9dd27eef0e06`; identical in NSYS and NCU captures. Every reported arithmetic/state width is 8 bytes. Dataset SHA `b8c7f9d67dba1811212d101dce7b99b3dd6d339dd7e6189dabd07a03889475cd`; 10,000 cells, 100 yr.

Warmed NSYS solve **12.309391s**, GPU kernel sum **12.204497s**. The table excludes warmup and setup. Fresh unprofiled five-run median: 12.267860 s (`metal-normalize-baseline.json`).

| Part | Launches | GPU s | % | Registers/thread |
| --- | ---: | ---: | ---: | ---: |
| Metals | 1505 | 7.136411 | 58.47 | [68] |
| Cooling rates | 1505 | 2.076791 | 17.02 | [128] |
| HHe/cooling update | 1505 | 1.346249 | 11.03 | [92] |
| Prep/radiation | 1505 | 1.131231 | 9.27 | [64] |
| Commit/control | 1505 | 0.477142 | 3.91 | [64] |
| Pending count | 1505 | 0.013448 | 0.11 | [25] |
| Iteration-limit check | 1504 | 0.010990 | 0.09 | [25] |
| Compaction scan | 2150 | 0.010000 | 0.08 | [56] |
| Interval init | 1 | 0.001330 | 0.01 | [64] |
| Final output | 1 | 0.000903 | 0.01 | [63] |
| Descriptor destruction | 1 | 0.000001 | 0.00 | [16] |

## Kernel names and measured counters

H/He remains RangePolicy lambda instance 4. Commit/control is now its own RangePolicy lambda instance 5; final output moves to instance 6. Metals remains the CudaTeamMember lambda, but contains no commit body. Logical labels are `rtz_python_molecules_hhe`, `rtz_python_metals`, `rtz_python_commit_control`. H2/CO are disabled in the main workload.

NCU wall time 45.190994s includes replay/instrumentation and cold setup; it is not solver performance. Counters below are first-invocation snapshots, not full-solve weighted averages.

| Kernel | Duration | Occupancy theoretical / achieved % | Registers | SM throughput % | Memory throughput % | Active threads/warp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Architecture query | 1.54 us | 50 / 3.11 | 16 | 0.01 | 1.58 | 1 |
| Lock initialization | 52.16 us | 100 / 99.79 | 16 | 3.08 | 7.06 | 32 |
| Setup spectrum | 105.11 ms | 100 / 3.13 | 56 | 1.93 | 0.05 | 6.52 |
| Setup quadrature | 1.17 s | 50 / 7.00 | 80 | 27.96 | 0.19 | 7.01 |
| Setup H2 | 73.83 ms | 100 / 3.13 | 56 | 2.22 | 0.02 | 5.79 |
| Cell construction | 1.08 ms | 100 / 49.98 | 18 | 1.06 | 63.95 | 31.95 |
| Descriptor construction | 29.63 us | 25 / 3.41 | 255 | 0.33 | 1.50 | 1.28 |
| Interval init | 1.23 ms | 100 / 68.63 | 64 | 16.81 | 47.25 | 31.95 |
| Prep/radiation | 5.57 ms | 100 / 68.47 | 64 | 6.53 | 42.12 | 31.92 |
| Cooling rates | 7.80 ms | 50 / 47.09 | 128 | 78.93 | 3.55 | 26.07 |
| HHe/cooling update | 3.86 ms | 62.50 / 56.26 | 92 | 63.14 | 19.23 | 19.88 |
| Metals | 17.56 ms | 50 / 48.50 | 68 | 85.20 | 22.48 | 13.07 |
| Commit/control | 578.08 us | 100 / 68.94 | 64 | 28.88 | 31.42 | 10.57 |
| Pending count | 18.40 us | 100 / 74.63 | 25 | 9.14 | 36.70 | 27.72 |
| Iteration-limit check | 21.38 us | 100 / 80.02 | 25 | 7.73 | 38.10 | 27.60 |
| Compaction scan | 16.03 us | 100 / 75.83 | 56 | 6.15 | 43.71 | 27.64 |
| Final output | 894.98 us | 100 / 68.41 | 63 | 14.21 | 45.81 | 31.95 |
| Descriptor destruction | 1.18 us | 100 / 12.60 | 16 | 0.02 | 1.99 | 32 |

## Interpretation

- Metals is still dominant. Its register count dropped 82→68 and achieved occupancy 43.38→48.50% after commit separation, near the unchanged 50% block-count ceiling. Average active lanes rose 11.53→13.07, but utilization remains poor. Only 0.06 eligible warps/scheduler; high aggregate SM throughput 85.20% does not prove a particular pipeline is saturated. This motivated the normalization trial below.
- Cooling is register-limited: 128 registers/thread, 50% theoretical / 47.09% achieved occupancy. SM throughput 78.93%, memory 3.55% in the snapshot: inspect compute redundancy/register lifetime before treating it as an external-bandwidth problem.
- H/He remains separately measured: 92 registers, 62.5% theoretical / 56.26% achieved occupancy; 19.88 active lanes. Prep has 64 registers and 100% theoretical / 68.47% achieved occupancy, but only 0.02 eligible warps/scheduler. Data latency/coalescing remains a plausible target; cache/stall metrics are in the full section dump.
- Commit/control now has 64 registers, 100% theoretical / 68.94% achieved occupancy, 10.57 active lanes. Its branches and serial per-cell loops still leave lanes inactive, but full-solve cost is now explicitly visible rather than hidden in the metal kernel.
- Pending/limit reductions and compaction have 100% theoretical occupancy and achieved 74.63/80.02/75.83%. Their combined NSYS cost remains tiny; they are not the first optimization target.
- Interval init and final output have 64/63 registers, achieved 68.63/68.41% occupancy, mostly streaming/state traffic. They run once each.
- Spectrum/H2 setup uses one block per kernel; low achieved occupancy reflects insufficient parallel work. Cross-section quadrature has uneven element/ion work. These are cold-start costs, excluded from the warmed solver table.
- Cell construction is initialization traffic. Descriptor construction/destruction, architecture query and lock initialization are tiny or one-time helpers; resource counts alone do not make them meaningful solver bottlenecks.

Previous pre-split metrics are preserved in git revision 3b5fb8f. Do not apply its metal/commit timing or lambda-instance mapping to the new separate commit kernel. Source-PC stall localization and per-arithmetic-pipeline counters were not captured; proposed optimizations remain hypotheses. No measured counter improvement for a future candidate is claimed here.

Artifacts: `results/venv-float64-profile.nsys-rep`, SQLite export, whole-trace CSVs, warm-only `venv-float64-kernel-breakdown.json`, and `venv-float64-ncu.ncu-repz` plus `venv-float64-ncu-details.txt`, `venv-float64-ncu-metrics.json`, `venv-float64-ncu-raw.csv`. Whole-trace CSVs include warmup; the warm-only JSON/table selects the second interval initialization onward.

## Retained trial: distribute metal normalization by ion

The element sums retain their original sequential order and float64 arithmetic.
Store those sums in team scratch, synchronize, then distribute the independent
ion divisions through the existing ion map. Previously each element occupied
one lane for both its sum and all its divisions. Scratch grows by 216 bytes;
team size remains 32. The Python binding enables the new path; legacy callers
retain the original path.

| Measurement | Control | Candidate | Time reduction |
| --- | ---: | ---: | ---: |
| Five-run solve median, same Python 3.14 environment | 12.388376 s | 11.985800 s | 3.25% |
| Five-run installed Python 3.13 median vs candidate | 12.267860 s | 11.985800 s | 2.30% |
| Warm metals GPU time, same Python 3.14 environment | 7.132435 s | 6.754642 s | 5.30% |

The same-environment control is the saved pre-trial binary, measured again
after the candidate. Its prior median was 12.372253 s. The installed control
is the newly profiled venv build above; interpreter/build differences mean
its comparison is supplementary. A final three-run candidate recheck after
the control/profile/tests gave 11.985468, 11.987000, 11.987407 s (median
11.987000 s), confirming the gain. Candidate profile: total solve 12.019237 s,
GPU sum 11.860777 s; cooling 2.114809 s, H/He 1.347165 s, prep 1.130793 s,
commit 0.476734 s. Compared with the prior Python 3.14 control profile,
these other phases are essentially unchanged. Metals still uses 68 registers.
Cooling uses 130 in this candidate build versus 128 in the installed build;
the pre-trial Python 3.14 build also used 130.

All four output arrays are bit-identical to the fresh installed control on
all 10,000 cells. Against the original reference, maximum relative error is
8.99e-12 for ions above 1e-4 and 2.88e-14 for temperature; all values finite.
Every reported arithmetic/state width is 8 bytes. CPU and CUDA binding tests:
18 passed each, three long sweeps excluded. **Retained** on the measured
dataset/GPU. No new candidate NCU occupancy/stall measurements are claimed;
the NCU table above describes the installed pre-normalization build.

Trial records: `results/metal-normalize.json`, `metal-normalize-baseline.json`,
`metal-normalize-control-recheck.json`, `metal-normalize-control-exact.json`,
`metal-normalize-tests.json`, `metal-normalize-recheck.json`, `metal-normalize-profile.json`, and
`metal-normalize-breakdown.json`; full NSYS trace/SQLite/CSVs use the
`metal-normalize-profile` prefix. Saved candidate binary:
`build/galaxy-gpu/metal-normalize-float64.cpython-314-x86_64-linux-gnu.so`.
The venv extension has not been replaced; rebuild it to use this change there.

## Further retained trial: cache every metal coefficient

Full coefficient caching is now enabled in the Python binding, including dust,
UVB, cosmic-ray/induced-UV, ISRF and all seven photon-group cross-sections.
Sixteen arrays have zero sentinels and forbidden endpoint coefficients are
zero. The active cached ion-update loop has no direct table accesses or
`j > 0` / `j < n_ions - 1` rate guards. Bounds checks remain in cache filling
and safe neighboring-fraction selection; the legacy fallback is preserved.
Runtime physics flags remain supported after the hard-coded-flags diagnostic.

| Trial, five warmed solves | Median s |
| --- | ---: |
| Fresh parallel-normalization parent | 12.038630 |
| Neighbor fractions cached once | 12.000238 |
| Neighbor fractions + four zero-sentinel rate caches | 12.000839 |
| Complete caches, dynamic flags, first build | 11.450372 |
| Complete caches, hard-coded true flags | 11.487743 |
| Final rebuilt complete-cache implementation, dynamic flags | 11.482720 |

The first two small trials do not show significant standalone gains. Complete
caching reduces final solve time **4.62% (1.04841x)** versus the fresh parent;
the earlier parent median 11.987000 s gives a 4.21% reduction. Static flags
give no additional gain and are restored to runtime values. Final SHA
`531b7855f3bce024e517886b8fcde6aa3d0104038a2a054600fb32dbf7368ee2`.

Warm NSYS solve **11.539300 s**, GPU sum **11.343592 s**:

| Part | Parent GPU s | Complete-cache GPU s | Registers in complete-cache build |
| --- | ---: | ---: | ---: |
| Metals | 6.754642 | 6.217209 | 90 |
| Cooling rates | 2.114809 | 2.116312 | 130 |
| HHe/cooling update | 1.347165 | 1.350590 | 92 |
| Prep/radiation | 1.130793 | 1.138260 | 64 |
| Commit/control | 0.476734 | 0.484474 | 64 |

Metal time drops **7.96%**; each main phase still launches 1,505 times.
The extra caches increase metal registers 68→90 and dynamic shared memory
3,031→8,183 bytes/block. These resource increases are included in the
measured timing; achieved occupancy/stall effects need a new NCU capture.
The installed-build NCU table above does not describe this candidate.

Every arithmetic/state width remains 8 bytes. All four output arrays are
bit-identical to the parent for all 10,000 cells, including the final rebuild;
the original-reference 1e-5 ion/temperature gate passes. Eight configurations
on 64 spread-out cells (all six flags on, each off separately, all off) are
bit-identical too, verified again on the final rebuild. CPU/CUDA regressions:
18 passed each; three long sweeps excluded each.

Full measurements/patches and trial decisions are recorded in GPU_TRIALS.md.
`results/metal-all-cache-{final,profile,breakdown,tests}.json`, NSYS trace,
SQLite and CSVs record the final build; `metal-cache-control-exact.json` and
`metal-cache-flags-final.json` record exact comparisons. The accepted-trials
PDF now includes the three recent metal optimizations. The installed venv
is unchanged; the source and `build/galaxy-gpu/_rtz…so` contain this change.
