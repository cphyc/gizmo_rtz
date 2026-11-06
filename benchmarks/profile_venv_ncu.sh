#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
unset PYTHONPATH
exec /usr/bin/ncu \
  --kernel-name-base demangled \
  --kernel-id '::regex:.*:1' \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
  --force-overwrite --export "$PWD/benchmarks/results/venv-float64-ncu" \
  "$PWD/.venv/bin/python" benchmarks/galaxy_gpu.py \
  --installed-package \
  --extension "$PWD/.venv/lib/python3.13/site-packages/rtz/_rtz.cpython-313-x86_64-linux-gnu.so" \
  --output "$PWD/benchmarks/results/venv-float64-ncu-run" \
  --require-float64 --warmups 0 --repetitions 1
