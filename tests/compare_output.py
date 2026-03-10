#!/usr/bin/env python3
"""Compare solver output against golden reference with tolerance checks."""
import sys
import numpy as np


def load_data(path):
    """Load space-separated output file, skipping non-numeric header lines."""
    from io import StringIO
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or (not line[0].isdigit() and line[0] != '-'):
                continue
            lines.append(line)
    return np.loadtxt(StringIO("\n".join(lines)))


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <golden_file> <test_file>", file=sys.stderr)
        sys.exit(2)

    golden_file, test_file = sys.argv[1], sys.argv[2]
    golden = load_data(golden_file)
    test = load_data(test_file)

    if golden.shape[0] != test.shape[0]:
        print(f"FAIL: Row count mismatch: golden={golden.shape[0]} test={test.shape[0]}")
        sys.exit(1)

    ncols = golden.shape[1]
    if ncols == 8:
        col_names = ["nH", "T", "T_mu", "mu", "iterations", "n_C", "nCO", "ddt"]
        check_cols = [0, 1, 2, 3, 5, 6]  # skip iterations(4) and ddt(7)
    elif ncols == 9:
        col_names = ["nH", "T", "T_mu", "mu", "iterations", "xHI", "xHII", "xH2", "ddt"]
        check_cols = [0, 1, 2, 3, 5, 6, 7]  # skip iterations(4) and ddt(8)
    else:
        # Generic: check all columns except the last (ddt) and col 4 (iterations)
        col_names = [f"col{i}" for i in range(ncols)]
        check_cols = [i for i in range(ncols) if i != 4 and i != ncols - 1]
    rtol = 1e-4  # relative tolerance

    max_rel_errors = {}
    failures = []

    for col in check_cols:
        name = col_names[col]
        g = golden[:, col]
        t = test[:, col]
        # Relative error with floor to avoid division by zero
        denom = np.maximum(np.abs(g), 1e-30)
        rel_err = np.abs(t - g) / denom
        max_rel = np.max(rel_err)
        max_idx = np.argmax(rel_err)
        max_rel_errors[name] = (max_rel, max_idx)

        if max_rel > rtol:
            failures.append(
                f"  {name}: max_rel_err={max_rel:.2e} at step {max_idx} "
                f"(golden={g[max_idx]:.6e} test={t[max_idx]:.6e})"
            )

    # Print summary
    print(f"Compared {golden.shape[0]} steps against golden reference")
    for name, (err, idx) in max_rel_errors.items():
        status = "OK" if err <= rtol else "FAIL"
        print(f"  {name:12s}: max_rel_err={err:.2e} [{status}]")

    if failures:
        print(f"\nFAIL: {len(failures)} column(s) exceeded tolerance {rtol}:")
        for f in failures:
            print(f)
        sys.exit(1)
    else:
        print(f"\nPASS: All columns within tolerance {rtol}")
        sys.exit(0)


if __name__ == "__main__":
    main()
