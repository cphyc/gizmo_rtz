"""
plot_comparison.py

Compares the reference ISM test output with the updated C++/Kokkos output.
Columns (0-indexed):
  0: nH   1: T_actual   2: T_mu   3: mu   4: iterations
  5: xHI  6: xHII       7: xH2    8: ddt
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF_FILE = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/cphyc/.copilot/session-state/e237bb89-206c-4135-af7f-a509f034f1bd/files/reference_output.txt"
)
NEW_FILE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/new_output.txt"
OUT_PNG  = sys.argv[3] if len(sys.argv) > 3 else "comparison.png"

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                continue  # skip header lines
    return np.array(rows)

ref = load(REF_FILE)
new = load(NEW_FILE)

assert ref.shape == new.shape, f"Shape mismatch: ref={ref.shape} new={new.shape}"

nH_ref = ref[:, 0];  T_ref  = ref[:, 1]
nH_new = new[:, 0];  T_new  = new[:, 1]
xHI_ref = ref[:, 5]; xHII_ref = ref[:, 6]; xH2_ref = ref[:, 7]
xHI_new = new[:, 5]; xHII_new = new[:, 6]; xH2_new = new[:, 7]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Reference vs C++/Kokkos — ISM equilibrium test", fontsize=13)

# --- Temperature ---
ax = axes[0, 0]
ax.loglog(nH_ref, T_ref, "k-",  lw=2,   label="reference")
ax.loglog(nH_new, T_new, "r--", lw=1.5, label="Kokkos")
ax.set_xlabel("nH [cm⁻³]"); ax.set_ylabel("T [K]")
ax.set_title("Temperature"); ax.legend()

# --- Relative T difference ---
ax = axes[0, 1]
rel_T = 2 * (T_ref - T_new) / (T_ref + T_new)
ax.semilogx(nH_ref, rel_T, "b-", lw=1.5)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("nH [cm⁻³]"); ax.set_ylabel("2(ref−new)/(ref+new)")
ax.set_title("Relative T difference")

# --- Ion fractions ---
ax = axes[1, 0]
ax.loglog(nH_ref, xHI_ref,  "b-",  lw=2,   label="xHI ref")
ax.loglog(nH_new, xHI_new,  "b--", lw=1.5, label="xHI new")
ax.loglog(nH_ref, xHII_ref, "r-",  lw=2,   label="xHII ref")
ax.loglog(nH_new, xHII_new, "r--", lw=1.5, label="xHII new")
ax.loglog(nH_ref, xH2_ref,  "g-",  lw=2,   label="xH2 ref")
ax.loglog(nH_new, xH2_new,  "g--", lw=1.5, label="xH2 new")
ax.set_xlabel("nH [cm⁻³]"); ax.set_ylabel("ion fraction")
ax.set_title("H ion fractions"); ax.legend(fontsize=8, ncol=2)

# --- Relative ion fraction differences ---
ax = axes[1, 1]
def reldiff(a, b):
    return 2 * (a - b) / (np.abs(a) + np.abs(b) + 1e-300)
ax.semilogx(nH_ref, reldiff(xHI_ref, xHI_new),   "b-",  lw=1.5, label="ΔxHI")
ax.semilogx(nH_ref, reldiff(xHII_ref, xHII_new),  "r-",  lw=1.5, label="ΔxHII")
ax.semilogx(nH_ref, reldiff(xH2_ref, xH2_new),    "g-",  lw=1.5, label="ΔxH2")
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("nH [cm⁻³]"); ax.set_ylabel("2(ref−new)/(ref+new)")
ax.set_title("Relative ion fraction differences"); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"Saved {OUT_PNG}")

# Numerical summary
max_T_err  = np.max(np.abs(rel_T))
max_xHI_err  = np.max(np.abs(reldiff(xHI_ref, xHI_new)))
max_xHII_err = np.max(np.abs(reldiff(xHII_ref, xHII_new)))
max_xH2_err  = np.max(np.abs(reldiff(xH2_ref, xH2_new)))
print(f"Max relative errors:  T={max_T_err:.2e}  xHI={max_xHI_err:.2e}  xHII={max_xHII_err:.2e}  xH2={max_xH2_err:.2e}")
