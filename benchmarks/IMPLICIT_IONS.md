# Opt-in implicit atomic ions

The legacy ion integrator remains the default. Select the experimental solver
with `rtz.solve(..., implicit_ions=True, x_pct_rule=0.2)`. The X-percent rule
defaults to 20% and applies to chemistry, temperature, and radiation updates,
including both ion integration paths.

The new atomic step solves backward Euler for H, He and the configured metals:

    x_new - x_old = dt * A(x_new) * x_new

Electron density, temperature, incident photons and secondary cosmic-ray
coefficients are frozen during the atomic solve. Charge exchange remains
bilinear in hydrogen and partner abundances even with fixed electron density.
It is evaluated at the new populations, rather than frozen at old hydrogen.

`implicit_ions.hpp` stores tridiagonal element blocks in coefficient vectors.
For a proposed HII fraction, each non-hydrogen block is solved by Thomas
elimination. The same factorization gives its derivative with respect to HII.
Hydrogen conservation eliminates HI; safeguarded Newton solves the remaining
scalar border (the sparse Schur complement). The hydrogen residual uses total
charge balance so stiff forward/reverse charge-exchange fluxes cancel
analytically. No dense matrix, BLAS or LAPACK is used.

The float64 physical ion step uses a scaled Newton residual tolerance of
1e-11; standalone sparse-core calls default to 2e-13. The float32 tolerance
remains 2e-6. These are nonlinear-solve tolerances, separate from `x_pct_rule`
and from time-integration error against another integrator.

CUDA assigns one thread to each active cell. Temporary sparse vectors are
interleaved by active-cell index, so neighboring threads access adjacent
workspace entries. The workspace is capped at 16,384 columns (119,537,664
bytes for float64) and reused across ordered batches; 10,000 cells use
72,960,000 bytes. CPU retains the team executor and contiguous scratch.
The same sparse-vector contract supports either stride, without changing
the equations. The CUDA legacy path does not allocate this workspace.

The pivot recurrence uses positive terms to avoid stiff diagonal subtraction.
Element mass is repaired for roundoff; abundances remain nonnegative. Molecular
evolution remains separate. The updated H2 abundance determines how much
hydrogen remains available to HI/HII. The temperature integrator and its
equations are unchanged; all coupled updates use the same `x_pct_rule`
acceptance threshold. Temperature output can nevertheless differ because the
chemistry changes cooling and molecular weight.

The X-percent rule is a timestep heuristic, not an error tolerance. Sparse
solve accuracy does not imply agreement with a different finite-step scheme.
The 20% experiment was explicitly accepted despite discrepancies from the
legacy reference; it is not claimed to pass the previous 1e-5 regression gate.

`_rtz.last_solve_statistics()` reports attempted, accepted and rejected steps
per cell, summed over requested intervals. `benchmarks/implicit_ion_trials.py`
records these distributions along with timings and legacy-reference errors.
`tests/implicit_ions.cpp` checks analytic two-state backward Euler, sparse
residuals, positivity, mass conservation, manufactured nonlinear charge
exchange, stiff detailed balance, hydrogen/H2 mass and serial/team agreement.
Measured trial results are recorded in `GPU_TRIALS.md`.
