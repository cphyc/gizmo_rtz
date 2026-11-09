#pragma once

// Included after types.hpp, also inside prism_gpu by dyablo_api_GPU.cpp.
// Sparse storage: four coefficient vectors, then old/x/dx/pivot vectors.
// Each element contributes a tridiagonal block; charge exchange couples these
// blocks only to the two hydrogen stages. Eliminating HI by conservation leaves
// a scalar hydrogen border, solved by safeguarded Newton (a Schur complement).
constexpr int IMPLICIT_ION_VECTORS = 8;
constexpr int IMPLICIT_ION_CONTROL = 8;
constexpr int IMPLICIT_ION_WORK = IMPLICIT_ION_VECTORS * MAX_TOTAL_IONS + IMPLICIT_ION_CONTROL;

struct SerialIonExecutor {
    template<class F> KOKKOS_INLINE_FUNCTION void each(int n, F f) const {
        for (int i = 0; i < n; ++i) f(i);
    }
    template<class F> KOKKOS_INLINE_FUNCTION Real sum(int n, F f) const {
        Real result = 0;
        for (int i = 0; i < n; ++i) result += f(i);
        return result;
    }
    template<class F> KOKKOS_INLINE_FUNCTION void single(F f) const { f(); }
    template<class F> KOKKOS_INLINE_FUNCTION Real maximum(int n, F f) const {
        Real result = 0;
        for (int i = 0; i < n; ++i) result = rtz_math::fmax(result, f(i));
        return result;
    }
    KOKKOS_INLINE_FUNCTION void barrier() const {}
};

template<class Team> struct TeamIonExecutor {
    const Team& team;
    template<class F> KOKKOS_INLINE_FUNCTION void each(int n, F f) const {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n), f);
    }
    template<class F> KOKKOS_INLINE_FUNCTION Real sum(int n, F f) const {
        Real result = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n),
            [&](int i, Real& value) { value += f(i); }, result);
        return result;
    }
    template<class F> KOKKOS_INLINE_FUNCTION void single(F f) const {
        Kokkos::single(Kokkos::PerTeam(team), f);
    }
    template<class F> KOKKOS_INLINE_FUNCTION Real maximum(int n, F f) const {
        Real result = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n),
            [&](int i, Real& value) { value = rtz_math::fmax(value, f(i)); }, Kokkos::Max<Real>(result));
        return rtz_math::fmax(result, Real(0));
    }
    KOKKOS_INLINE_FUNCTION void barrier() const { team.team_barrier(); }
};

struct SparseIonSystem {
    Real *up, *down, *ct_up, *ct_down, *old, *x, *dx, *pivot, *control;
    KOKKOS_INLINE_FUNCTION explicit SparseIonSystem(Real* work)
        : up(work), down(up + MAX_TOTAL_IONS), ct_up(down + MAX_TOTAL_IONS),
          ct_down(ct_up + MAX_TOTAL_IONS), old(ct_down + MAX_TOTAL_IONS),
          x(old + MAX_TOTAL_IONS), dx(x + MAX_TOTAL_IONS),
          pivot(dx + MAX_TOTAL_IONS), control(pivot + MAX_TOTAL_IONS) {}

    // (I-dt*A_i(h)) x_i = old_i. Reuse the factorization to compute dx_i/dh.
    // The positive pivot recurrence avoids subtracting nearly equal stiff
    // diagonal terms in the usual Thomas elimination. No dense matrix/BLAS.
    KOKKOS_INLINE_FUNCTION bool block(int first, int n, Real dt, Real nh,
                                     Real h, Real neutral) const {
        Real complement = 1;
        Real mass = 0;
        for (int j = 0; j < n; ++j) {
            const int k = first + j;
            const Real u = up[k] + nh * h * ct_up[k];
            const Real d = down[k] + nh * neutral * ct_down[k];
            const Real remainder = Real(1) + dt * d * complement;
            pivot[k] = remainder + dt * u;
            if (!(pivot[k] > 0) || !Kokkos::isfinite(pivot[k])) return false;
            complement = remainder / pivot[k];
            x[k] = old[k];
            mass += old[k];
            if (j > 0) {
                const Real previous_up = up[k-1] + nh * h * ct_up[k-1];
                x[k] += dt * previous_up / pivot[k-1] * x[k-1];
            }
        }
        Real new_mass = 0;
        for (int j = n - 1; j >= 0; --j) {
            const int k = first + j;
            if (j + 1 < n) {
                const Real next_down = down[k+1] + nh * neutral * ct_down[k+1];
                x[k] += dt * next_down * x[k+1];
            }
            x[k] /= pivot[k];
            if (!(x[k] >= 0) || !Kokkos::isfinite(x[k])) return false;
            new_mass += x[k];
        }
        if (!(new_mass > 0)) return false;
        // Roundoff repair of the exactly conserved block mass.
        for (int j = 0; j < n; ++j) x[first+j] *= mass / new_mass;
        for (int j = 0; j < n; ++j) {
            const int k = first + j;
            Real rhs = (ct_down[k] - ct_up[k]) * x[k];
            if (j > 0) rhs += ct_up[k-1] * x[k-1];
            if (j + 1 < n) rhs -= ct_down[k+1] * x[k+1];
            dx[k] = dt * nh * rhs;
            if (j > 0) {
                const Real previous_up = up[k-1] + nh * h * ct_up[k-1];
                dx[k] += dt * previous_up / pivot[k-1] * dx[k-1];
            }
        }
        Real derivative_mass = 0;
        for (int j = n - 1; j >= 0; --j) {
            const int k = first + j;
            if (j + 1 < n) {
                const Real next_down = down[k+1] + nh * neutral * ct_down[k+1];
                dx[k] += dt * next_down * dx[k+1];
            }
            dx[k] /= pivot[k];
            derivative_mass += dx[k];
        }
        for (int j = 0; j < n; ++j) dx[first+j] -= x[first+j] * derivative_mass / mass;
        return true;
    }

    // Coefficients and ne/T are fixed; the H/metal charge-exchange products
    // are evaluated at the NEW populations, not frozen or omitted.
    template<class Executor>
    KOKKOS_INLINE_FUNCTION bool solve(const Executor& exec, const Element* elements,
                                      const CompactIonData& ions, Real dt,
                                      Real atomic_hydrogen,
                                      Real nonlinear_tolerance = sizeof(Real) == 8 ? Real(2e-13) : Real(2e-6)) const {
        const int hi = ions.offsets[1];
        const Real nh = ions.n_element_new[1];
        exec.single([&]() {
            control[0] = Real(0); // lower bracket
            control[1] = atomic_hydrogen; // upper bracket
            control[2] = (old[hi+1] + dt * up[hi] * atomic_hydrogen)
                       / (Real(1) + dt * (up[hi] + down[hi+1]));
            control[3] = 0; // converged
            control[4] = 0; // failed
            if (!(old[hi+1] >= 0 && old[hi+1] <= atomic_hydrogen)) control[4] = 1;
        });
        exec.barrier();
        for (int iteration = 0; iteration < 64; ++iteration) {
            if (control[4] != 0) break;
            const Real h = control[2], neutral = atomic_hydrogen - h;
            exec.each(MAX_ELEMENTS - 2, [&](int t) {
                const int i = t + 2;
                if (ions.n_element[i] == 0) return;
                const int first = ions.offsets[i];
                if (!block(first, elements[i].n_ions, dt, nh, h, neutral)) x[first] = Real(-1);
            });
            exec.barrier();
            const Real failed = exec.sum(MAX_ELEMENTS - 2, [&](int t) {
                const int i = t + 2;
                return ions.n_element[i] != 0 && x[ions.offsets[i]] < 0 ? Real(1) : Real(0);
            });
            // Form the hydrogen Schur residual from total charge balance.
            // CT fluxes cancel analytically, avoiding dt*large nearly equal
            // forward/reverse CT fluxes at stiff equilibrium.
            const Real charge = exec.sum(MAX_ELEMENTS - 2, [&](int t) {
                const int i = t + 2;
                Real value = 0;
                if (ions.n_element[i] != 0)
                    for (int j = 0; j < elements[i].n_ions; ++j) {
                        const int k = ions.offsets[i] + j;
                        value += Real(j) * (x[k] - old[k]) - dt * (up[k] - down[k]) * x[k];
                    }
                return value * ions.n_element_new[i] / nh;
            });
            const Real dcharge = exec.sum(MAX_ELEMENTS - 2, [&](int t) {
                const int i = t + 2;
                Real value = 0;
                if (ions.n_element[i] != 0)
                    for (int j = 0; j < elements[i].n_ions; ++j) {
                        const int k = ions.offsets[i] + j;
                        value += (Real(j) - dt * (up[k] - down[k])) * dx[k];
                    }
                return value * ions.n_element_new[i] / nh;
            });
            exec.single([&]() {
                const Real u = up[hi], d = down[hi+1];
                const Real f = h - old[hi+1] + charge - dt * (u * neutral - d * h);
                const Real derivative = Real(1) + dcharge + dt * (u + d);
                const Real scale = rtz_math::fmax(Real(1), rtz_math::fabs(derivative));
                const Real tolerance = nonlinear_tolerance;
                control[5] = Real(iteration + 1);
                if (failed != 0 || !Kokkos::isfinite(f)) control[4] = 1;
                else if (rtz_math::fabs(f) / scale <= tolerance) control[3] = 1;
                else {
                    if (f < 0) control[0] = h;
                    else control[1] = h;
                    const Real candidate = h - f / derivative;
                    control[2] = candidate > control[0] && candidate < control[1]
                               ? candidate : (control[0] + control[1]) / Real(2);
                }
            });
            exec.barrier();
            if (control[3] != 0 || control[4] != 0) break;
        }
        exec.single([&]() {
            x[hi] = atomic_hydrogen - control[2];
            x[hi+1] = control[2];
        });
        exec.barrier();
        return control[3] != 0 && control[4] == 0;
    }
};
