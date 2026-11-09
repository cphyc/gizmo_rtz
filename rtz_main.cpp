#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <array>

#include <Kokkos_Core.hpp>

#include "dyablo_api_GPU.cpp"
using namespace prism_gpu;

// ---------------------------------------------------------------------------
// Single-cell subcycle driver
//
// dyablo_api_GPU.cpp deliberately does NOT define RTZ::solve_chemistry_and_cooling:
// the split-kernel driver (CoolingUpdate_PRISM_GPU.cpp) owns the subcycle loop on
// the host and launches each physics step as its own kernel. This is that same
// step sequence fused back into one thread -- the shape of that driver's
// PRISM_tail kernel, plus the preamble/epilogue of the reference
// RTZ::solve_chemistry_and_cooling and the equilibrium exit conditions of
// subcycle_chemistry (dyablo_api.cpp) -- so the standalone tests keep a
// single-cell entry point.
//
// total_dt <= 0 runs to equilibrium (conv_its consecutive successful commits),
// otherwise the cell is integrated to total_dt.
// ---------------------------------------------------------------------------
template <bool constant_temperature, bool include_H2, bool include_CO,
          bool rt_advect, bool include_self_shielding>
KOKKOS_INLINE_FUNCTION
void solve_chemistry_and_cooling_split(
    const Real TK,                  // T/mu on entry (T when constant_temperature)
    const Real dx,                  // cell size [cm], also the self-shielding length
    const Real aexp,
    const Real total_dt,            // [s]; <= 0 -> run to equilibrium
    const Element* elements,
    const CompactTabulatedData& tabData,
    const Real UV_background_G0,
    const Real primary_cr_rate,
    CompactIonData& n_and_ion_fracs,
    Real& nCO,
    std::array<Real, N_GROUPS>& N_PHOT,
    Array2D_RT& F_PHOT,
    Real& out_Tmu,
    Real& out_mu,
    const int conv_its,
    const int max_its,
    int& total_iter_reached,
    const PhysicsFlags flags,
    const bool rt_smooth,
    const std::array<Real, N_GROUPS>& dNpdt,
    const Array2D_RT& dFpdt,
    // Hold the photon field at its incident value for the whole solve instead of
    // letting the cell deplete it. Required to run to equilibrium at a fixed G0:
    // with depletion on, the subcycle just drains the reservoir and every G0
    // converges to the same unilluminated answer.
    const bool static_radiation = false,
    const bool implicit_ions = false,
    const Real ion_change_limit = Real(0.2))
{
    // --- preamble of RTZ::solve_chemistry_and_cooling ---
    const Real dust_ratio = get_dust_mass_and_depletion(
        elements, n_and_ion_fracs.n_element[8], n_and_ion_fracs.n_element[1], nCO);
    const Real loc_rho = get_rho(elements, n_and_ion_fracs);
    apply_dust_depletion(elements, dust_ratio, n_and_ion_fracs, nCO);

    // --- entry of subcycle_chemistry ---
    Real ss_factor = 1.0;
    if constexpr (include_self_shielding)
        ss_factor = exp(-1.0 * n_and_ion_fracs[1].n_element / 1E-2);

    SubcycleState state{};
    state.success_counter = 0;
    state.Tmu_old = TK;
    state.Tmu_new = TK;
    state.ne_initial = get_ne(elements, n_and_ion_fracs, false);

    // Reference initial subcycle timestep: 10^4 yr, or the whole step when
    // integrating a fixed dt (try to take it in one go).
    Real ddt = (total_dt > 0.0) ? total_dt : 10000.0 * 365.25 * 60.0 * 60.0;
    Real total_time = 0.0;
    int iterations = 0;

    std::array<Real, N_GROUPS> N_phot_new{};
    std::array<Real, N_GROUPS> phot_att{};
    Real cool_rate_pair[2] = {0.0, 0.0};

    Real implicit_work[IMPLICIT_ION_WORK];
    uint16_t all_ion_map[MAX_TOTAL_IONS];
    int all_ion_count = 0;
    for (int i = 1; i < MAX_ELEMENTS; ++i) {
        if (n_and_ion_fracs.n_element[i] == 0) continue;
        for (int j = 0; j < elements[i].n_ions; ++j)
            all_ion_map[all_ion_count++] = static_cast<uint16_t>((i << 8) | j);
    }
    while (true) {
        iterations += 1;

        subcycle_prep_iteration<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            elements, n_and_ion_fracs, nCO, total_dt, total_time,
            primary_cr_rate, ddt, state);

        // rad_residual_floor = 0 (and a null reference field) keeps the
        // reference physics: every photon group gates the timestep.
        radiation_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            ddt, elements, n_and_ion_fracs, tabData, nCO, dx, dust_ratio,
            N_PHOT, N_phot_new, phot_att, state,
            0.0, nullptr, rt_smooth, dNpdt, static_radiation);

        // Two evaluations of all_cooling (at T and 1.001*T) for the Newton derivative
        cooling_rate_lane<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            0, cool_rate_pair, aexp, elements, n_and_ion_fracs, tabData, flags,
            nCO, dust_ratio, primary_cr_rate, ss_factor, UV_background_G0,
            N_phot_new, state);
        cooling_rate_lane<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            1, cool_rate_pair, aexp, elements, n_and_ion_fracs, tabData, flags,
            nCO, dust_ratio, primary_cr_rate, ss_factor, UV_background_G0,
            N_phot_new, state);
        cooling_update<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            cool_rate_pair, ddt, loc_rho, state);

        molecular_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            ddt, elements, n_and_ion_fracs, tabData, nCO, dust_ratio,
            UV_background_G0, N_phot_new, state, implicit_ions ? ion_change_limit : X_PCT_RULE);

        if (implicit_ions) {
            implicit_ions_step<include_H2, rt_advect>(SerialIonExecutor{}, ddt, elements,
                n_and_ion_fracs, tabData, flags, dust_ratio, UV_background_G0, ss_factor,
                N_phot_new, all_ion_map, all_ion_count, implicit_work, state, ion_change_limit);
        } else {
        HandHe_ions_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            ddt, elements, n_and_ion_fracs, tabData, flags, dust_ratio,
            UV_background_G0, primary_cr_rate, ss_factor, N_phot_new, state);
        metal_ions_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            ddt, elements, n_and_ion_fracs, tabData, flags, dust_ratio,
            UV_background_G0, primary_cr_rate, ss_factor, N_phot_new, state);

        }
        subcycle_commit_and_control<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            elements, tabData, n_and_ion_fracs, nCO, N_PHOT, F_PHOT,
            N_phot_new, phot_att, ddt, total_time, state,
            rt_smooth, dFpdt);

        // Exit conditions of subcycle_chemistry
        if (total_dt > 0.0 && fabs(total_time - total_dt) / total_dt < 1e-6) break;
        if (state.success_counter > conv_its) break;
        if (iterations > max_its) break;
    }

    //printf("N_PHOT[1] after subcycle: %f\n", N_PHOT[1]);

    total_iter_reached = iterations;

    // --- epilogue of RTZ::solve_chemistry_and_cooling ---
    const Real ne_final = get_ne(elements, n_and_ion_fracs, false);
    out_mu  = get_mu(ne_final, elements, n_and_ion_fracs, false, include_H2, nCO);
    out_Tmu = state.Tmu_old;

    remove_dust_depletion(elements, dust_ratio, n_and_ion_fracs, nCO);
}

// Common initialization: elements + depletion + ion fractions
static void setup_elements(std::array<Element, MAX_ELEMENTS> &elements,
                           std::array<Real, MAX_ELEMENTS> &nelements_loc,
                           Real* xions_loc,
                           Real rho,
                           Real metallicity,
                           bool include_H2)
{
    initialize_elements(elements, include_H2);

    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;
        elements[i].depletion = 1.0;
    }

    std::array<ParticleIonData, MAX_ELEMENTS> n_and_ion_fracs{};
    initialize_ion_fracs(elements, n_and_ion_fracs, rho, metallicity, 2);

    int iion = 0;
    for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
        if (elements[ii].atomic_number < 1) continue;
        nelements_loc[ii] = n_and_ion_fracs[ii].n_element;
        for (int jj = 0; jj < elements[ii].n_ions + elements[ii].n_mol; jj++)
            xions_loc[iion++] = n_and_ion_fracs[ii].ion_fracs[jj];
    }
}

struct StepResult {
    Real nH, T_actual, Tmu, mu;
    Real xHI, xHII, xH2;
    Real xHeI;
    Real n_C, nCO;
    Real total_iter;
};

template <bool constant_temperature, bool include_H2, bool include_CO,
          bool rt_advect, bool include_self_shielding>
void solve_one_step(const RTZ<constant_temperature, include_H2, include_CO,
                              rt_advect, include_self_shielding> &rtz_solver,
                    Kokkos::View<Real[MAX_ELEMENTS]> &d_nel,
                    Kokkos::View<Real[MAX_TOTAL_IONS]> &d_xions,
                    Kokkos::View<Real[1]> &d_TK,
                    Kokkos::View<Real[1]> &d_nCO,
                    Real aexp, PhysicsFlags flags,
                    int conv_its, int max_its,
                    // Incident photon number density per group [cm^-3]. Held fixed
                    // for the whole solve (static external field).
                    std::array<Real, N_GROUPS> N_PHOT_in,
                    Kokkos::View<StepResult[1]> &d_result)
{
    // RTZ's accessors are host-only, so pull what the kernel needs out here
    // and let the lambda capture the plain values / Views by copy.
    auto elements_d = rtz_solver.elements_d;
    const auto tabData = rtz_solver.get_tabData();
    const int* ion_offsets = tabData.network.offsets.data();
    const Real UV_G0 = rtz_solver.get_UV_background_G0();
    const Real primary_cr_rate = rtz_solver.get_generic_cosmic_ray_ionization_rate();

    Kokkos::parallel_for("solve_step", 1, KOKKOS_LAMBDA(int) {
        // Build CompactIonData from views
        const Element* elements = elements_d.data();
        CompactIonData n_and_ion_fracs{};
        n_and_ion_fracs.offsets = ion_offsets;

        int iion = 0;
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (elements[i].atomic_number < 1) continue;
            Real nel = d_nel(i);
            // if (nel == 0.0) continue;
            n_and_ion_fracs.n_element[i] = nel;
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++) {
                n_and_ion_fracs[i].ion_fracs[j] = d_xions(iion);
                iion++;
            }
        }

        Real TK = d_TK(0);
        Real nCO = d_nCO(0);
        Real out_Tmu, out_mu;
        std::array<Real, N_GROUPS> N_PHOT = N_PHOT_in;
        Array2D_RT F_PHOT{};

        int total_iter_reached = 0;

        const Real dx = 4.4 * 3.086e18; // 4.4 pc
        solve_chemistry_and_cooling_split<constant_temperature, include_H2, include_CO,
                                          rt_advect, include_self_shielding>(
            TK, dx, aexp, -1.0,
            elements, tabData, UV_G0, primary_cr_rate,
            n_and_ion_fracs, nCO,
            N_PHOT, F_PHOT,
            out_Tmu, out_mu,
            conv_its, max_its, total_iter_reached, flags,
            false, N_PHOT, F_PHOT,
            true);  // static external radiation field

        // Write results back to views. The element loop must stay in lockstep
        // with the read loop above: skipping an element here without advancing
        // iion would shift every later element's ion fractions by its ion count.
        d_nCO(0) = nCO;
        d_TK(0) = out_Tmu;  // carry T/mu forward, as the ion fractions already are
        iion = 0;
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (elements[i].atomic_number < 1) continue;
            d_nel(i) = n_and_ion_fracs.n_element[i];
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++) {
                d_xions(iion) = n_and_ion_fracs[i].ion_fracs[j];
                iion++;
            }
        }

        d_result(0).nH = n_and_ion_fracs.n_element[1];
        d_result(0).T_actual = out_Tmu * out_mu;
        d_result(0).Tmu = out_Tmu;
        d_result(0).mu = out_mu;
        d_result(0).xHI = n_and_ion_fracs[1].ion_fracs[0];
        d_result(0).xHII = n_and_ion_fracs[1].ion_fracs[1];
        d_result(0).xH2 = (elements[1].n_mol > 0) ? n_and_ion_fracs[1].ion_fracs[2] : 0.0;
        d_result(0).xHeI = n_and_ion_fracs[2].ion_fracs[0];
        d_result(0).n_C = n_and_ion_fracs.n_element[6];
        d_result(0).nCO = nCO;
        d_result(0).total_iter = (Real)total_iter_reached;
    });
}

// G0_habing: strength of the incident FUV field in Habing units, i.e. the value
// radiation_step will compute for state.G0_local. The field is held static, so
// this is the equilibrium curve's control parameter.
// continue_T: how the temperature is seeded at each density step.
//   false (default) -- reseed from Tmu_init at every density. Points are
//                      independent and the curve is reproducible and
//                      path-independent.
//   true            -- carry the previous density's equilibrium T forward.
//                      Cheaper, but it traces a continuation path and so shows
//                      hysteresis (and can freeze into a cold state) wherever
//                      the thermal equilibrium is bistable.
static int run_ism_test(Real G0_habing, bool continue_T)
{
    Real redshift = 0.0;
    Real aexp = 1.0 / (1.0 + redshift);
    Real metallicity = 1E-1;
    int NT = 101;
    const int conv_its = 20000;
    const int max_its  = 40000;
    const Real Tmu_init = 1e5;

    const bool include_H2 = true;

    RTZ<false, include_H2, true, true, false> rtz_solver("../data");
    rtz_solver.set_UV_background_G0(0);
    // MUST precede need_to_update_cross_sections: update_cross_sections bakes
    // C_CGS * c_frac into cs_ph / cs_dust, and c_frac (a Kokkos::View) is
    // zero-initialised. Setting it afterwards leaves every photoionization and
    // dust-absorption rate constant at exactly zero.
    rtz_solver.set_reduced_speed_of_light_factor(1.0);
    rtz_solver.need_to_update_UVB(redshift);

    PhysicsFlags flags;
    flags.include_HM12_UVB = false;

    // Photon groups: 5.6-11.2 eV, 11.2-13.6 eV, 13.6-15.2 eV, 15.2-24.5 eV, 24.5-54.4 eV, 54.4-500 eV
    std::array<Real, N_GROUPS> E_min{0.0}, E_max{0.0};
    std::array<Real, 7> lower = {5.6, 11.2, 13.6, 15.2, 24.5, 54.4, 500.0};
    for (size_t g = 0; g < lower.size() - 1; g++) {
        E_min[g] = lower[g];
        E_max[g] = lower[g + 1];
    }
    rtz_solver.set_photon_groups(E_min, E_max);
    const Real T_blackbody = 1.0e5;
    rtz_solver.need_to_update_cross_sections(T_blackbody);

    // Incident FUV field: fix the spectral shape (3:1 between the 5.6-11.2 and
    // 11.2-13.6 eV bins), then normalise it so that the G0_local radiation_step
    // derives from it equals the requested value in Habing units.
    std::array<Real, N_GROUPS> N_PHOT_in{};
    N_PHOT_in[0] = 3.0;
    N_PHOT_in[1] = 1.0;
    {
        const auto tab = rtz_solver.get_tabData();
        auto h_ge = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tab.group_energy);
        auto h_cf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tab.c_frac);
        const Real G_conv = 1.6E-3; // Habing 1968 [erg/s/cm2]
        Real G0_per_unit = 0.0;
        for (int g = 0; g < N_GROUPS; g++)
            if (h_ge(g) > 5.6 && h_ge(g) < 13.6)
                G0_per_unit += N_PHOT_in[g] * (C_CGS * h_cf()) * h_ge(g) * EV_2_ERG / G_conv;
        const Real scale = (G0_per_unit > 0.0) ? G0_habing / G0_per_unit : 0.0;
        for (int g = 0; g < N_GROUPS; g++) N_PHOT_in[g] *= scale;
        fprintf(stderr, "ISM test: G0 = %.6g Habing -> N_PHOT = [%.17g, %.17g] cm^-3\n",
                G0_habing, N_PHOT_in[0], N_PHOT_in[1]);
    }

    std::array<Element, MAX_ELEMENTS> elements;
    std::array<Real, MAX_ELEMENTS> nelements_loc{};
    Real xions_loc[MAX_TOTAL_IONS] = {};

    Real rho_old = pow(10.0, -3.0);
    setup_elements(elements, nelements_loc, xions_loc, rho_old, metallicity, include_H2);

    // Compute total ion count
    int n_total_ions = 0;
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;
        n_total_ions += elements[i].n_ions + elements[i].n_mol;
    }

    // Device views for mutable state
    Kokkos::View<Real[MAX_ELEMENTS]> d_nel("nel");
    Kokkos::View<Real[MAX_TOTAL_IONS]> d_xions("xions");
    Kokkos::View<Real[1]> d_TK("TK");
    Kokkos::View<Real[1]> d_nCO("nCO");
    Kokkos::View<StepResult[1]> d_result("result");

    // Host mirrors
    auto h_nel = Kokkos::create_mirror_view(d_nel);
    auto h_xions = Kokkos::create_mirror_view(d_xions);
    auto h_TK = Kokkos::create_mirror_view(d_TK);
    auto h_nCO = Kokkos::create_mirror_view(d_nCO);
    auto h_result = Kokkos::create_mirror_view(d_result);

    for (int i = 0; i < MAX_ELEMENTS; i++) h_nel(i) = nelements_loc[i];
    for (int i = 0; i < n_total_ions; i++) h_xions(i) = xions_loc[i];
    h_TK(0) = Tmu_init;
    h_nCO(0) = 0.0;
    Kokkos::deep_copy(d_nel, h_nel);
    Kokkos::deep_copy(d_xions, h_xions);
    Kokkos::deep_copy(d_TK, h_TK);
    Kokkos::deep_copy(d_nCO, h_nCO);

    printf("# ISM equilibrium curve, G0 = %.6g Habing, Z = %.3g Zsun, %s\n",
           G0_habing, metallicity,
           continue_T ? "T continued between densities" : "T reseeded per density");
    printf("# nH  T_K  Tmu  mu  n_C  nCO  xHI  xHII  xH2  n_iter\n");

    for (int i = 0; i < NT; i++) {
        Real rho = pow(10.0, -3.0 + (Real)i * 7.0 / (NT - 1));

        // Recompute element densities from scratch at each density step
        Kokkos::deep_copy(h_nel, d_nel);
        for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
            if (elements[ii].atomic_number < 1) continue;
            if (ii < 3) {
                h_nel(ii) = rho * elements[ii].z_solar;                // TOTAL (H, He)
            } else {
                h_nel(ii) = rho * elements[ii].z_solar * metallicity;  // TOTAL (metals)
            }
        }
        rho_old = rho;
        Kokkos::deep_copy(d_nel, h_nel);

        // Reset nCO at each density step
        h_nCO(0) = 0.0;
        Kokkos::deep_copy(d_nCO, h_nCO);

        if (!continue_T) {
            h_TK(0) = Tmu_init;
            Kokkos::deep_copy(d_TK, h_TK);
        }

        solve_one_step(rtz_solver, d_nel, d_xions, d_TK, d_nCO,
                       aexp, flags, conv_its, max_its, N_PHOT_in, d_result);

        Kokkos::deep_copy(h_result, d_result);
        auto &r = h_result(0);
        // The subcycle exits on max_its whether or not it converged, so a point
        // that ran out of iterations is otherwise indistinguishable from a good one.
        if (r.total_iter > (Real)max_its)
            fprintf(stderr, "WARNING: nH=%.4e hit max_its=%d without converging\n",
                    r.nH, max_its);
        printf("%.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.0f\n",
               r.nH, r.T_actual, r.Tmu, r.mu,
               r.n_C, r.nCO, r.xHI, r.xHII, r.xH2, r.total_iter);
    }
    return 0;
}

static int run_cie_test()
{
    Real redshift = 0.0;
    Real aexp = 1.0 / (1.0 + redshift);
    Real metallicity = 1E-1;
    int NT = 101;
    bool write_output = false;

    if (write_output) {
        printf("Overwriting CIE curves...\n");
    }

    RTZ<true, true, false, false, false> rtz_solver("../data");
    rtz_solver.need_to_update_UVB(redshift);

    PhysicsFlags flags;
    flags.include_collisional_ionization = true;
    flags.include_photoionization = false;
    flags.include_cosmic_ray_ionization = false;
    flags.include_HM12_UVB = false;
    flags.include_dust_recombination = false;
    flags.include_charge_exchange = false;

    std::array<Element, MAX_ELEMENTS> elements;
    std::array<Real, MAX_ELEMENTS> nelements_loc{};
    Real xions_loc[MAX_TOTAL_IONS] = {};

    Real rho = 1e-2;
    setup_elements(elements, nelements_loc, xions_loc, rho, metallicity, true);

    // Compute total ion count
    int n_total_ions = 0;
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;
        n_total_ions += elements[i].n_ions + elements[i].n_mol;
    }

    // Device views
    Kokkos::View<Real[MAX_ELEMENTS]> d_nel("nel");
    Kokkos::View<Real[MAX_TOTAL_IONS]> d_xions("xions");
    Kokkos::View<Real[1]> d_TK("TK");
    Kokkos::View<Real[1]> d_nCO("nCO");
    Kokkos::View<StepResult[1]> d_result("result");

    auto h_nel = Kokkos::create_mirror_view(d_nel);
    auto h_xions = Kokkos::create_mirror_view(d_xions);
    auto h_TK = Kokkos::create_mirror_view(d_TK);
    auto h_nCO = Kokkos::create_mirror_view(d_nCO);
    auto h_result = Kokkos::create_mirror_view(d_result);

    for (int i = 0; i < MAX_ELEMENTS; i++) h_nel(i) = nelements_loc[i];
    for (int i = 0; i < n_total_ions; i++) h_xions(i) = xions_loc[i];
    h_nCO(0) = 0.0;
    Kokkos::deep_copy(d_nel, h_nel);
    Kokkos::deep_copy(d_xions, h_xions);
    Kokkos::deep_copy(d_nCO, h_nCO);

    auto element_name = [](int Z) -> const char* {
        switch (Z) {
            case 1:  return "H";
            case 2:  return "He";
            case 6:  return "C";
            case 7:  return "N";
            case 8:  return "O";
            case 10: return "Ne";
            case 12: return "Mg";
            case 14: return "Si";
            case 16: return "S";
            case 26: return "Fe";
            default: return "X";
        }
    };

    // One file per active element; columns are ionization states (+ H2 for H).
    FILE* ion_files[MAX_ELEMENTS] = {};
    if (write_output) {
        for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
            if (elements[ii].atomic_number < 1) continue;
            const char* name = element_name(elements[ii].atomic_number);
            char fname[256];
            snprintf(fname, sizeof(fname), "../data/CIE_init/cie_%s.dat", name);
            ion_files[ii] = fopen(fname, "w");
            fprintf(ion_files[ii], "# T_K");
            for (int j = 0; j < elements[ii].n_ions; j++)
                fprintf(ion_files[ii], " %s_%d", name, j + 1);
            if (ii == 1 && elements[1].n_mol > 0)
                fprintf(ion_files[ii], " H2");
            fprintf(ion_files[ii], "\n");
        }
    }

    for (int i = 0; i < NT; i++) {
        // go from 1e3 to 1e9 K
        h_TK(0) = pow(10.0, (((Real)i / (Real)(NT - 1)) * (9.0 - 3.0)) + 3.0);
        Kokkos::deep_copy(d_TK, h_TK);

        auto t_start = std::chrono::high_resolution_clock::now();
        // rt_advect=false for this solver, so radiation_step is a no-op and the
        // incident field is unused; pass zeros.
        solve_one_step(rtz_solver, d_nel, d_xions, d_TK, d_nCO,
                       aexp, flags, 20000, 40000,
                       std::array<Real, N_GROUPS>{}, d_result);
        auto t_end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

        Kokkos::deep_copy(h_result, d_result);
        Kokkos::deep_copy(h_xions, d_xions);

        fprintf(stderr, "step %d/%d Tmu=%e solver_time=%ld ms\n", i+1, NT, h_TK(0), elapsed);

        int ion_index = 0;
        printf("%.17e", h_TK(0));
        for (int element_index = 1; element_index < MAX_ELEMENTS; element_index++) {
            if (elements[element_index].atomic_number < 1) continue;
            for (int stage = 0; stage < elements[element_index].n_ions; stage++)
                printf(" %.17e", h_xions(ion_index++));
            ion_index += elements[element_index].n_mol;
        }
        printf("\n");

        if (write_output) {
            int counter = 0;
            for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
                if (elements[ii].atomic_number < 1) continue;
                fprintf(ion_files[ii], "%.7e", h_TK(0));
                for (int j = 0; j < elements[ii].n_ions + elements[ii].n_mol; j++) {
                    fprintf(ion_files[ii], " %.7e", h_xions(counter));
                    counter++;
                }
                fprintf(ion_files[ii], "\n");
            }
        }
    }

    if (write_output) {
        for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
            if (ion_files[ii]) fclose(ion_files[ii]);
        }
    }
    return 0;
}


// One coupled photoionization+photoheating sub-step with a FIXED incident photon
// field (constant illumination). Advances T and the ion fractions over total_dt_s.
// The incident photon density d_Nphot is left unchanged (the cell is continuously
// re-supplied by the source), so we test the steady photoheating, not a one-shot
// reservoir. d_result(0).total_iter is repurposed to report the solver iteration count.
template <bool constant_temperature, bool include_H2, bool include_CO,
          bool rt_advect, bool include_self_shielding>
void solve_photoheat_step(const RTZ<constant_temperature, include_H2, include_CO,
                                    rt_advect, include_self_shielding> &rtz_solver,
                          Kokkos::View<Real[MAX_ELEMENTS]> &d_nel,
                          Kokkos::View<Real[MAX_TOTAL_IONS]> &d_xions,
                          Kokkos::View<Real[1]> &d_TK,
                          Kokkos::View<Real[N_GROUPS]> &d_Nphot,
                          Real aexp, Real total_dt_s,
                          PhysicsFlags flags,
                          int conv_its, int max_its,
                          Kokkos::View<StepResult[1]> &d_result)
{
    auto elements_d = rtz_solver.elements_d;
    const auto tabData = rtz_solver.get_tabData();
    const int* ion_offsets = tabData.network.offsets.data();
    const Real UV_G0 = rtz_solver.get_UV_background_G0();
    const Real primary_cr_rate = rtz_solver.get_generic_cosmic_ray_ionization_rate();

    Kokkos::parallel_for("photoheat_step", 1, KOKKOS_LAMBDA(int) {
        const Element* elements = elements_d.data();
        CompactIonData n_and_ion_fracs{};
        n_and_ion_fracs.offsets = ion_offsets;

        int iion = 0;
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (elements[i].atomic_number < 1) continue;
            n_and_ion_fracs.n_element[i] = d_nel(i);
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++)
                n_and_ion_fracs[i].ion_fracs[j] = d_xions(iion++);
        }

        Real TK = d_TK(0);
        Real nCO = 0.0;
        Real out_Tmu, out_mu;
        std::array<Real, N_GROUPS> N_PHOT;
        for (int g = 0; g < N_GROUPS; g++) N_PHOT[g] = d_Nphot(g);  // fixed incident field
        Array2D_RT F_PHOT{};
        int total_iter_reached = 0;

        const Real dx = 4 * 3.086e18; // 4 pc
        solve_chemistry_and_cooling_split<constant_temperature, include_H2, include_CO,
                                          rt_advect, include_self_shielding>(
            TK, dx, aexp, total_dt_s,
            elements, tabData, UV_G0, primary_cr_rate,
            n_and_ion_fracs, nCO,
            N_PHOT, F_PHOT,
            out_Tmu, out_mu,
            conv_its, max_its, total_iter_reached, flags,
            false, N_PHOT, F_PHOT,
            true);  // fixed incident field: the source keeps re-supplying the cell

        // Advance the state. NOTE: the solver's temperature I/O is T/mu (Tmu), NOT T,
        // so we must carry Tmu across steps (feed back out_Tmu, not out_Tmu*out_mu).
        // Feeding back the physical T would re-multiply by mu every step -> spurious
        // ~mu^n "heating" independent of rt_advect.
        d_TK(0) = out_Tmu;
        iion = 0;
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (elements[i].atomic_number < 1) continue;
            d_nel(i) = n_and_ion_fracs.n_element[i];
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++)
                d_xions(iion++) = n_and_ion_fracs[i].ion_fracs[j];
        }

        d_result(0).nH = n_and_ion_fracs.n_element[1];
        d_result(0).T_actual = out_Tmu * out_mu;
        d_result(0).Tmu = out_Tmu;
        d_result(0).mu = out_mu;
        d_result(0).xHI = n_and_ion_fracs[1].ion_fracs[0];
        d_result(0).xHII = n_and_ion_fracs[1].ion_fracs[1];
        d_result(0).xHeI = n_and_ion_fracs[2].ion_fracs[0];
        d_result(0).nCO = nCO;
        d_result(0).total_iter = (Real)total_iter_reached;  // report solver iterations
    });
}

// Photoheating test: a single neutral H+He cell at T=100 K is held under a fixed
// incident ionizing photon field and integrated forward in time. We watch T(t) to
// see whether it settles near photoionization equilibrium (~1-2e4 K) or runs away.
static int run_photoheating_test()
{
    const Real redshift = 0.0;
    const Real aexp = 1.0;
    const Real metallicity = 0.0;        // pure H+He (Stromgren-like)
    const bool   include_H2 = false;

    // constant_temperature=false, include_H2=false, include_CO=false,
    // rt_advect=true, include_self_shielding=true  (matches the PRISM config)
    RTZ<false, false, false, true, false> rtz_solver("../data");
    rtz_solver.need_to_update_UVB(redshift);
    rtz_solver.set_reduced_speed_of_light_factor(1.0);   // full speed of light

    if (N_GROUPS != 6) {
        printf("Error: N_GROUPS must be 6 for the photoheating test. Current N_GROUPS=%d\n", N_GROUPS);
        return -1;
    }

    // Photon groups: 5.6-11.2 eV, 11.2-13.6 eV, 13.6-15.2 eV, 15.2-24.5 eV, 24.5-54.4 eV, 54.4-500 eV
    std::array<Real, N_GROUPS> E_min, E_max;
    std::array<Real, 7> lower = {5.6, 11.2, 13.6, 15.2, 24.5, 54.4, 500.0};
    for (int g = 0; g < N_GROUPS; g++) {
        E_min[g] = lower[g];
        E_max[g] = lower[g + 1];
    }
    rtz_solver.set_photon_groups(E_min, E_max);
    const Real T_blackbody = 1.0e5;
    rtz_solver.need_to_update_cross_sections(T_blackbody);

    PhysicsFlags flags;
    flags.include_collisional_ionization = true;
    flags.include_photoionization        = true;
    flags.include_cosmic_ray_ionization  = false;
    flags.include_HM12_UVB               = false;
    flags.include_dust_recombination     = false;
    flags.include_charge_exchange        = false;

    // ---------------- test-cell parameters (tweak these) ----------------
    const Real rho          = 1.0e-2;     // n_H ~ 100 cm^-3 (dense, like the hot cell)
    const Real T_init       = 5e3;     // K, cold neutral gas
    const Real dt_step_s    = 3e15;     // s per step
    const int    NT           = 1000;
    // Incident photon number density [cm^-3], per band. Hydrogen needs photons
    // above 13.6 eV, which is group 2 onward -- groups 0 and 1 are FUV
    // (5.6-13.6 eV) and cannot ionize H at all. With metallicity 0 there is no
    // dust and no metals either, so FUV in groups 0/1 does nothing whatsoever:
    // the flux has to go in the ionizing bands for this test to test anything.
    std::array<Real, N_GROUPS> Nphot0{};
    Nphot0[0] = 0;          // 5.6-11.2  eV  FUV, cannot ionize H
    Nphot0[1] = 0;          // 11.2-13.6 eV  FUV, cannot ionize H
    Nphot0[2] = 1e0;        // 13.6-15.2 eV  softest H-ionizing band
    Nphot0[3] = 0;          // 15.2-24.5 eV
    Nphot0[4] = 0;          // 24.5-54.4 eV
    Nphot0[5] = 0;          // 54.4-500  eV
    // --------------------------------------------------------------------

    std::array<Element, MAX_ELEMENTS> elements;
    std::array<Real, MAX_ELEMENTS> nelements_loc{};
    Real xions_loc[MAX_TOTAL_IONS] = {};
    setup_elements(elements, nelements_loc, xions_loc, rho, metallicity, include_H2);

    // Force a fully neutral initial state (ground state of every element).
    int n_total_ions = 0, iion = 0;
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;
        for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++)
            // Nearly neutral, but with a tiny ionization seed so n_e > 0.
            // A *fully* neutral start (n_e = 0) leaves the solver unable to
            // bootstrap, and it maxes out the iteration count without converging.
            xions_loc[iion++] = (j == 0) ? (1.0 - 1e-8) : (j == 1 ? 1e-8 : 0.0);
        n_total_ions += elements[i].n_ions + elements[i].n_mol;
    }

    Kokkos::View<Real[MAX_ELEMENTS]> d_nel("nel");
    Kokkos::View<Real[MAX_TOTAL_IONS]> d_xions("xions");
    Kokkos::View<Real[1]> d_TK("TK");
    Kokkos::View<Real[N_GROUPS]> d_Nphot("Nphot");
    Kokkos::View<StepResult[1]> d_result("result");

    auto h_nel = Kokkos::create_mirror_view(d_nel);
    auto h_xions = Kokkos::create_mirror_view(d_xions);
    auto h_TK = Kokkos::create_mirror_view(d_TK);
    auto h_Nphot = Kokkos::create_mirror_view(d_Nphot);
    auto h_result = Kokkos::create_mirror_view(d_result);

    for (int i = 0; i < MAX_ELEMENTS; i++) h_nel(i) = nelements_loc[i];
    for (int i = 0; i < n_total_ions; i++) h_xions(i) = xions_loc[i];
    // d_TK carries T/mu (Tmu), matching the solver's convention. Convert the desired
    // physical T_init using the neutral H+He mean molecular weight (mu ~ 1.24).
    const Real mu_neutral = 1.24;
    h_TK(0) = T_init / mu_neutral;
    for (int g = 0; g < N_GROUPS; g++) h_Nphot(g) = Nphot0[g];
    Kokkos::deep_copy(d_nel, h_nel);
    Kokkos::deep_copy(d_xions, h_xions);
    Kokkos::deep_copy(d_TK, h_TK);
    Kokkos::deep_copy(d_Nphot, h_Nphot);

    Real Nphot_ionizing = 0.0;
    for (int g = 2; g < N_GROUPS; g++) Nphot_ionizing += Nphot0[g];
    printf("# photoheating test: nH=%.3e cm^-3  T_init=%.1f K  U=N_ion/nH=%.3e\n",
           nelements_loc[1], T_init, Nphot_ionizing / nelements_loc[1]);
    printf("# bands [eV]:");
    for (int g = 0; g < N_GROUPS; g++) printf(" [%.1f-%.1f]", E_min[g], E_max[g]);
    printf("\n# step  time_yr  nH  T_K  xHI  xHII  xHeI  n_iter\n");

    Real t = 0.0;
    for (int i = 0; i < NT; i++) {
        solve_photoheat_step(rtz_solver, d_nel, d_xions, d_TK, d_Nphot,
                             aexp, dt_step_s,
                             flags, 200000, 4000000, d_result);
        t += dt_step_s * 3.17e-8;  // convert to yr
        Kokkos::deep_copy(h_result, d_result);
        auto &r = h_result(0);
        printf("%d %.0f %.1e %.6e %.6e %.6e %.6e %.0f\n",
               i, t, r.nH, r.T_actual, r.xHI, r.xHII, r.xHeI, r.total_iter);
    }
    return 0;
}



static void print_usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <test_name> [G0]\n", prog);
    fprintf(stderr, "  test_name: ISM, CIE, or PHOTOHEAT\n");
    fprintf(stderr, "  G0:        ISM only -- incident FUV field in Habing units\n");
    fprintf(stderr, "             (default 1.0). The field is held static, so this\n");
    fprintf(stderr, "             is the equilibrium curve's control parameter.\n");
    fprintf(stderr, "  [3rd arg]  ISM only -- 1 to carry T between density steps\n");
    fprintf(stderr, "             (continuation); default 0 = reseed each point.\n");
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *test_name = argv[1];
    int rc = 0;

    Kokkos::initialize(argc, argv);
    {
        if (strcmp(test_name, "ISM") == 0) {
            Real G0_habing = 1.0;
            if (argc > 2) G0_habing = atof(argv[2]);
            const bool continue_T = (argc > 3) && atoi(argv[3]) != 0;
            rc = run_ism_test(G0_habing, continue_T);
        } else if (strcmp(test_name, "CIE") == 0) {
            rc = run_cie_test();
        } else if (strcmp(test_name, "PHOTOHEAT") == 0) {
            rc = run_photoheating_test();
        } else {
            fprintf(stderr, "Unknown test: %s\n", test_name);
            print_usage(argv[0]);
            rc = 1;
        }
    }
    clear_ion_offset_cache();
    Kokkos::finalize();
    return rc;
}
